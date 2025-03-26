import torch
import triton
import triton.language as tl
from triton import Config

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    # calculate scale factor
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    # turn output to low percision dtype
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

def act_quant(x, block_size=64):
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'

    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1] // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s

@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)

def weight_dequant(x, s, block_size=64):
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'

    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y

fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        acc += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = acc.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)
    
def fp8_gemm(a, a_s, b, b_s):
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

@triton.jit
def rmsnorm_fwd_kernel(
    X,
    Y,
    W,
    Rstd,
    stride_ml,
    stride_n,
    L,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Setup for batched execution over M and L
    row = tl.program_id(0)
    batch = tl.program_id(1)

    # Calculate the base index for the current matrix slice
    base_idx = row * stride_ml + batch * stride_n
    Y += base_idx
    X += base_idx

    _rms = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _rms += a * a
    rms = tl.sqrt(tl.sum(_rms) / N + eps)

    # Store the reciprocal of the standard deviation
    tl.store(Rstd + row * L + batch, rms)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x / rms
        y = x_hat * W
        tl.store(Y + cols, y, mask=mask)

@triton.jit
def rmsnorm_bwd_kernel(
    input_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    grad_output_ptr: tl.pointer_type,
    input_row_stride: tl.uint32,
    grad_input_ptr: tl.pointer_type,
    grad_weight_accum_ptr: tl.pointer_type,
    num_elements: tl.uint32,
    eps: tl.float32,
    block_size: tl.constexpr,
):
    # Calculate the row index for this program instance
    row_idx = tl.program_id(0)

    # Create an array of offsets within the block
    offsets = tl.arange(0, block_size)

    # Calculate memory access ranges for the inputs and gradients
    input_offsets = row_idx * input_row_stride + offsets
    input_ptrs = input_ptr + input_offsets
    weight_ptrs = weight_ptr + offsets
    grad_output_offsets = grad_output_ptr + input_offsets

    # Create masks to handle cases where block size may exceed the number of elements
    valid_elements_mask = offsets < num_elements

    # Load input values, weights, and gradient outputs using the computed offsets and masks
    input_values = tl.load(input_ptrs, mask=valid_elements_mask, other=0)
    weights = tl.load(weight_ptrs, mask=valid_elements_mask, other=0)
    grad_outputs = tl.load(grad_output_offsets, mask=valid_elements_mask, other=0)

    # Compute the normalization factor from the input values
    norm_factor = tl.sqrt(tl.sum(input_values * input_values) / num_elements + eps)

    # Compute partial gradients with respect to weights
    grad_weight_partial = input_values * grad_outputs / norm_factor
    tl.store(
        grad_weight_accum_ptr + input_offsets,
        grad_weight_partial,
        mask=valid_elements_mask,
    )

    # Compute partial gradients with respect to input values
    grad_input_first_term = grad_outputs * weights / norm_factor
    grad_input_second_term = (
        tl.sum(input_values * grad_outputs * weights)
        * input_values
        / (num_elements * norm_factor * norm_factor * norm_factor)
    )
    grad_input_values = grad_input_first_term - grad_input_second_term
    tl.store(
        grad_input_ptr + input_offsets, grad_input_values, mask=valid_elements_mask
    )

class RMSNormTritonAutogradFuncClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, eps=1e-5):
        M, L, N = x.shape
        y = torch.empty_like(x, dtype=torch.float32, device=x.device)
        rstd = torch.empty(M * L, dtype=torch.float32, device=x.device)

        grid = (M, L)
        rmsnorm_fwd_kernel[grid](
            x, y, g, rstd, x.stride(0), x.stride(1), L, N, eps, BLOCK_SIZE=128
        )

        ctx.block_size = triton.next_power_of_2(N)
        ctx.save_for_backward(x, g)
        ctx.eps = eps
        ctx.N = N
        ctx.L = L

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        H = x.shape[-1]
        x_shape = x.shape

        x = x.view(-1, H)
        n_rows = x.shape[0]

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty_like(x)

        rmsnorm_bwd_kernel[(n_rows,)](
            x,
            weight,
            grad_output,
            x.stride(0),
            grad_x,
            partial_grad_weight,
            H,
            eps,
            num_warps=16,
            block_size=ctx.block_size,
        )
        return grad_x.view(*x_shape), partial_grad_weight.sum(dim=0)

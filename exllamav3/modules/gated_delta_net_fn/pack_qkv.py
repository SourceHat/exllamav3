"""Window-local compact Q/K/V preparation for scalar GDN prefill."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _pack_qkv_kernel(
    Q, K, V, Q_out, K_out, V_out,
    T,
    H: tl.constexpr, HV: tl.constexpr,
    DQ: tl.constexpr, DV: tl.constexpr,
    SQ: tl.constexpr, SK: tl.constexpr, SV: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    block = tl.program_id(0)
    q_blocks = tl.cdiv(T * H, BT)
    if block < 2 * q_blocks:
        is_k = block >= q_blocks
        rows = ((block % q_blocks) * BT + tl.arange(0, BT)).to(tl.int64)
        cols = tl.arange(0, BK)
        source = tl.where(is_k, K, Q)
        dest = tl.where(is_k, K_out, Q_out)
        stride = tl.where(is_k, SK, SQ)
        offsets = (rows // H)[:, None] * stride + (rows % H)[:, None] * DQ + cols[None, :]
        mask = (rows[:, None] < T * H) & (cols[None, :] < DQ)
        x = tl.load(source + offsets, mask=mask, other=0.0)
        tl.store(dest + rows[:, None] * DQ + cols[None, :], x, mask=mask)
    else:
        v_rows = ((block - 2 * q_blocks) * BT + tl.arange(0, BT)).to(tl.int64)
        v_cols = tl.arange(0, BV)
        v_offsets = (v_rows // HV)[:, None] * SV + (v_rows % HV)[:, None] * DV + v_cols[None, :]
        v_mask = (v_rows[:, None] < T * HV) & (v_cols[None, :] < DV)
        v_values = tl.load(V + v_offsets, mask=v_mask, other=0.0)
        tl.store(V_out + v_rows[:, None] * DV + v_cols[None, :], v_values, mask=v_mask)


def pack_qkv_l2norm(q, k, v):
    """Pack one batch row/window of bf16 planes with contiguous head features."""
    from ...vendor.fla.l2norm import l2norm_fwd
    _, tokens, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    v_out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    # Copy-only packing can use larger tiles than a fused floating-point reduction.
    block_rows = min(128, 16384 // max(triton.next_power_of_2(key_dim), triton.next_power_of_2(value_dim)))
    grid = (2 * triton.cdiv(tokens * heads, block_rows)
            + triton.cdiv(tokens * value_heads, block_rows),)
    with torch.cuda.device(q.device):
        _pack_qkv_kernel[grid](
            q, k, v, q_out, k_out, v_out, tokens,
            heads, value_heads, key_dim, value_dim,
            q.stride(1), k.stride(1), v.stride(1),
            block_rows, triton.next_power_of_2(key_dim), triton.next_power_of_2(value_dim),
            num_warps=4,
        )
        # Reuse the vendored reduction so autotuned layouts retain their exact rounding.
        return l2norm_fwd(q_out)[0], l2norm_fwd(k_out)[0], v_out

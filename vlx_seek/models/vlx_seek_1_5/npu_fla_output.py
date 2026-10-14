"""Ascend output tiling for FLA's 64-token Gated Delta Rule chunks.

Compute 16 query rows per program while attending to their entire 64-token
chunk. This reduces live UB tensors without changing the recurrence or cache.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_output(
    Q, K, V, H0, G, O, T: tl.constexpr, HEADS: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr,
    ROWS: tl.constexpr = 16, BK: tl.constexpr = 32, BV: tl.constexpr = 32,
):
    value_block = tl.program_id(0)
    row_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    chunk = row_block // (64 // ROWS)
    chunks = tl.cdiv(T, 64)
    rows = row_block * ROWS + tl.arange(0, ROWS)
    cols = chunk * 64 + tl.arange(0, 64)
    values = value_block * BV + tl.arange(0, BV)
    attn = tl.zeros((ROWS, 64), tl.float32)
    output = tl.zeros((ROWS, BV), tl.float32)
    for block in range(tl.cdiv(DK, BK)):
        keys = block * BK + tl.arange(0, BK)
        q = tl.load(
            Q + ((batch * T + rows[:, None]) * HEADS + head) * DK + keys[None, :],
            mask=(rows[:, None] < T) & (keys[None, :] < DK), other=0,
        )
        k = tl.load(
            K + ((batch * T + cols[None, :]) * HEADS + head) * DK + keys[:, None],
            mask=(cols[None, :] < T) & (keys[:, None] < DK), other=0,
        )
        state = tl.load(
            H0 + (((batch * chunks + chunk) * HEADS + head) * DK + keys[:, None]) * DV + values[None, :],
            mask=(keys[:, None] < DK) & (values[None, :] < DV), other=0,
        )
        attn += tl.dot(q, k)
        output += tl.dot(q, state)
    row_g = tl.load(G + (batch * T + rows) * HEADS + head, rows < T, other=0)
    col_g = tl.load(G + (batch * T + cols) * HEADS + head, cols < T, other=0)
    output *= tl.exp2(row_g)[:, None]
    attn *= tl.exp2(row_g[:, None] - col_g[None, :])
    attn = tl.where((rows[:, None] >= cols[None, :]) & (cols[None, :] < T), attn, 0)
    v = tl.load(
        V + ((batch * T + cols[:, None]) * HEADS + head) * DV + values[None, :],
        mask=(cols[:, None] < T) & (values[None, :] < DV), other=0,
    )
    output = (output + tl.dot(attn.to(v.dtype), v)) * SCALE
    tl.store(
        O + ((batch * T + rows[:, None]) * HEADS + head) * DV + values[None, :],
        output.to(O.dtype.element_ty),
        mask=(rows[:, None] < T) & (values[None, :] < DV),
    )


def chunk_output(q, k, v, h, g):
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    output = torch.empty_like(v)
    _chunk_output[(triton.cdiv(value_dim, 32), triton.cdiv(length, 16), batch * heads)](
        q, k, v, h, g, output, length, heads, key_dim, value_dim,
        key_dim ** -0.5, num_warps=2, num_stages=1,
        multibuffer=False, code_motion=False,
    )
    return output


def chunk_gated_delta_rule(
    q, k, v, g, beta, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, cu_seqlens=None,
):
    """FLA prefill pipeline with an output kernel sized for Ascend 910B UB."""
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    if cu_seqlens is not None:
        raise ValueError("NPU FLA prefill currently requires unpacked sequences.")
    q, k, v, g, beta = [x.contiguous() for x in (q, k, v, g, beta)]
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size=64, scale=RCP_LN2)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, use_exp2=True)
    h, v_new, state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=initial_state,
        output_final_state=output_final_state, use_exp2=True,
    )
    return chunk_output(q, k, v_new, h, g), state

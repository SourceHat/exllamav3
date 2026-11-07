"""Exact scalar-GDN packing parity and recurrent slot/subrange invariants."""
import importlib

import pytest
import torch



def _planes(mixed, key_dim = 128, value_dim = 128):
    batch, tokens, _ = mixed.shape
    q, k, v = mixed.split((16 * key_dim, 16 * key_dim, 48 * value_dim), -1)
    return (q.view(batch, tokens, 16, key_dim), k.view(batch, tokens, 16, key_dim),
            v.view(batch, tokens, 48, value_dim))


def _reference(q, k, v, g, beta, state, save_state, sub):
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    tokens = q.shape[1]
    outputs = []
    for start in range(0, tokens, sub):
        end = min(start + sub, tokens)
        output, state = chunk_gated_delta_rule(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g = g[:, start:end], beta = beta[:, start:end], initial_state = state,
            output_final_state = save_state if tokens <= sub else True,
            use_qk_l2norm_in_kernel = True,
        )
        outputs.append(output)
    return torch.cat(outputs, dim = 1), state if save_state else None


@pytest.mark.parametrize("tokens,batch,key_dim,value_dim,scale", [
    (48, 1, 128, 128, 1.0),
    (65, 2, 64, 96, 1.0),
    (2049, 1, 128, 128, 1.0),
    (65, 1, 128, 128, 0.0),
    (65, 1, 128, 128, 1e-7),
    (65, 1, 128, 128, 1e4),
    (48, 1, 512, 256, 1.0),
])
@torch.inference_mode()
def test_compact_normalized_planes(device, tokens, batch, key_dim, value_dim, scale):
    from exllamav3.modules.gated_delta_net_fn.pack_qkv import pack_qkv_l2norm
    from exllamav3.vendor.fla.l2norm import l2norm_fwd
    torch.manual_seed(tokens + key_dim)
    features = 32 * key_dim + 48 * value_dim
    backing = torch.randn(batch, tokens + 2, features + 17, device = device, dtype = torch.bfloat16) * scale
    mixed = backing[:, 1:tokens + 1, 7:features + 7]
    q, k, v = _planes(mixed, key_dim, value_dim)
    for row in range(batch):
        for start in range(0, tokens, 2048):
            end = min(start + 2048, tokens)
            qi, ki, vi = q[row:row + 1, start:end], k[row:row + 1, start:end], v[row:row + 1, start:end]
            actual = pack_qkv_l2norm(qi, ki, vi)
            expected = l2norm_fwd(qi.contiguous())[0], l2norm_fwd(ki.contiguous())[0], vi.contiguous()
            for found, reference in zip(actual, expected):
                assert found.is_contiguous()
                torch.testing.assert_close(found, reference, rtol = 0, atol = 0)


@pytest.mark.parametrize("tokens,batch,with_state,save_state", [
    (48, 1, False, False),
    (2048, 1, True, True),
    (2049, 2, True, True),
    (4097, 1, False, True),
    (2049, 2, True, False),
])
@pytest.mark.parametrize("state_dtype", [torch.float, torch.half])
@torch.inference_mode()
def test_prefill_output_and_slot_state(device, tokens, batch, with_state, save_state, state_dtype):
    adapter = importlib.import_module("exllamav3.modules.gated_delta_net_fn.gated_delta_rule")
    torch.manual_seed(tokens + 31 * batch + with_state)
    mixed = torch.randn(batch, tokens, 10240, device = device, dtype = torch.bfloat16)
    q, k, v = _planes(mixed)
    gates = -torch.rand(batch, tokens, 48, device = device, dtype = torch.float32) * 0.1
    beta = torch.rand(batch, tokens, 48, device = device, dtype = torch.bfloat16)
    pool = ((torch.randn(4, 1, 48, 128, 128, device = device, dtype = torch.float32) * 0.1).to(state_dtype)
            if with_state else None)
    initial = pool.clone() if pool is not None else None
    expected_pool = pool.clone() if pool is not None else None
    slots = torch.tensor([3, 1][:batch], dtype = torch.long)
    expected = []
    sub = adapter._scan_sub_range if adapter._scan_sub_range > 0 else tokens
    for row, slot in enumerate(slots.tolist()):
        state = expected_pool[slot, 0].float().unsqueeze(0) if with_state else None
        output, final = _reference(q[row:row + 1], k[row:row + 1], v[row:row + 1],
                                   gates[row:row + 1], beta[row:row + 1], state, save_state, sub)
        if with_state and save_state:
            expected_pool[slot, 0].copy_(final.squeeze(0))
        expected.append(output)
    actual = adapter.gated_delta_rule_fn(
        mixed, beta, gates, pool, slots.to(device), False, save_state,
        16, 48, 2048, 6144, 128, 128, params = {"recurrent_slots_scan": slots},
    )
    torch.testing.assert_close(actual, torch.cat(expected), rtol = 0, atol = 0)
    if with_state:
        torch.testing.assert_close(pool, expected_pool, rtol = 0, atol = 0)
        for slot in set(range(4)) - set(slots.tolist()):
            torch.testing.assert_close(pool[slot], initial[slot], rtol = 0, atol = 0)
    if batch == 1:
        before = actual.clone()
        mixed.zero_()
        if pool is not None:
            pool.zero_()
        torch.testing.assert_close(actual, before, rtol = 0, atol = 0)


@pytest.mark.parametrize("layout,dtype", [("pitched", torch.bfloat16), ("compact", torch.float16)])
@torch.inference_mode()
def test_generic_input_fallback(device, layout, dtype):
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    adapter = importlib.import_module("exllamav3.modules.gated_delta_net_fn.gated_delta_rule")
    torch.manual_seed(934)
    width = 10240 + (3 if layout == "pitched" else 0)
    mixed = torch.randn(1, 65, width, device = device, dtype = dtype)[..., :10240]
    beta = torch.rand(1, 65, 48, device = device, dtype = torch.bfloat16)
    gates = -torch.rand(1, 65, 48, device = device, dtype = torch.float32)
    expected, _ = chunk_gated_delta_rule(*_planes(mixed), g = gates, beta = beta,
                                        use_qk_l2norm_in_kernel = True)
    actual = adapter.gated_delta_rule_fn(mixed, beta, gates, None, None, False, False,
                                         16, 48, 2048, 6144, 128, 128)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


@torch.inference_mode()
def test_chunked_history_uses_distinct_input_and_output_rows(device):
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    adapter = importlib.import_module("exllamav3.modules.gated_delta_net_fn.gated_delta_rule")
    torch.manual_seed(941)
    mixed = torch.randn(2, 65, 512, dtype = torch.bfloat16, device = device)
    q, k, v = mixed.split((128, 128, 256), -1)
    q, k, v = q.view(2, 65, 2, 64), k.view(2, 65, 2, 64), v.view(2, 65, 4, 64)
    beta = torch.rand(2, 65, 4, dtype = torch.bfloat16, device = device)
    gates = -torch.rand(2, 65, 4, device = device)
    pool = torch.randn(4, 1, 4, 64, 64, dtype = torch.half, device = device) * 0.05
    expected_pool = pool.clone()
    rows_out, rows_in = torch.tensor([3, 1]), torch.tensor([2, 0])
    expected = []
    for index, (so, si) in enumerate(zip(rows_out.tolist(), rows_in.tolist())):
        output, state = chunk_gated_delta_rule(
            q[index:index + 1], k[index:index + 1], v[index:index + 1],
            g = gates[index:index + 1], beta = beta[index:index + 1],
            initial_state = expected_pool[si, 0].float().unsqueeze(0),
            output_final_state = True, use_qk_l2norm_in_kernel = True)
        expected_pool[so, 0].copy_(state.squeeze(0))
        expected.append(output)
    actual = adapter.gated_delta_rule_fn(
        mixed, beta, gates, pool, rows_out.to(device), True, True, 2, 4, 128, 256, 64, 64,
        params = {"recurrent_slots_scan": rows_out, "recurrent_slots_scan_in": rows_in},
        slots_in = rows_in.to(device))
    torch.testing.assert_close(actual, torch.cat(expected), rtol = 0, atol = 0)
    torch.testing.assert_close(pool, expected_pool, rtol = 0, atol = 0)

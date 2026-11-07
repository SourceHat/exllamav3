"""Prefill reads and writes pitched views without touching neighboring heads."""
import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill


@pytest.mark.parametrize("head_dim", [96, 128])
@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("feature_step", [1, 2])
@torch.inference_mode()
def test_head_views(device, head_dim, splits, feature_step):
    torch.manual_seed(619)
    batch, length, heads, kv_heads, pages = 2, 257, 4, 2, 3
    parent = torch.randn((batch, length, 10, head_dim * feature_step), dtype = torch.half, device = device)
    original = parent.clone()
    query = parent[:, :, 2:2 + heads, ::feature_step]
    output_parent = torch.full((batch, length, 11, head_dim * feature_step), 17.0,
                               dtype = torch.half, device = device)
    output = output_parent[:, :, 3:3 + heads, ::feature_step]
    keys = torch.randn((batch * pages, 256, kv_heads, head_dim), dtype = torch.half, device = device)
    values = torch.randn_like(keys)
    table = torch.randperm(batch * pages, dtype = torch.int32, device = device).view(batch, pages)
    lengths = torch.tensor([255, 383], dtype = torch.int32, device = device)
    options = dict(causal = True, pre_appended_len = length, num_splits = splits)
    expected = paged_attn_triton_prefill(query.contiguous(), None, None, keys, values, table, lengths, **options)
    actual = paged_attn_triton_prefill(query, None, None, keys, values, table, lengths, out = output, **options)
    torch.testing.assert_close(actual, expected, atol = 3e-3, rtol = 3e-3)
    assert torch.equal(parent, original)
    assert torch.all(output_parent[:, :, :3] == 17.0)
    assert torch.all(output_parent[:, :, 3 + heads:] == 17.0)
    if feature_step > 1:
        assert torch.all(output_parent[..., 1::feature_step] == 17.0)


@pytest.mark.vram(10)
@pytest.mark.parametrize("splits", [1, 2])
@torch.inference_mode()
def test_large_batch_stride_uses_wide_offsets(device, splits):
    torch.manual_seed(621)
    shape = (2, 33, 4, 96)
    strides = (2**31, 4 * 96, 96, 1)
    query = torch.empty_strided(shape, strides, dtype = torch.half, device = device)
    query.copy_(torch.randn(shape, dtype = torch.half, device = device))
    output = torch.empty_strided(shape, strides, dtype = torch.half, device = device)
    keys = torch.randn((2, 256, 2, 96), dtype = torch.half, device = device)
    values = torch.randn_like(keys)
    table = torch.tensor([[0], [1]], dtype = torch.int32, device = device)
    lengths = torch.tensor([0, 0], dtype = torch.int32, device = device)
    options = dict(causal = True, pre_appended_len = shape[1], num_splits = splits)
    expected = paged_attn_triton_prefill(query.contiguous(), None, None, keys, values, table, lengths, **options)
    actual = paged_attn_triton_prefill(query, None, None, keys, values, table, lengths, out = output, **options)
    torch.testing.assert_close(actual, expected, atol = 3e-3, rtol = 3e-3)


@torch.inference_mode()
def test_overlapping_output_is_rejected(device):
    query = torch.randn((1, 33, 4, 64), dtype = torch.half, device = device)
    keys = torch.randn((1, 256, 2, 64), dtype = torch.half, device = device)
    table = torch.zeros((1, 1), dtype = torch.int32, device = device)
    lengths = torch.zeros((1,), dtype = torch.int32, device = device)
    output = torch.empty((1, 1, 4, 64), dtype = torch.half, device = device).expand_as(query)
    with pytest.raises(ValueError, match = "non-overlapping"):
        paged_attn_triton_prefill(query, None, None, keys, keys, table, lengths,
                                pre_appended_len = 33, out = output)

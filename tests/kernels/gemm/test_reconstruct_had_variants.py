"""Fused packed reconstruction preserves both transforms, tile offsets and batch selection."""
import pytest
import torch

VARIANTS = [(bits, codebook) for bits in range(1, 9) for codebook in range(3)]
VARIANTS += [(bits + 0.5, 2) for bits in range(1, 4)]


@pytest.mark.parametrize("rate,codebook", VARIANTS)
@torch.inference_mode()
def test_fused_reconstruct_whole_slice_and_batch(device, rate, codebook):
    from exllamav3.ext import exllamav3_ext as ext

    generator = torch.Generator().manual_seed(5700 + int(rate * 2) * 7 + codebook)
    k, n, batch = 256, 384, 3
    packed = torch.randint(0, 65536, (batch, k // 16, n // 16, int(16 * rate)),
                           dtype = torch.int32, generator = generator).to(torch.int16).to(device)
    packed[0, 0, 0].zero_()
    packed[0, -1, -1].fill_(-1)
    packed[1, 0, -1].fill_(0x55aa)
    packed[2, -1, 0].fill_(-21846)
    su = torch.sign(torch.randn(batch, k, generator = generator)).half().to(device)
    sv = torch.sign(torch.randn(batch, n, generator = generator)).half().to(device)
    whole = torch.empty(batch, k, n, dtype = torch.half, device = device)
    flags = (rate, codebook == 1, codebook == 2)
    for index in range(batch):
        ext.reconstruct_had_slice(whole[index], packed[index], su[index], sv[index], *flags, 0)
    for offset in (0, 128, 256):
        sliced = torch.empty(k, 128, dtype = torch.half, device = device)
        ext.reconstruct_had_slice(sliced, packed[1], su[1], sv[1, offset:], *flags, offset)
        torch.testing.assert_close(sliced, whole[1, :, offset:offset + 128], rtol = 0, atol = 0)
    pointers = lambda values: torch.tensor([v.data_ptr() for v in values], dtype = torch.int64, device = device)
    packed_ptrs, su_ptrs, sv_ptrs = pointers(packed), pointers(su), pointers(sv)
    batched = torch.empty_like(whole)
    ext.reconstruct_had_batch(batched, packed_ptrs, su_ptrs, sv_ptrs, *flags)
    torch.testing.assert_close(batched, whole, rtol = 0, atol = 0)

    rotated = torch.empty(k, n, dtype = torch.half, device = device)
    ext.reconstruct(rotated, packed[0], *flags)
    h = torch.ones(1, 1, device = device)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= 128 ** 0.5
    reference = torch.einsum("ij,bjn->bin", h, rotated.float().view(k // 128, 128, n)).reshape(k, n)
    reference = torch.einsum("bki,ij->bkj", reference.view(k, n // 128, 128).transpose(0, 1), h)
    reference = reference.transpose(0, 1).reshape(k, n)
    reference = (reference * su[0].float()[:, None] * sv[0].float()[None, :]).half()
    error = (whole[0].float() - reference.float()).abs().max()
    scale = reference.float().abs().max()
    assert error < 2e-3 * scale

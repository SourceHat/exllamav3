"""EXL3 8-bpw circular byte windows must reconstruct the unchanged codebook values."""
import pytest
import torch



def _codebook(codebook):
    states = torch.arange(65536, dtype = torch.int64)
    multiplier = (89226354, 0xCBAC1FED, 0x83DCD12D)[codebook]
    product = (states * multiplier + (64248484 if codebook == 0 else 0)) & 0xffffffff
    if codebook == 2:
        byte_sum = sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
        scale = torch.tensor([0x1eee], dtype = torch.uint16).view(torch.float16).float()
        bias = torch.tensor([0xc931], dtype = torch.uint16).view(torch.float16).float()
        return ((1024 + byte_sum).float() * scale + bias).half()
    bits = ((product & 0x8fff8fff) ^ 0x3b603b60).to(torch.int64)
    lo = (bits & 65535).to(torch.uint16).view(torch.float16)
    hi = (bits >> 16).to(torch.uint16).view(torch.float16)
    return lo + hi


def _packed_and_expected(codebook, device):
    # Distinct adjacent tiles and every byte value expose wrap, word-order and lane mapping errors.
    bytes_ = (torch.arange(256).view(1, 256) + torch.arange(64).view(64, 1) * 17) & 255
    bytes_[0].zero_()
    bytes_[1].fill_(255)
    bytes_[2] = torch.where(torch.arange(256) % 2 == 0, 0x55, 0xaa)
    bytes_[3] = torch.where(torch.arange(256) % 2 == 0, 0xaa, 0x55)
    windows = (bytes_.roll(1, dims = 1) << 8) | bytes_
    values = _codebook(codebook)[windows]
    row = torch.arange(256) // 8
    item = torch.arange(256) % 8
    matrix_rows = 2 * (row % 4) + (item % 2) + 8 * ((item // 2) % 2)
    matrix_cols = row // 4 + 8 * (item // 4)
    tiles = torch.empty(64, 16, 16, dtype = torch.float16)
    tiles[:, matrix_rows, matrix_cols] = values
    expected = tiles.view(8, 8, 16, 16).permute(0, 2, 1, 3).reshape(128, 128)
    words = ((bytes_[:, 0::4] << 24) | (bytes_[:, 1::4] << 16)
             | (bytes_[:, 2::4] << 8) | bytes_[:, 3::4]).to(torch.int32)
    packed = words.view(torch.int16).reshape(8, 8, 128).contiguous().to(device)
    return packed, expected.to(device)


@pytest.mark.parametrize("codebook", [0, 1, 2])
@torch.inference_mode()
def test_eightbit_reconstruct_circular_windows(device, codebook):
    from exllamav3.ext import exllamav3_ext as ext
    packed, expected = _packed_and_expected(codebook, device)
    actual = torch.empty_like(expected)
    ext.reconstruct(actual, packed, 8, codebook == 1, codebook == 2)
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


@pytest.mark.parametrize("output_rows", [1, 4, 17])
@torch.inference_mode()
def test_eightbit_gemm_uses_reconstructed_fragments(device, output_rows):
    from exllamav3.ext import exllamav3_ext as ext
    packed, weight = _packed_and_expected(2, device)
    generator = torch.Generator().manual_seed(1794)
    a = (torch.randn(output_rows, 128, generator = generator) * 0.1).half().to(device)
    scales = torch.ones(128, dtype = torch.float16, device = device)
    transformed = torch.empty_like(a)
    ext.had_r_128(a, transformed, scales, None, 1.0)
    expected = transformed.float() @ weight.float()
    expected_half = expected.half()
    ext.had_r_128(expected_half, expected_half, None, scales, 1.0)
    actual = torch.empty_like(expected_half)
    ext.exl3_gemm(a, packed, actual, scales, transformed, scales, 0, False, True, 0)
    # The numerical contract includes the native GEMM's accumulation/reduction ordering.
    rms = expected_half.float().square().mean().sqrt()
    error = (actual.float() - expected_half.float()).square().mean().sqrt()
    assert error <= 0.02 * rms

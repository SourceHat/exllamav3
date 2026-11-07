"""CPU table uploads retain every finite 16-bit value and existing fallback behavior."""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.embedding import Embedding
from exllamav3.model.model_ls import Model_LSMixin
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX


def embedding(table, output = torch.float, normalize = False, multiplier = 1.0):
    layer = Embedding(None, "test", table.shape[0], table.shape[1], out_dtype = output,
                      normalize = normalize, multiplier = multiplier)
    layer.device = torch.device("cpu")
    layer.embedding = torch.nn.Embedding.from_pretrained(table, freeze = True)
    return layer


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("pinned", [False, True])
@torch.inference_mode()
def test_all_finite_values(device, dtype, pinned):
    values = torch.arange(-32768, 32768, dtype = torch.int16).view(dtype)
    table = values[torch.isfinite(values)].reshape(-1, 1)
    layer = embedding(table)
    ids = torch.arange(len(table), dtype = torch.long).reshape(1, -1)
    expected = layer.forward(ids, {}).to(device)
    actual = layer.forward(ids, {"_embedding_output_device": device, "pinned_staging": pinned})
    assert actual.dtype == torch.float and actual.device == device
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    if pinned:
        assert len(layer._pinned_staging) == 1
        assert next(iter(layer._pinned_staging.values())).dtype == dtype


@pytest.mark.parametrize("output,normalize,multiplier", [
    (torch.float, True, 1.0), (torch.float, False, 1.125), (torch.half, False, 1.0),
])
@torch.inference_mode()
def test_transformed_and_half_output_fallback(device, output, normalize, multiplier):
    table = torch.tensor([[0.0, -0.0, 1.5, -2.0], [0.25, -0.5, 3.0, 4.0]], dtype = torch.half)
    layer = embedding(table, output, normalize, multiplier)
    ids = torch.tensor([[1, 0, 1]], dtype = torch.long)
    expected = layer.forward(ids, {})
    actual = layer.forward(ids, {"_embedding_output_device": device})
    assert actual.device.type == "cpu" and actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@torch.inference_mode()
def test_indexed_embeddings_and_deepstack(device):
    table = torch.tensor([[0.25, -0.5], [1.5, -2.0]], dtype = torch.half)
    indexed = SimpleNamespace(first_index = FIRST_MM_EMBEDDING_INDEX, mm_length = 1,
                              embeddings = torch.tensor([[3.0, -4.0]], dtype = torch.half),
                              deepstack_embeddings = [torch.tensor([[0.5, -0.75]], dtype = torch.half)])
    ids = torch.tensor([[1, FIRST_MM_EMBEDDING_INDEX, 0]], dtype = torch.long)
    params = {"indexed_embeddings": [indexed], "_embedding_output_device": device}
    actual = embedding(table).forward(ids, params)
    expected = torch.tensor([[[1.5, -2.0], [3.0, -4.0], [0.25, -0.5]]])
    assert actual.device.type == "cpu" and torch.equal(actual, expected)
    assert torch.equal(params["deepstack_emb"][0], torch.tensor([[[0.0, 0.0], [0.5, -0.75], [0.0, 0.0]]]))


@pytest.mark.parametrize("method", ["forward_ls", "prefill_ls"])
@torch.inference_mode()
def test_layer_split_dispatch_supplies_destination(device, method):
    table = torch.tensor([[0.25, -0.5], [1.5, -2.0]], dtype = torch.half)
    layer = embedding(table)
    seen = []

    def consume(x, params):
        assert x.device == device and x.dtype == torch.float
        assert "_embedding_output_device" not in params
        seen.append(x.clone())
        return x

    consumer = SimpleNamespace(device = device, caps = {},
                               prepare_for_device = lambda x, params: x,
                               forward = consume)
    model = SimpleNamespace(config = SimpleNamespace(),
                            fwd_modules = [(layer, 0, 0), (consumer, 0, 1)],
                            last_kv_module_idx_instance = (1, 0))
    ids = torch.tensor([[1, 0, 1]])
    getattr(Model_LSMixin, method)(model, ids, {})
    assert len(seen) == 1
    assert torch.equal(seen[0].cpu(), table[ids].float())

"""Asynchronous snapshots preserve committed rows, ownership and host-pool reuse."""
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from exllamav3.cache.recurrent import host_pool
from exllamav3.generator.disk_cache import DiskPageCache
from exllamav3.model.model_tp_fn import pack_stash, unpack_stash
from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState


pytestmark = pytest.mark.cuda_only


class _Cache:
    def __init__(self, layers):
        self.layers = layers
        self.model = SimpleNamespace(loaded_tp = False)

    def get_all_recurrent_layers(self):
        return self.layers


@pytest.fixture(autouse = True)
def release_pool(device):
    host_pool.release()
    yield
    torch.cuda.synchronize(device)
    host_pool.release()


def make_state(device, slots = 1, slot = 0, parity = 0, dtype = torch.float):
    geometry = SimpleNamespace(fdim_qkv = 80, conv_kernel_size = 4,
                               num_v_heads = 3, k_head_dim = 16, v_head_dim = 16)
    layer = GDNLayerState(geometry, slots, 3, 123, state_dtype = dtype)
    layer.alloc(device)
    cache = _Cache({("layer", 0): layer})
    state = GDNState(cache, slot, 0, clear = False)
    state.parity = parity
    for row in range(2 * slots):
        layer.recurrent_state[row].fill_(7.25 + row)
    for index in range(slots):
        layer.conv_state[index].fill_(4.5 + index)
    torch.cuda.synchronize(device)
    return cache, state, layer


def delayed_stream(cache, device):
    stream = torch.cuda.Stream(device = device)
    cache._checkpoint_copy_streams = {device: stream}
    with torch.cuda.stream(stream):
        torch.cuda._sleep(100_000_000)
    return stream


@pytest.mark.parametrize("operation", ["index", "get", "dict", "copy", "pop", "items", "values"])
@torch.inference_mode()
def test_mapping_reads_wait(device, operation):
    cache, state, layer = make_state(device)
    stream = delayed_stream(cache, device)
    snapshot = state.stash()
    key = ("layer", 0)
    if operation == "index":
        saved = snapshot[key]
    elif operation == "get":
        saved = snapshot.get(key)
    elif operation == "dict":
        saved = dict(snapshot)[key]
    elif operation == "copy":
        saved = snapshot.copy()[key]
    elif operation == "pop":
        saved = snapshot.pop(key)
    elif operation == "items":
        saved = dict(snapshot.items())[key]
    else:
        saved = next(value for value in snapshot.values() if isinstance(value, tuple))
    assert stream.query()
    assert torch.equal(saved[0], torch.full_like(saved[0], 7.25))
    assert torch.equal(saved[1], torch.full_like(saved[1], 4.5))


@torch.inference_mode()
def test_snapshot_does_not_retain_owner(device):
    cache, state, layer = make_state(device)
    snapshot = state.stash()
    owner, source = weakref.ref(cache), weakref.ref(layer.recurrent_state)
    del cache, state, layer
    gc.collect()
    assert owner() is None and source() is None
    assert torch.equal(snapshot[("layer", 0)][0], torch.full_like(snapshot[("layer", 0)][0], 7.25))


@torch.inference_mode()
def test_each_mutation_stream_waits(device):
    cache, state, layer = make_state(device)
    delayed_stream(cache, device)
    snapshot = state.stash()
    first, second = torch.cuda.Stream(device = device), torch.cuda.Stream(device = device)
    with torch.cuda.stream(first):
        layer.get_state_tensors()
    with torch.cuda.stream(second):
        layer.clear(0)
    recurrent, conv = snapshot[("layer", 0)]
    assert torch.equal(recurrent, torch.full_like(recurrent, 7.25))
    assert torch.equal(conv, torch.full_like(conv, 4.5))
    second.synchronize()
    assert torch.count_nonzero(layer.recurrent_state).item() == 0
    first.synchronize()


@pytest.mark.parametrize("slot", range(4))
@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_committed_row_restore_and_free(device, slot, parity, dtype):
    cache, state, layer = make_state(device, 4, slot, parity, dtype)
    expected = layer.recurrent_state[state.base_row():state.base_row() + 1].cpu()
    expected_conv = layer.conv_state[slot, :, :4].cpu()
    snapshot = state.stash()
    layer.clear(slot)
    assert torch.equal(snapshot[("layer", 0)][0], expected)
    state.unstash(snapshot)
    assert torch.equal(layer.recurrent_state[state.base_row():state.base_row() + 1].cpu(), expected)
    assert torch.equal(layer.conv_state[slot, :, :4].cpu(), expected_conv)
    retained = state.stash()
    layer.free()
    layer.free()
    assert torch.equal(retained[("layer", 0)][0], expected)
    assert torch.equal(retained[("layer", 0)][1], expected_conv)


@torch.inference_mode()
def test_pool_reuse_waits_and_separates_pinned_buffers(device):
    cache, state, layer = make_state(device)
    stream = delayed_stream(cache, device)
    snapshot = state.stash()
    host_pool.give({"target": snapshot})
    assert stream.query()
    recurrent, conv = snapshot[("layer", 0)]
    pinned = host_pool.take(recurrent.shape, recurrent.dtype, pin_memory = True)
    assert pinned is recurrent and pinned.is_pinned()
    pageable = host_pool.take(recurrent.shape, recurrent.dtype)
    assert pageable is not recurrent and not pageable.is_pinned()


@torch.inference_mode()
def test_disk_structure_materializes_paired_snapshot(device):
    cache, state, layer = make_state(device, parity = 1)
    delayed_stream(cache, device)
    snapshot = state.stash()
    paired = {"position": 0, "checkpoint_size": state.checkpoint_size, "target": snapshot}
    disk = DiskPageCache.__new__(DiskPageCache)
    top, sides = disk._sides(paired)
    assert set(sides) == {"target"} and "target" not in top
    tensors = []
    packed = pack_stash(paired, tensors)
    restored = unpack_stash(packed, tensors)
    assert torch.equal(restored["target"][("layer", 0)][0], torch.full_like(tensors[0], 8.25))
    assert all(t.device.type == "cpu" and t.is_pinned() for t in tensors)

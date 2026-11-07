"""Synthetic operator timings; run the same script against base and patched sources."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--source", type = Path, default = Path.cwd(), help = "ExLlamaV3 source checkout")
    parser.add_argument("--device", default = "cuda:0")
    parser.add_argument("--tokens", type = int, default = 2048)
    parser.add_argument("--matrix-size", type = int, default = 2048)
    parser.add_argument("--warmup", type = int, default = 3)
    parser.add_argument("--repetitions", type = int, default = 20)
    parser.add_argument("--iterations-per-sample", type = int, default = 10,
                        help = "Amortize synchronization for GPU-only cases; host-buffer cases always synchronize each pass")
    parser.add_argument("--cases", nargs = "+", choices = ["attention", "gdn", "embedding", "checkpoint", "reconstruct"],
                        default = ["attention", "gdn", "embedding", "checkpoint", "reconstruct"])
    parser.add_argument("--output", type = Path)
    args = parser.parse_args()
    if args.tokens < 48 or args.matrix_size < 128 or args.matrix_size % 128 or args.warmup < 1 or args.repetitions < 2 or args.iterations_per_sample < 1:
        parser.error("Use tokens >= 48, matrix size a positive multiple of 128, warmup >= 1 and repetitions >= 2")
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    import torch
    import exllamav3
    from exllamav3.ext import exllamav3_ext as ext

    if Path(exllamav3.__file__).resolve() != source / "exllamav3/__init__.py":
        raise RuntimeError("The imported package is not from --source")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1207)
    results = {}

    def measure(name, fn, host_buffers = False):
        for _ in range(args.warmup):
            fn()
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        samples = []
        iterations = 1 if host_buffers else args.iterations_per_sample
        for _ in range(args.repetitions):
            started = time.perf_counter()
            for _ in range(iterations):
                fn()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - started) * 1000 / iterations)
        results[name] = {"median_ms": statistics.median(samples), "samples_ms": samples,
                         "iterations_per_sample": iterations}

    with torch.inference_mode():
        if "attention" in args.cases:
            from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
            q = torch.randn((1, args.tokens, 32, 128), dtype = torch.half, device = device)[:, :, 8:24]
            out = torch.empty((1, args.tokens, 32, 128), dtype = torch.half, device = device)[:, :, 8:24]
            pages = (args.tokens + 2048 + 255) // 256
            k = torch.randn((pages, 256, 4, 128), dtype = torch.half, device = device)
            v = torch.randn_like(k)
            table = torch.arange(pages, dtype = torch.int32, device = device).view(1, pages)
            lengths = torch.tensor([2048], dtype = torch.int32, device = device)
            options = dict(causal = True, pre_appended_len = args.tokens, num_splits = 1)
            expected = paged_attn_triton_prefill(q.contiguous(), None, None, k, v, table, lengths, **options)
            try:
                paged_attn_triton_prefill(q, None, None, k, v, table, lengths, out = out, **options)
            except ValueError as error:
                if str(error) != "q must be contiguous":
                    raise
                def attention():
                    value = paged_attn_triton_prefill(q.contiguous(), None, None, k, v, table, lengths, **options)
                    out.copy_(value)
            else:
                def attention():
                    paged_attn_triton_prefill(q, None, None, k, v, table, lengths, out = out, **options)
            attention()
            torch.testing.assert_close(out, expected, atol = 3e-3, rtol = 3e-3)
            measure("strided_attention_including_pack_or_copyback", attention)
            del q, out, k, v, expected

        if "gdn" in args.cases:
            from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import gated_delta_rule_fn
            mixed = torch.randn((1, args.tokens, 10240), dtype = torch.bfloat16, device = device)
            beta = torch.rand((1, args.tokens, 48), dtype = torch.bfloat16, device = device)
            gates = -torch.rand((1, args.tokens, 48), dtype = torch.float, device = device) * 0.1
            params = {"recurrent_slots_scan": torch.tensor([0])}
            measure("scalar_gdn_prefill", lambda: gated_delta_rule_fn(
                mixed, beta, gates, None, None, False, False, 16, 48, 2048, 6144, 128, 128, params = params))
            del mixed, beta, gates

        if "embedding" in args.cases:
            from exllamav3.modules.embedding import Embedding
            table = torch.randn((4096, 4096), dtype = torch.half)
            layer = Embedding(None, "benchmark", 4096, 4096)
            layer.device = torch.device("cpu")
            layer.embedding = torch.nn.Embedding.from_pretrained(table, freeze = True)
            ids = torch.randint(0, 4096, (1, args.tokens))
            def upload():
                return layer.forward(ids, {"_embedding_output_device": device, "pinned_staging": True}).to(
                    device, non_blocking = True)
            torch.testing.assert_close(upload(), table[ids].float().to(device), atol = 0, rtol = 0)
            measure("fp16_embedding_to_fp32_cuda", upload, host_buffers = True)
            del layer, table, ids

        if "checkpoint" in args.cases:
            from exllamav3.cache.recurrent import host_pool
            from exllamav3.modules.gated_delta_net import GDNLayerState, GDNState
            geometry = SimpleNamespace(fdim_qkv = 10240, conv_kernel_size = 4,
                                       num_v_heads = 48, k_head_dim = 128, v_head_dim = 128)
            layers = {}
            for index in range(16):
                layer = GDNLayerState(geometry, 1, 0, index)
                layer.alloc(device)
                layers[index] = layer
            cache = SimpleNamespace(model = SimpleNamespace(loaded_tp = False),
                                    get_all_recurrent_layers = lambda: layers)
            state = GDNState(cache, 0, 0, clear = False)
            a = torch.randn((args.matrix_size, args.matrix_size), dtype = torch.half, device = device)
            b = torch.randn_like(a)
            c = torch.empty_like(a)
            def checkpoint():
                snapshot = state.stash()
                torch.mm(a, b, out = c)
                host_pool.give(snapshot)
            measure("checkpoint_plus_independent_matmul", checkpoint, host_buffers = True)
            buffers = [tensor for bucket in host_pool.free.values() for tensor in bucket]
            results["checkpoint_plus_independent_matmul"].update({
                "retained_host_pool_bytes": sum(t.numel() * t.element_size() for t in buffers),
                "pinned_host_pool_bytes": sum(t.numel() * t.element_size() for t in buffers if t.is_pinned()),
            })
            del buffers
            host_pool.release()
            del state, cache, layers, layer, a, b, c

        if "reconstruct" in args.cases:
            size = args.matrix_size
            su = torch.ones(size, dtype = torch.half, device = device)
            sv = torch.ones_like(su)
            out = torch.empty((size, size), dtype = torch.half, device = device)
            for bits in (4, 8):
                packed = torch.randint(0, 65536, (size // 16, size // 16, 16 * bits),
                                       dtype = torch.int32, device = device).to(torch.int16)
                measure(f"reconstruct_hadamard_{bits}bit", lambda: ext.reconstruct_had_slice(
                    out, packed, su, sv, bits, False, True, 0))

    digest = hashlib.sha256()
    for path in sorted((source / "exllamav3").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh", ".cpp", ".h"):
            digest.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
    result = {"scope": "Synthetic operator wall time including host work and completion; not serving throughput",
              "source_sha256": digest.hexdigest(), "gpu": torch.cuda.get_device_name(device),
              "compute_capability": torch.cuda.get_device_capability(device), "torch": torch.__version__,
              "tokens": args.tokens, "matrix_size": args.matrix_size, "warmup": args.warmup,
              "repetitions": args.repetitions, "seed": 1207, "results": results}
    text = json.dumps(result, indent = 2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end = "")


if __name__ == "__main__":
    main()

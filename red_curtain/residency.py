"""Move unused compute weights to RAM between acoustic-token and DAC stages.

Expert storage is deliberately excluded: moving the whole model back to CUDA
would also move its 32 GB CPU expert buffers, defeating offloading entirely.
"""
import psutil
import torch


class StageResidency:
    def __init__(self, model, cache, device="cuda:0", check=lambda: None):
        self.model, self.cache = model, cache
        self.device, self.check = torch.device(device), check

    def compute_modules(self):
        backbone = self.model.backbone
        yield backbone.model.embed_tokens
        yield backbone.model.norm
        yield backbone.lm_head
        for layer in backbone.model.layers:
            for name, child in layer.named_children():
                if name == "mlp":
                    for mlp_name, module in child.named_children():
                        if mlp_name != "experts":
                            yield module
                else:
                    yield child
        yield self.model.audio_decoder

    def discard_request_caches(self):
        model = self.model
        model.decode_graphs = None
        model.cache_capacity = 0
        model.backbone.max_batch_size = model.backbone.max_seq_len = 0
        model.backbone.inverse_frequencies.clear()
        for layer in model.backbone.model.layers:
            if layer.layer_type == "full_attention":
                layer.self_attn.k_cache = layer.self_attn.v_cache = None
            else:
                layer.linear_attn.conv_state = layer.linear_attn.recurrent_state = None
        tower = model.audio_decoder
        tower.main_candidates = None
        for layer in (*tower.shared_layers, *tower.residual_layers):
            attention = layer.transformer_layer.attention
            attention.k_cache = attention.v_cache = None
            attention.static_kv_len = None

    @torch.inference_mode()
    def park(self):
        """Keep the model in CPU RAM until the next generation needs it."""
        modules = tuple(self.compute_modules())
        size = sum(t.numel() * t.element_size() for module in modules
                   for t in (*module.parameters(), *module.buffers()) if t.device.type != 'cpu')
        if psutil.virtual_memory().available < size + 1024**3:
            raise MemoryError("Insufficient system RAM to park text/audio-tower weights for DAC decoding")
        self.cache.clear()
        self.discard_request_caches()
        for module in modules:
            self.check()
            module.to('cpu')
        torch.cuda.empty_cache()
        return size

    @torch.inference_mode()
    def resume(self):
        """Restore only compute weights; routed experts must stay in CPU RAM."""
        for module in self.compute_modules():
            self.check()
            module.to(self.device)

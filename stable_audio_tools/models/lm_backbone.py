import torch
from torch import nn
import torch.nn.functional as F
from x_transformers import ContinuousTransformerWrapper, Decoder
from einops.layers.torch import Rearrange

from .mamba_lm import MambaModel 
from mamba_ssm.utils.generation import InferenceParams
from .transformer import ContinuousTransformer

# Interface for backbone of a language model
# Handles conditioning and cross-attention
# Does not have to deal with patterns or quantizer heads
class AudioLMBackbone(nn.Module):
    def __init__(self, embed_dim: int, use_generation_cache=False, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_generation_cache = use_generation_cache

    def forward(
        self, 
        x, 
        cross_attn_cond=None, 
        prepend_cond=None, 
        prepend_cond_mask=None,
        global_cond=None,
        use_cache=False,
        **kwargs
        ):
        raise NotImplementedError
    
    def reset_generation_cache(
        self,
        max_seq_len, 
        batch_size,
        dtype=None
    ):
        pass

    def update_generation_cache(
        self,
        seqlen_offset
    ):
        pass

# class AdaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.zeros(hidden_size))
#         self.cond = None

#     def set_cond(self, cond): #don't want to modify x-transformers forward pass. 
#         self.cond = cond

#     def forward(self, x):
#         if self.cond is None:
#             return x
        
#         scale = F.linear(self.cond, self.weight) + 1
#         var = torch.mean(x ** 2, dim=-1, keepdim=True)
#         x = x * torch.rsqrt(var + self.eps) * scale.unsqueeze(-1)
#         return x


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float):
    dtype = torch.promote_types(x.dtype, torch.float32)
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, cond_size: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, cond_size, **factory_kwargs))
        self.register_buffer("bias", None)
        self.reset_parameters()

        self.cond = None

    def set_cond(self, cond): #don't want to modify x-transformers forward pass. 
        self.cond = cond

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x,):
        return rms_norm(
            x,
            F.linear(self.cond, self.weight) + 1,
            eps=self.eps,
        )

class XTransformersAudioLMBackbone(AudioLMBackbone):
    def __init__(self,
                 embed_dim: int,
                 cross_attn_cond_dim: int = 0,
                 prepend_cond_dim: int = 0,
                 global_cond_dim: int = 0,
                 depth: int = 24,
                 **kwargs):
        super().__init__(embed_dim=embed_dim)

        # Embeddings are done in the AudioLanguageModel, so we use the continuous-input transformer
        self.model = ContinuousTransformerWrapper(
            dim_in=embed_dim,
            dim_out=embed_dim,
            max_seq_len=0, #Not relevant without absolute positional embeds,
            attn_layers=Decoder(
                dim=embed_dim,
                depth=depth,
                attn_flash = True,
                cross_attend = cross_attn_cond_dim > 0,
                zero_init_branch_output=True,
                use_abs_pos_emb = False,
                rotary_pos_emb=True,
                ff_swish = True,
                ff_glu = True,
                **kwargs
            )
        )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if cross_attn_cond_dim > 0:
            # Cross-attention conditioning
            self.to_cross_attn_embed = nn.Sequential(
                nn.Linear(cross_attn_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if global_cond_dim > 0:
            # Global conditioning
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, depth*global_cond_dim*2, bias=False),
                Rearrange('b (n d) -> n b d', n=depth*2, d=global_cond_dim)
            )

            # Replace the norms in each layer with AdaRMSNorm 
            for layer in self.model.attn_layers.layers:
                norms = layer[0]
                for i, norm in enumerate(norms):
                    if norm is not None:
                        norms[i] = AdaRMSNorm(self.model.attn_layers.dim, global_cond_dim)

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):

        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_length = prepend_cond.shape[1]

            if prepend_cond_mask is not None:
                # Cast mask to bool
                prepend_cond_mask = prepend_cond_mask.bool()

        if cross_attn_cond is not None:
            # Project the cross-attention conditioning to the embedding dimension
            cross_attn_cond = self.to_cross_attn_embed(cross_attn_cond)

        if global_cond is not None:
            # Project the global conditioning to the embedding dimension
            global_cond = self.to_global_embed(global_cond)
            current_norm_count = 0
            for layer in self.model.attn_layers.layers:
                norms = layer[0]
                for norm in norms:
                    if norm is not None:
                        norm.set_cond(global_cond[current_norm_count])
                        current_norm_count += 1
        

        return self.model(x, mask=mask, context=cross_attn_cond, prepend_embeds=prepend_cond, prepend_mask=prepend_cond_mask)[:, prepend_length:, :]
    
class ContinuousTransformerAudioLMBackbone(AudioLMBackbone):
    def __init__(self,
                 embed_dim: int,
                 cross_attn_cond_dim: int = 0,
                 prepend_cond_dim: int = 0,
                 **kwargs):
        super().__init__(embed_dim=embed_dim)

        # Embeddings are done in the AudioLanguageModel, so we use the continuous-input transformer
        self.model = ContinuousTransformer(
            dim=embed_dim,
            dim_in=embed_dim,
            dim_out=embed_dim,
            cross_attend = cross_attn_cond_dim > 0,
            cond_token_dim = cross_attn_cond_dim,
            causal=True,
            **kwargs
        )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if cross_attn_cond_dim > 0:
            # Cross-attention conditioning
            self.to_cross_attn_embed = nn.Sequential(
                nn.Linear(cross_attn_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):

        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_length = prepend_cond.shape[1]

            if prepend_cond_mask is not None:
                # Cast mask to bool
                prepend_cond_mask = prepend_cond_mask.bool()

        if cross_attn_cond is not None:
            # Project the cross-attention conditioning to the embedding dimension
            cross_attn_cond = self.to_cross_attn_embed(cross_attn_cond)

        return self.model(x, mask=mask, context=cross_attn_cond, prepend_embeds=prepend_cond, prepend_mask=prepend_cond_mask)[:, prepend_length:, :]
    
class MambaAudioLMBackbone(AudioLMBackbone):
    def __init__(self,
                 embed_dim: int,
                 n_layer: int,
                 prepend_cond_dim: int = 0,
                 global_cond_dim: int = 0,
                 **kwargs):
        super().__init__(embed_dim=embed_dim, use_generation_cache=True)

        # Embeddings are done in the AudioLanguageModel, so we use the continuous-input transformer
        self.d_model = embed_dim
        self.model = MambaModel(
            d_model=embed_dim,
            n_layer=n_layer,
            global_cond_dim=global_cond_dim,
            **kwargs,
        )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        if global_cond_dim > 0:
            # Global conditioning
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, n_layer*global_cond_dim, bias=False),
                Rearrange('b (n d) -> n b d', n=n_layer, d=global_cond_dim)
            )

        self.inference_params = None

        self.cuda_stream = None
        self.graph_warmups = 2
        self.cuda_graph = None
        self.cuda_graph_captured = False
        self.captured_x = None
        self.captured_logits = None

    def reset_generation_cache(self, max_seq_len, batch_size, dtype=None):

        if dtype is None:
            dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else next(self.parameters()).dtype

        if self.inference_params is None:
            self.inference_params = InferenceParams(max_seqlen=max_seq_len, max_batch_size=batch_size)
        
        if self.inference_params.max_seqlen != max_seq_len or self.inference_params.max_batch_size != batch_size:
            self.inference_params.key_value_memory_dict = self.model.allocate_inference_cache(batch_size, max_seq_len, dtype=dtype)
            self.cuda_graph_captured = False

        self.inference_params.reset(max_seq_len, batch_size)

    def update_generation_cache(self, seqlen_offset):
        self.inference_params.seqlen_offset = seqlen_offset

    def init_graph(self, dtype: torch.dtype):
        return
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        batch_size = self.inference_params.max_batch_size
        decoding_seqlen = 1
        hidden_dim = self.d_model

        self.captured_x = torch.full((batch_size, decoding_seqlen, hidden_dim), 0.0, dtype=dtype, device="cuda")

        with torch.cuda.stream(s):
            for _ in range(self.graph_warmups):
                self.model(self.captured_x, inference_params=self.inference_params)
            s.synchronize()
        torch.cuda.current_stream().wait_stream(s)

        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.captured_logits = self.model(self.captured_x, inference_params=self.inference_params)

        self.cuda_graph_captured = True

    def forward(self, x, mask=None, prepend_cond=None, prepend_cond_mask=None, cross_attn_cond=None, global_cond=None, use_cache=False):

        prepend_length = 0
        if prepend_cond is not None and not (use_cache and self.inference_params.seqlen_offset > 0):
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond.to(x.dtype))
            prepend_length = prepend_cond.shape[1]

            x = torch.cat([prepend_cond, x], dim=1)

        if global_cond is not None:# and not (use_cache and self.inference_params.seqlen_offset > 0):
            # Project the global conditioning to the embedding dimension
            global_cond = self.to_global_embed(global_cond.to(x.dtype)).contiguous()

        if use_cache and self.inference_params.seqlen_offset == 1 and not self.cuda_graph_captured:
            # Second iteration, first time using the step() function, we need to capture the graph here
            self.init_graph(x.dtype)

        if use_cache and self.cuda_graph_captured and self.inference_params.seqlen_offset > 0:
            self.captured_x.copy_(x)
            self.cuda_graph.replay()
            return self.captured_logits.clone()

        return self.model(
            x,
            inference_params=self.inference_params if use_cache else None,
            global_cond=global_cond,
        )[:, prepend_length:, :]
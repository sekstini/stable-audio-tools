from typing import Literal, cast
from einops import rearrange
from torch import nn
import torch

class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError
    
    def tokenize(self, x):
        raise NotImplementedError
    
    def decode_tokens(self, tokens):
        raise NotImplementedError

class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False):
        super().__init__(enable_grad=False, io_channels=model.io_channels, is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete)
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale=scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate
        
        self.model_half = model_half
        self.iterate_batch = iterate_batch

        self.encoded_channels = model.latent_dim

        self.num_quantizers = model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        self.codebook_size = model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None

        if self.model_half:
            self.model.half()
    
    def encode(self, x, **kwargs):
        
        if self.model_half:
            x = x.half()

        encoded = self.model.encode(x, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()

        decoded = self.model.decode(z, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded
    
    def tokenize(self, x, **kwargs):
        assert self.model.is_discrete, "Cannot tokenize with a continuous model"

        _, info = self.model.encode(x, return_info = True, **kwargs)

        return info[self.model.bottleneck.tokens_id]
    
    def decode_tokens(self, tokens, **kwargs):
        assert self.model.is_discrete, "Cannot decode tokens with a continuous model"

        return self.model.decode_tokens(tokens, **kwargs)
    
    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)

class WaveletPretransform(Pretransform):
    def __init__(self, channels, levels, wavelet):
        super().__init__(enable_grad=False, io_channels=channels, is_discrete=False)

        from .wavelets import WaveletEncode1d, WaveletDecode1d

        self.encoder = WaveletEncode1d(channels, levels, wavelet)
        self.decoder = WaveletDecode1d(channels, levels, wavelet)

        self.downsampling_ratio = 2 ** levels
        self.io_channels = channels
        self.encoded_channels = channels * self.downsampling_ratio
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
class PQMFPretransform(Pretransform):
    def __init__(self, attenuation=100, num_bands=16):
        # TODO: Fix PQMF to take in in-channels
        super().__init__(enable_grad=False, io_channels=1, is_discrete=False)
        from .pqmf import PQMF
        self.pqmf = PQMF(attenuation, num_bands)

    def encode(self, x):
        # x is (Batch x Channels x Time)
        x = self.pqmf.forward(x)
        # pqmf.forward returns (Batch x Channels x Bands x Time)
        # but Pretransform needs Batch x Channels x Time
        # so concatenate channels and bands into one axis
        return rearrange(x, "b c n t -> b (c n) t")

    def decode(self, x):
        # x is (Batch x (Channels Bands) x Time), convert back to (Batch x Channels x Bands x Time) 
        x = rearrange(x, "b (c n) t -> b c n t", n=self.pqmf.num_bands)
        # returns (Batch x Channels x Time) 
        return self.pqmf.inverse(x)
        
class PretrainedDACPretransform(Pretransform):
    def __init__(self, model_type="44khz", model_bitrate="8kbps", scale=1.0, quantize_on_decode: bool = True, chunked=True, num_quantizers=None):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        import dac
        
        model_path = dac.utils.download(model_type=model_type, model_bitrate=model_bitrate)
        
        self.model = dac.DAC.load(model_path)

        self.quantize_on_decode = quantize_on_decode

        if model_type == "44khz":
            self.downsampling_ratio = 512
        else:
            self.downsampling_ratio = 320

        self.io_channels = 1

        self.scale = scale

        self.chunked = chunked

        self.encoded_channels = self.model.latent_dim

        self.num_quantizers = num_quantizers or self.model.n_codebooks

        self.codebook_size = self.model.codebook_size

    def encode(self, x):

        latents = self.model.encoder(x)

        if self.quantize_on_decode:
            output = latents
        else:
            z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.num_quantizers)
            output = z
        
        if self.scale != 1.0:
            output = output / self.scale
        
        return output

    def decode(self, z):
        
        if self.scale != 1.0:
            z = z * self.scale

        if self.quantize_on_decode:
            z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.num_quantizers)

        return self.model.decode(z)

    def tokenize(self, x):
        return self.model.encode(x)[1]
    
    def decode_tokens(self, tokens):
        latents = self.model.quantizer.from_codes(tokens)
        return self.model.decode(latents)
    
class AudiocraftCompressionPretransform(Pretransform):
    def __init__(self, model_type="facebook/encodec_32khz", scale=1.0, quantize_on_decode: bool = True, num_quantizers=None):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)
        
        from audiocraft.models import CompressionModel
               
        self.model = CompressionModel.get_pretrained(model_type)

        self.quantize_on_decode = quantize_on_decode

        self.downsampling_ratio = round(self.model.sample_rate / self.model.frame_rate)

        self.io_channels = self.model.channels

        self.scale = scale

        #self.encoded_channels = self.model.latent_dim

        self.num_quantizers = num_quantizers or self.model.num_codebooks
        self.model.n_quantizers = num_quantizers or self.model.num_codebooks

        self.codebook_size = self.model.cardinality

    def encode(self, x):

        assert False, "Audiocraft compression models do not support continuous encoding"

        # latents = self.model.encoder(x)

        # if self.quantize_on_decode:
        #     output = latents
        # else:
        #     z, _, _, _, _ = self.model.quantizer(latents, n_quantizers=self.model.n_codebooks)
        #     output = z
        
        # if self.scale != 1.0:
        #     output = output / self.scale
        
        # return output

    def decode(self, z):
        
        assert False, "Audiocraft compression models do not support continuous decoding"

        # if self.scale != 1.0:
        #     z = z * self.scale

        # if self.quantize_on_decode:
        #     z, _, _, _, _ = self.model.quantizer(z, n_quantizers=self.model.n_codebooks)

        # return self.model.decode(z)

    def tokenize(self, x):
        return self.model.encode(x)[0]
    
    def decode_tokens(self, tokens):
        return self.model.decode(tokens)


class VocosCompressionPretransform(Pretransform):
    def __init__(self, model_type="charactr/vocos-encodec-24khz", bandwidth: float = 1.5):
        super().__init__(enable_grad=False, io_channels=1, is_discrete=True)

        assert bandwidth in [1.5, 3.0, 6.0, 12.0], "Vocos bandwidth must be one of 1.5, 3.0, 6.0, 12.0"

        from vocos import Vocos
        from vocos.feature_extractors import EncodecFeatures

        self.model = Vocos.from_pretrained(model_type)
        self.feature_extractor = cast(EncodecFeatures, self.model.feature_extractor)
        self.encodec = self.feature_extractor.encodec

        self.encodec.set_target_bandwidth(bandwidth)

        self.downsampling_ratio = round(self.encodec.sample_rate / self.encodec.frame_rate)
        self.io_channels = self.encodec.channels
        self.num_quantizers = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=bandwidth,
        )
        self.codebook_size = self.encodec.quantizer.bins
        self.bandwidth_id = [1.5, 3.0, 6.0, 12.0].index(bandwidth)

    def encode(self, x):
        assert False, "Vocos compression models do not support continuous encoding"

    def decode(self, z):
        assert False, "Vocos compression models do not support continuous decoding"

    def tokenize(self, x):
        emb = self.encodec.encoder(x)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth) # [num_quantizers, batch, time]
        return codes.transpose(0, 1).contiguous() # [batch, num_quantizers, time]

    def decode_tokens(self, tokens):
        tokens = tokens.transpose(0, 1).contiguous() # [num_quantizers, batch, time]
        offsets = torch.arange(0, self.codebook_size * self.num_quantizers, self.codebook_size, device=tokens.device)
        embeddings_idxs = tokens + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.feature_extractor.codebook_weights).sum(dim=0)
        features = features.transpose(1, 2)

        bandwidth_id = torch.tensor([self.bandwidth_id], device=tokens.device)
        return self.model.decode(features, bandwidth_id=bandwidth_id).unsqueeze(1)

"""
Subquadratic attention combining sliding window and linear attentions
- Using "standard" sliding windows
- Didactically computes outputs with n^2 attention weights for now
- Copied + adapted from linear_window_attention_tk.py for single-file reference

For each layer: 
- We first compute (softmax) attention over sliding windows
- We then compute standard linear attention to "fill in" the earlier parts
- We combine to model the entire sequence
"""
from typing import List, Tuple, Optional, Callable
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache

from .linear_attention import (
    LolcatsLinearAttention, LinearAttentionState, 
    softmax_attention, 
)

# ----------------------
# Sliding window helpers
# ----------------------
def get_masks(window_size: int, q_len: int, k_len: int, 
              device: torch.device) -> tuple[torch.Tensor]:
    """
    Return masks for softmax and linear attention terms
    -> 1 is include, 0 is ignore
    """
    kwargs = {'device': device, 'dtype': int}
    
    causal_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len)
    linear_mask = torch.ones((q_len, k_len), **kwargs).tril(k_len - q_len - window_size)
    window_mask = causal_mask - linear_mask
    # Return softmax mask (window), linear attention mask
    # -> shapes broadcast over (b, h, q_len, k_len)
    return window_mask[None, None, ...], linear_mask[None, None, ...]


def hybrid_attention_quadratic(q: torch.Tensor, k: torch.Tensor, 
                               f_q: torch.Tensor, f_k: torch.Tensor,
                               f_qp: torch.Tensor, f_kp: torch.Tensor,
                               alpha: torch.Tensor,
                               v: torch.Tensor,
                               window_factor: torch.Tensor,
                               linear_factor: torch.Tensor,
                               window_size: int,
                               kv_state: torch.Tensor = None,
                               k_state: torch.Tensor = None,
                               eps: float = 1e-12,
                               mask_value: float=-1e8):
    """
    Hybrid attention combining sliding window and linear attentions
    """

    mask_window, mask_linear = get_masks(window_size, q.shape[-2], k.shape[-2], q.device)

    # 1. Sliding window (softmax attention)
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (k.shape[-1] ** -0.5)
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    # torch.softmax(a_sm, dim=-1), but we account for the max when combining
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm   = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Under window (linear attention)
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
    a_lnp = torch.einsum('bhmd,bhnd->bhmn', f_qp.float(), f_kp.float())

    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    a_lnp = linear_factor * a_lnp.masked_fill(~mask_linear.bool(), 0)
    a_ln = a_ln - alpha * a_lnp

    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine
    a = ((a_sm + a_ln) / (sum_sm + sum_ln)).to(q.dtype)  # Save attention weights
    # Allow outputs to also depend on prior kv_state and k_state
    y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v.float())
    if kv_state is not None:  # Combine with prior kv_state and k_state
        y += linear_factor * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float())
        sum_ln += linear_factor * torch.einsum(
            'bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]
    y = (y / (sum_sm + sum_ln)).to(q.dtype)
    return y, a  # attention weights only for the last chunk


# ---------------------
# Attention layer class
# ---------------------
class LolcatsSlidingWindowAttentionWithDiff(LolcatsLinearAttention):
    """
    Lolcats attention combining sliding window and linear attention, using the difference of two feature maps
    """
    def __init__(self, 
                 window_size: int = 64, 
                 decode_window_size: int = None,
                 affine_attention_factors: bool = False,
                 init_window_factor: float = 0,
                 train_window_factor: bool = True,
                 state_grad_enabled: bool = False,
                 **kwargs):
        self.window_size = window_size
        self.decode_window_size = (
            decode_window_size if decode_window_size is not None else window_size
        )
        self.window_kwargs = {'dimension': 2, 'size': window_size, 'step': 1}

        super().__init__(**kwargs)
        self.attention_type = kwargs['attention_type']  #  'hedgehog_llama_window_sw'
        # Determine how we compute attentions
        self.quadratic_attention = hybrid_attention_quadratic
        self.attention_type = kwargs['attention_type']  # 'hedgehog_long_llama_window_sw'
        # Learnable factor for combining attentions
        self.affine_attention_factors = affine_attention_factors
        device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
        if train_window_factor:
            self.window_factors = nn.Parameter(
                init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "window_factors", init_window_factor * torch.ones(1, self.num_heads, 1, 1, device=device, dtype=dtype)
            )
        # Whether we use original flash attention 2 inference (use during attention transfer)
        self.base_inference = False
        self.state_grad_enabled = state_grad_enabled
        
        # Duplicate the key and query feature maps
        self.feature_map_q_p = copy.deepcopy(self.feature_map_q)
        if self.tie_qk_kernels:
            self.feature_map_k_p = self.feature_map_q_p
        else:
            self.feature_map_k_p = copy.deepcopy(self.feature_map_q_p)
        self.feature_map_diff_alpha = nn.Parameter(torch.randn(1, dtype=dtype, device=device))

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with the option to compute attention weights multiple ways
        if self.train_attention is True
        -> Consistent with HuggingFace Transformers for easy use with their pretrained models
        """
        b, l, _ = hidden_states.size()
        q, k, v, kv_seq_len = self.process_qkv(hidden_states, attention_mask, 
                                               position_ids, past_key_value)
        f_q, f_k = self.feature_map_q(q), self.feature_map_k(k)  # Have to do after repeat for grouped-query attn if we use same fmap
        f_qp, f_kp = self.feature_map_q_p(q), self.feature_map_k_p(k)
        f_alpha = self.feature_map_diff_alpha

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            # 2. Compute "predicted" attention outputs
            # compute attn weights under sliding window
            window_factors = F.sigmoid(self.window_factors)
            linear_factors = 1 - window_factors if self.affine_attention_factors else 1
            y_pred, a_pred = self.quadratic_attention(q, k, f_q, f_k, f_qp, f_kp, f_alpha, v,
                                                      window_factors, linear_factors,
                                                      window_size=self.window_size)
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            attn_weights = None
            # attention_mask = None  # For now this is always True
            if past_key_value is None:  # Regular training
                window_factors = F.sigmoid(self.window_factors)
                linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                y_true, a_pred = self.quadratic_attention(q, k, f_q, f_k, f_qp, f_kp, f_alpha, v,
                                                          window_factors, linear_factors,
                                                          window_size=self.window_size)
                attn_weights = a_pred
            else:
                past_key_value.window_size = self.decode_window_size
                if f_q.shape[2] == 1 and kv_seq_len > 1 and not self.training:  # Generating
                    assert use_cache is True
                    _kv = past_key_value.update_for_decoding(k, v, self.layer_idx,
                                                             self.feature_map_k,
                                                             self.feature_map_k_p,
                                                             dtype=q.dtype)
                    k_cache, v_cache, f_kv_state, f_k_state, f_kv_state_p, f_k_state_p = _kv

                    # Sliding window + linear attention decode
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1

                    # Softmax attention terms
                    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k_cache.float()) * (k.shape[-1] ** -0.5)
                    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
                    a_sm   = window_factors * torch.exp(a_sm - a_sm_max)
                    sum_sm = a_sm.sum(dim=-1, keepdim=True)

                    # Combine with linear attention terms
                    y_true = (torch.einsum('bhmn,bhnd->bhmd', a_sm, v_cache.float())
                              + linear_factors * (torch.einsum('bhlf,bhfd->bhld', f_q.float(), f_kv_state.float()) 
                                                  - f_alpha * torch.einsum('bhlf,bhfd->bhld', f_qp.float(), f_kv_state_p.float()))
                                                  )
                    sum_ln = linear_factors * (torch.einsum(
                        'bhlf,bhnf->bhl', f_q.float(), f_k_state.float())[..., None] - f_alpha * torch.einsum(
                        'bhlf,bhnf->bhl', f_qp.float(), f_k_state_p.float())[..., None])
                    y_true = (y_true / (sum_sm + sum_ln)).to(q.dtype) 

                else:  # Stateful training
                    try:
                        kv_state = past_key_value.kv_states[self.layer_idx]
                        k_state  = past_key_value.k_states[self.layer_idx]
                    except IndexError:
                        kv_state, k_state = None, None
                    window_factors = F.sigmoid(self.window_factors)
                    linear_factors = 1 - window_factors if self.affine_attention_factors else 1
                    y_true, _ = self.quadratic_attention(q, k, f_q, f_k, f_qp, f_kp, self.feature_map_diff_alpha, v,
                                                         window_factors, linear_factors,
                                                         window_size=self.window_size,
                                                         kv_state=kv_state,
                                                         k_state=k_state)
                    # Save and update KV cache and states
                    # past_key_value.update(k, v.detach(), self.layer_idx,
                    #                       fmap_key_states=f_k.detach(),
                    #                       accumulate_in_fp32=True)
                    past_key_value.update(k, v, self.layer_idx,
                                        fmap_key_states=f_k,
                                        fmap_key_state_p=f_kp,
                                        accumulate_in_fp32=True)
            # Concatenate heads and apply output projection
            y_true = y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
            y_true = self.o_proj(y_true)
        return y_true, attn_weights, past_key_value


class LinearAttentionSlidingWindowDiffCache(LinearAttentionState):
    """
    Class for `past_key_values`
    -> Alternative to KV cache; here we only maintain a "KV state" and "K state"
    -> Modified from transformers.cache_utils.DynamicCache (v4.36)
    """
    def __init__(self, window_size: int = 64) -> None:
        super().__init__()
        self._seen_tokens = 0  # should be `self.seen_tokens` in Transformers v4.36
        self._seen_tokens_by_layer: List[int] = []

        self.kv_states: List[torch.Tensor] = []
        self.kv_states_p: List[torch.Tensor] = []
        self.k_states:  List[torch.Tensor] = []
        self.k_states_p: List[torch.Tensor] = []

        # Account for sliding windows
        self.decode_kv_states_p: List[torch.Tensor] = []
        self.decode_k_states_p: List[torch.Tensor] = []

        self.decode_kv_states: List[torch.Tensor] = []
        self.decode_k_states: List[torch.Tensor] = []
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []
        self.window_size = window_size

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, 
               layer_idx: Optional[int] = None, cache_kwargs: Optional[any] = None,
               accumulate_in_fp32: bool = False, 
               fmap_key_states: torch.Tensor = None,  # should not be None
               fmap_key_states_p: torch.Tensor = None, # should not be None
               grad_enabled: bool = False,
               **kwargs: any,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV, K states; and KV cache during training
        - For decoding, use `self.decode_kv_states` to keep track of KV states 
          up to sliding window terms
        - For (chunked) training, use `self.kv_states` to keep track of KV states
          up to end of sequence
        - Likewise for `self.decode_k_states` and `self.k_states`
        """
        with torch.set_grad_enabled(grad_enabled):
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[-2]

            dtype = key_states.dtype
            if accumulate_in_fp32:
                # key_states = key_states.float()
                fmap_key_states_p = fmap_key_states_p.float()
                fmap_key_states = fmap_key_states.float()
                value_states = value_states.float()

            # Decoding KV state (KV terms up to last window_size)
            decode_kv_state = torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, :-self.window_size], value_states[:, :, :-self.window_size]
            )
            decode_kv_state_p = torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states_p[:, :, :-self.window_size], value_states[:, :, :-self.window_size]
            )

            # KV state
            kv_state = decode_kv_state + torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states[:, :, -self.window_size:], value_states[:, :, -self.window_size:]
            )
            kv_state_p = decode_kv_state_p + torch.einsum(
                'bhlf,bhld->bhfd', fmap_key_states_p[:, :, -self.window_size:], value_states[:, :, -self.window_size:]
            )

            # shape is b, h, 1, f; note the 1
            decode_k_state = fmap_key_states[:, :, :-self.window_size].sum(dim=-2, keepdim=True)
            k_state = (decode_k_state + fmap_key_states[:, :, -self.window_size:].sum(dim=-2, keepdim=True))

            decode_k_state_p = fmap_key_states_p[:, :, :-self.window_size].sum(dim=-2, keepdim=True)
            k_state_p = (decode_k_state_p + fmap_key_states_p[:, :, -self.window_size:].sum(dim=-2, keepdim=True))

            # Update the cache
            if len(self.k_states) <= layer_idx:  # Initializing kv and k states
                self.kv_states.append(kv_state.to(dtype))
                self.k_states.append(k_state.to(dtype))
                self.kv_states_p.append(kv_state_p.to(dtype))
                self.k_states_p.append(k_state_p.to(dtype))

                self.decode_kv_states.append(decode_kv_state.to(dtype))
                self.decode_k_states.append(decode_k_state.to(dtype))

                self.decode_kv_states_p.append(decode_kv_state_p.to(dtype))
                self.decode_k_states_p.append(decode_k_state_p.to(dtype))

                self.k_cache.append(key_states[:, :, -self.window_size:, :])
                self.v_cache.append(value_states[:, :, -self.window_size:, :].to(dtype))
                # self._seen_tokens_by_layer[layer_idx].append(key_states.shape[-2])
            else:
                # Update kv and k states recurrently
                kv_state = (self.kv_states[layer_idx].to(kv_state.dtype) + kv_state).to(dtype)
                k_state  = (self.k_states[layer_idx].to(kv_state.dtype) + k_state).to(dtype)
                kv_state_p = (self.kv_states_p[layer_idx].to(kv_state.dtype) + kv_state_p).to(dtype)
                
                self.kv_states[layer_idx] = kv_state
                self.k_states[layer_idx]  = k_state
                self.kv_states_p[layer_idx] = kv_state_p

                decode_kv_state = (self.decode_kv_states[layer_idx].to(kv_state.dtype) 
                                   + decode_kv_state).to(dtype)
                decode_k_state  = (self.decode_k_states[layer_idx].to(kv_state.dtype) 
                                   + decode_k_state).to(dtype)
                
                
                decode_kv_state_p = (self.decode_kv_states_p[layer_idx].to(kv_state.dtype)
                                     + decode_kv_state_p).to(dtype)
                decode_k_state_p  = (self.decode_k_states_p[layer_idx].to(kv_state.dtype)
                                     + decode_k_state_p).to(dtype)

                self.decode_kv_states[layer_idx] = decode_kv_state
                self.decode_k_states[layer_idx]  = decode_k_state
                self.decode_kv_states_p[layer_idx] = decode_kv_state_p
                self.decode_k_states_p[layer_idx]  = decode_k_state_p

                self.k_cache[layer_idx] = key_states[:, :, -self.window_size:, :]
                self.v_cache[layer_idx] = value_states[:, :, -self.window_size:, :]
            self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        return self.kv_states[layer_idx], self.k_states[layer_idx]

    def update_for_decoding(self, keys: torch.Tensor, values: torch.Tensor, 
                            layer_idx: int, feature_map_k: Callable, feature_map_k_p: Callable, dtype: torch.dtype):
        """
        Update the decoding KV and K states, and KV cache, during decodeing
        """
        with torch.no_grad():
            k_cache = self.k_cache[layer_idx]
            v_cache = self.v_cache[layer_idx]

            if k_cache.shape[-2] < self.window_size:  # build window-size cache
                self.k_cache[layer_idx] = torch.cat([k_cache, keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache, values], dim=-2)
            else:
                # MZ 6/3: handle short inputs; zero-out padding when initial k.shape[2] < self.window_size
                # if k_cache[:, :, :1, :].sum() == 0:   # heuristic for zeroing out padding in cache
                #     f_k_state = torch.zeros(k_cache[:, :, :1, :].shape, dtype=dtype, device=k_cache.device)
                # else:
                #     f_k_state = feature_map_k(k_cache[:, :, :1, :])
                # -> MZ (later): above only relevant if we zero-pad in our hybrid attention computation
                k_state = feature_map_k(k_cache[:, :, :1, :])
                k_state_p = feature_map_k_p(k_cache[:, :, :1, :])
                v_state = v_cache[:, :, :1, :]
                kv_state = torch.einsum('bhlf,bhld->bhfd', k_state.float(), v_state.float()).to(dtype) # b, h, f, d
                kv_state_p = torch.einsum('bhlf,bhld->bhfd', k_state_p.float(), v_state.float()).to(dtype) # b, h, f, d

                self.decode_kv_states[layer_idx] += kv_state
                self.decode_kv_states_p[layer_idx] += kv_state_p

                self.decode_k_states[layer_idx] += k_state
                self.decode_k_states_p[layer_idx] += k_state_p
                
                self.k_cache[layer_idx] = torch.cat([k_cache[:, :, 1:, :], keys], dim=-2)
                self.v_cache[layer_idx] = torch.cat([v_cache[:, :, 1:, :], values], dim=-2)
            
            if layer_idx == 0:
                self._seen_tokens += keys.shape[-2]
            self._seen_tokens_by_layer[layer_idx] += keys.shape[-2]
            return (self.k_cache[layer_idx], self.v_cache[layer_idx], 
                    self.decode_kv_states[layer_idx], self.decode_k_states[layer_idx],
                    self.decode_kv_states_p[layer_idx], self.decode_k_states_p[layer_idx])

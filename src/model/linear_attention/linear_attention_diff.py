"""
Diff linear attention class
"""
from typing import Tuple, Optional
import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from .linear_attention import LolcatsLinearAttention, softmax_attention

def diff_quadratic_attention(q_1: torch.Tensor, k_1: torch.Tensor, q_2: torch.Tensor, k_2: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                        causal: bool = True, fp32_attention: bool = False, eps: float = 1e-12,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute attention with feature maps by instantiating L x L matrix of attention weights
    -> Use for attention distillation
    -> Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); v is shape (b, h, l, head_dim)
    """
    # print("Diff quadratic attention shapes: ", q_1.shape, k_1.shape, q_2.shape, k_2.shape, v.shape)
    y = None
    dtype = q_1.dtype
    if fp32_attention:
        q_1, k_1 = q_1.float(), k_1.float()
        q_2, k_2 = q_2.float(), k_2.float()
    
    # if not(0 <= alpha.item() and alpha.item() <= 1):
    #     breakpoint()

    a_1 = torch.einsum('bhmd,bhnd->bhmn', q_1, k_1)  # note we don't scale, tho we could
    a_2 = torch.einsum('bhmd,bhnd->bhmn', q_2, k_2)

    if causal:  # Apply causal mask
        m, n = a_1.shape[-2:]
        causal_mask = torch.ones((m, n), device = a_1.device, dtype = torch.bool).triu(n - m + 1)
        a_1 = a_1.masked_fill(causal_mask, 0)
        a_2 = a_2.masked_fill(causal_mask, 0)

    a_1 = a_1
    a_1 = a_1.to(dtype=dtype) if fp32_attention else a_1

    a_2 = a_2
    a_2 = a_2.to(dtype=dtype) if fp32_attention else a_2

    # Normalize to compute attention
    if torch.isnan(a_1).sum() > 0 or torch.isnan(a_2).sum() > 0:
        breakpoint()

    a = a_1 - alpha * a_2
    a /= a_1.sum(dim=-1, keepdim=True) - alpha * a_2.sum(dim=-1, keepdim=True) + eps
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
        # print("Diff quadratic attention shapes: ", y.shape, a.shape, v.shape)
    return y, a

class LolcatsDiffLinearAttention(LolcatsLinearAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (kwargs.get("feature_map_prime", kwargs["feature_map"]) == kwargs["feature_map"]):
            self.feature_map_k_prime, self.feature_map_q_prime = copy.deepcopy(self.feature_map_k_prime), copy.deepcopy(self.feature_map_q_prime)
        else:
            self.init_feature_map_(
                kwargs["feature_map_prime"],
                kwargs["feature_map_kwargs"],
                kwargs["learned_kernel"],
                kwargs["learned_kernel_kwargs"],
                write_to_prime=True
            )

        self.lambda_init = kwargs.get("lambda_init", 0)
        self.lambda_parameterized = kwargs.get("lambda_parameterized", False)

        if self.lambda_parameterized:
            self.lambda_q1 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k1 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_q2 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k2 = nn.Parameter(
                torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
            )

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

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
        if self.lambda_init > 0:
            f_qp, f_kp = self.feature_map_q_prime(q).sigmoid(), self.feature_map_k_prime(k).sigmoid()
            f_qp = f_qp * f_q
            f_kp = f_kp * f_k
        else:
            f_qp, f_kp = self.feature_map_q_prime(q), self.feature_map_k_prime(k)

        if self.train_attention:
            # 1. Compute "ground-truth" attention output and weights
            with torch.no_grad():
                _y_true, a_true = softmax_attention(q, k, v)[:2]
                y_true = _y_true.transpose(1, 2).contiguous().view(b, l, self.hidden_size)
                y_true = self.o_proj(y_true)

            # print("Shapes: f_qp: ", f_qp.shape, " f_kp: ", f_kp.shape, " f_k: ", f_k.shape, " f_q: ", f_q.shape)

            if self.lambda_parameterized:
                lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float().type_as(q)
                lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float().type_as(q)
                lambda_full = torch.tanh(torch.relu(lambda_1 - lambda_2 + self.lambda_init))
            else:
                lambda_full = self.lambda_init

            y_pred, a_pred = diff_quadratic_attention(f_q, f_k, f_qp, f_kp, v, lambda_full)
            attn_weights = ((a_pred, a_true), (y_pred, _y_true))
        else:
            raise NotImplementedError("Only training attention is supported")

        return y_true, attn_weights, None

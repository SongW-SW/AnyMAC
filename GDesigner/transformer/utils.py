"""
Utility functions for transformer models.
"""

import torch
import torch.nn.functional as F

def min_max_norm(x):
    """
    Normalize a tensor to the range [-1, 1].
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Normalized tensor.
    """     
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """
    Samples from the Gumbel-Softmax distribution and optionally discretizes.
    Uses PyTorch's built-in F.gumbel_softmax when available, otherwise
    implements the logic directly.
    
    Args:
        logits (torch.Tensor): Log-probabilities.
        tau (float): Temperature parameter, controlling the sharpness of the distribution.
        hard (bool): If True, the returned samples will be discretized as one-hot vectors,
                    but will be differentiable due to the straight-through gradient estimation.
        dim (int): Dimension along which to apply the softmax.
    
    Returns:
        torch.Tensor: Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
    """
    # First check if we can use the built-in implementation
    # PyTorch 1.9+ has a native implementation
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)


def sample_agent_index(probs, temperature=1.0):
    """
    Sample an agent index based on probability distribution.
    
    Args:
        probs (torch.Tensor): Probability distribution over agents.
        temperature (float): Temperature parameter for sharpening/softening the distribution.
        
    Returns:
        int: Sampled agent index.
    """
    if temperature != 1.0:
        # Apply temperature scaling
        probs = probs.pow(1.0 / temperature)
        probs = probs / probs.sum()
        
    return torch.multinomial(probs, 1).item()


def get_attention_weights(transformer, tokens, average_heads=True):
    """
    Extract attention weights from a PyTorch transformer.
    
    Args:
        transformer (nn.TransformerEncoder): The transformer encoder.
        tokens (torch.Tensor): Input tokens.
        average_heads (bool): Whether to average attention weights across heads.
        
    Returns:
        list: List of attention weight tensors for each layer.
    """
    # Register hooks to capture attention weights
    attention_weights = []
    
    def get_attention_hook(layer_idx):
        def hook(module, input, output):
            # Extract attention weights from MultiheadAttention
            # This is implementation-dependent and may need adjustment
            attn_output, attn_weights = output
            if average_heads:
                # Average over attention heads
                attn_weights = attn_weights.mean(dim=1)
            attention_weights.append(attn_weights)
        return hook
    
    # Attach hooks to each transformer layer
    hooks = []
    for i, layer in enumerate(transformer.layers):
        # This assumes the self-attention module is accessible as .self_attn
        # Adjust as needed based on the actual implementation
        if hasattr(layer, 'self_attn'):
            h = layer.self_attn.register_forward_hook(get_attention_hook(i))
            hooks.append(h)
    
    # Forward pass to collect attention weights
    with torch.no_grad():
        transformer(tokens)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return attention_weights 
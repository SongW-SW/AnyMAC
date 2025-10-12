"""
Transformer models for routing agents in the GDesigner framework.
Using pretrained GPT-2 small from Hugging Face.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

class RoutingTransformer(nn.Module):
    """
    GPT-2 small model for routing agents in the GDesigner framework.
    
    This model processes agent tokens and a next agent token to determine
    which agent should be executed next in the workflow.
    
    Args:
        hidden_dim (int): Hidden dimension of the transformer.
        num_layers (int): Number of transformer layers (not used when pretrained=True).
        num_heads (int): Number of attention heads (not used when pretrained=True).
        ff_dim (int): Dimension of the feed-forward network (not used when pretrained=True).
        dropout (float): Dropout rate (not used when pretrained=True).
        max_seq_len (int): Maximum sequence length for positional encoding.
        pretrained (bool): Whether to load pretrained GPT-2 small weights.
    """
    def __init__(self, hidden_dim=768, max_seq_len=1024):
        super().__init__()
        # Load pretrained GPT-2 small model
        self.transformer = GPT2Model.from_pretrained("gpt2")
        # Set max position embeddings if needed
        if max_seq_len > self.transformer.config.n_positions:
            print(f"Warning: requested max_seq_len {max_seq_len} is greater than GPT-2's default {self.transformer.config.n_positions}")
            print("The model will be limited to the default maximum sequence length.")
        # Final layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, num_prefix_tokens: int = 0):
        """
        Forward pass through the GPT-2 model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, hidden_dim) or 
                             (batch_size, seq_len, hidden_dim).
                             For agent routing, seq_len = num_agents + 1,
                             where the +1 is for the next agent token.
            num_prefix_tokens (int): The number of initial tokens to exclude from 
                                    positional encoding. If > 0, these tokens will have
                                    zero position embeddings.
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # Add batch dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (1, seq_len, hidden_dim)
        
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        
        # Create attention mask (all ones since we want to attend to all tokens)
        attention_mask = torch.ones((batch_size, seq_len), device=x.device)
        
        if num_prefix_tokens > 0 and num_prefix_tokens < seq_len:
            # We need to handle position embeddings manually to zero out prefix tokens
            # First, create position IDs for the sequence
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Set the position IDs for prefix tokens to zero
            position_ids[:, :num_prefix_tokens] = 0
            
            # Forward pass through GPT-2 with custom position IDs
            # import ipdb; ipdb.set_trace()
            outputs = self.transformer(
                inputs_embeds=x, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )
        else:
            # Standard forward pass through GPT-2
            outputs = self.transformer(
                inputs_embeds=x, 
                attention_mask=attention_mask
            )
        # import ipdb; ipdb.set_trace()
        x = outputs.last_hidden_state
        
        # Apply final normalization
        x = self.norm(x)
        
        # Remove batch dimension if it was added
        if batch_size == 1 and len(x.shape) == 3:
            x = x.squeeze(0)  # (seq_len, hidden_dim)
            
        return x
    

    def get_embedding_info(self):
        """
        Print information about the transformer's embedding dimensions.
        
        Returns:
            dict: A dictionary containing embedding dimension information
        """
        info = {
            "embedding_dim": self.transformer.config.n_embd,
            "model_hidden_dim": self.hidden_dim,
            "vocab_size": self.transformer.config.vocab_size,
            "max_position_embeddings": self.transformer.config.n_positions,
            "num_layers": self.transformer.config.n_layer,
            "num_heads": self.transformer.config.n_head
        }
        
        print(f"Transformer Embedding Dimension: {info['embedding_dim']}")
        print(f"Model Hidden Dimension: {info['model_hidden_dim']}")
        print(f"Max Position Embeddings: {info['max_position_embeddings']}")
        print(f"Number of Layers: {info['num_layers']}")
        print(f"Number of Attention Heads: {info['num_heads']}")
        
        return info 
    
    def eval(self):
        self.transformer.eval()
        self.norm.eval()
        
    def train(self):
        self.transformer.train()
        self.norm.train()
        
    def get_gpt_parameters(self):
        """
        Returns the GPT-2 model parameters for optimization.
        
        This is useful when you want to specifically optimize the GPT-2 parameters
        in an optimizer instead of all parameters in the RoutingTransformer.
        
        Returns:
            list: List of parameters from the GPT-2 model
        """
        return list(self.transformer.parameters()) + list(self.norm.parameters())
    
        

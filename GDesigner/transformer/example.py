"""
Example usage of the RoutingTransformer for agent routing.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from  transformer import RoutingTransformer
from  utils import gumbel_softmax, sample_agent_index, min_max_norm 


def simulate_routing_sequence():
    """
    Simulate a sequence of routing decisions for multiple steps.
    """
    # Initialize parameters
    hidden_dim = 192
    num_agents = 4  # 4 regular agents + 1 decision agent (index 4)
    num_steps = 5
    decision_agent_idx = num_agents  # The decision agent is the last one
    
    # Create agent embeddings
    agent_embeddings = torch.randn(num_agents + 1, hidden_dim)  # +1 for decision agent
    
    # Initialize model
    transformer = RoutingTransformer(
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=3,
        ff_dim=hidden_dim * 4,
        dropout=0.1
    )
    
    # Initialize next agent token (learnable parameter in practice)
    next_agent_token = torch.randn(hidden_dim)
    
    # Track agent states
    agent_states = [{"id": i, "role": f"agent_{i}", "outputs": []} for i in range(num_agents + 1)]
    
    print(f"Simulating routing sequence with {num_agents} agents + 1 decision agent")
    print(f"Maximum steps: {num_steps}")
    
    # Simulation loop
    step = 0
    current_context = "Initial context"
    routing_path = []
    
    while step < num_steps:
        # Get agent tokens with current context information
        agent_tokens = []
        
        for i in range(num_agents + 1):
            # In real implementation, you would combine:
            # - task embedding
            # - agent role embedding
            # - last output embedding
            # - current context embedding
            agent_token = agent_embeddings[i].clone()
            # Add some noise to simulate context influence
            agent_token += 0.1 * torch.randn_like(agent_token) * len(agent_states[i]["outputs"])
            agent_tokens.append(agent_token.unsqueeze(0))
        
        agent_tokens = torch.cat(agent_tokens, dim=0)
        
        # Get similarity scores
        similarity_scores = transformer.get_agent_similarities(agent_tokens, next_agent_token)
        
        # Apply min-max normalization
        similarity_scores = min_max_norm(similarity_scores)
        
        # Apply temperature and gumbel_softmax
        temperature = 1
        # make similarity_score 1 or -1
        # similarity_scores = torch.where(similarity_scores > 0, 1, -1)
        probabilities = gumbel_softmax(3*similarity_scores, tau=temperature, hard=True)
        
        # Sample next agent
        chosen_idx = sample_agent_index(probabilities)  
        chosen_agent = agent_states[chosen_idx]
        
        # Record routing step
        routing_path.append({
            "step": step,
            "chosen_id": chosen_agent["id"],
            "chosen_role": chosen_agent["role"],
            "probabilities": probabilities.detach().cpu().numpy().tolist()
        })
        
        print(f"\nStep {step+1}:")
        print(f"  Similarity scores: {[round(s.item(), 3) for s in similarity_scores]}")
        print(f"  Selected agent: {chosen_agent['role']}")
        print(f"  Selection probabilities: {[round(p, 3) for p in probabilities.tolist()]}")
        
        # Check if decision agent was selected (termination condition)
        if chosen_idx == decision_agent_idx:
            print("  Decision agent selected - routing complete!")
            break
        
        # Simulate agent execution and generate output
        agent_output = f"Output from {chosen_agent['role']} at step {step+1}"
        agent_states[chosen_idx]["outputs"].append(agent_output)
        
        # Update context for next step
        current_context = agent_output
        
        step += 1
    
    # Final summary
    print("\nRouting sequence complete!")
    print(f"Total steps: {step+1}")
    print("Routing path:")
    for i, step in enumerate(routing_path):
        print(f"  {i+1}. {step['chosen_role']} (Probability: {round(max(step['probabilities']), 3)})")
    
    # Visualize routing probabilities
    visualize_routing_path(routing_path, num_agents + 1)
    
    return routing_path


def visualize_routing_path(routing_path, num_agents):
    """
    Visualize the routing probabilities over steps.
    
    Args:
        routing_path (list): List of routing steps with probabilities.
        num_agents (int): Number of agents.
    """
    try:
        # Extract probabilities for each step
        steps = len(routing_path)
        prob_matrix = np.zeros((steps, num_agents))
        
        for i, step in enumerate(routing_path):
            probs = step["probabilities"]
            prob_matrix[i, :] = probs
        
        # Create agent labels
        agent_labels = [f"Agent {i}" for i in range(num_agents-1)]
        agent_labels.append("Decision")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.imshow(prob_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Probability')
        plt.xlabel('Agent')
        plt.ylabel('Step')
        plt.title('Agent Selection Probabilities Over Steps')
        plt.xticks(range(num_agents), agent_labels, rotation=45)
        plt.yticks(range(steps), [f"Step {i+1}" for i in range(steps)])
        
        # Mark selected agents
        for i, step in enumerate(routing_path):
            chosen_id = step["chosen_id"]
            plt.plot(chosen_id, i, 'ro', markersize=10, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('routing_visualization.png')
        print("\nRouting visualization saved to 'routing_visualization.png'")
    except Exception as e:
        print(f"Could not generate visualization: {e}")


def main():
    """
    Run routing transformer examples.
    """
    print("=" * 50)
    print("ROUTING SEQUENCE SIMULATION")
    print("=" * 50)
    simulate_routing_sequence()


if __name__ == "__main__":
    main() 
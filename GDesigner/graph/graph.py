import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import torch.nn.functional as F
import copy

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN,MLP
from GDesigner.transformer.transformer import RoutingTransformer
from torch_geometric.utils import dense_to_sparse
from GDesigner.transformer.utils import gumbel_softmax
from copy import deepcopy

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                use_transformer:bool = False,
                max_routing:int = 10,
                available_roles:List[str] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4) if not use_transformer else shortuuid.ShortUUID().random(length=max_routing)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.max_routing:int = max_routing
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]

        self.role_embeddings = None        
        self.optimizer = None
        self.role_to_idx = None
        self.idx_to_role = None
        
        self.decision_method = decision_method

        
        self.prompt_set = PromptSetRegistry.get(domain)
        if not use_transformer: 
            self.init_nodes() # add nodes to the self.nodes
            self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
            self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        
        if not use_transformer:
            self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
            self.mlp = MLP(384,16,16)

        if use_transformer:
            transformer_hidden_dim = 768
            # Transformer components for agent routing
            self.transformer_hidden_dim = transformer_hidden_dim
            # Projection layer from concatenated features (384*4) to transformer hidden dim (192)
            self.proj_to_transformer_dim_history = torch.nn.Linear(384*2, transformer_hidden_dim)
            self.proj_to_transformer_dim_task = torch.nn.Linear(384, transformer_hidden_dim)
            self.proj_to_transformer_dim_role = torch.nn.Linear(384, transformer_hidden_dim)
            # Projection layers for NAP and NHP instead of learned vectors
            self.proj_to_transformer_dim_nap = torch.nn.Linear(384, transformer_hidden_dim)
            self.proj_to_transformer_dim_nhp = torch.nn.Linear(384, transformer_hidden_dim)
            # Routing transformer 
            self.routing_transformer = RoutingTransformer(hidden_dim=transformer_hidden_dim, max_seq_len=max_routing + 1) # +1 for the decision node


        if not use_transformer:
            init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
            # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
            #                                          requires_grad=optimized_spatial) # trainable edge logits
            self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

            init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
            self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                    requires_grad=optimized_temporal) # trainable edge logits
            self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks


        # Initialize role embeddings
        self.role_embeddings = {}

        self.cos_scaling = 1
        
        # Add regular agents
        embedding_dim = 384  # Standard embedding dimension
        for role in available_roles:
            # Get role description and create embedding
            role_desc = self.prompt_set.get_description(role)
            role_embedding = torch.tensor(get_sentence_embedding(role_desc))  # (384,)
            # Store as tensor (not Parameter) to make it fixed/not trainable
            self.role_embeddings[role] = role_embedding

        # Add decision node
        decision_role = self.decision_method
        decision_desc = self.prompt_set.get_decision_role()
        decision_embedding = torch.tensor(get_sentence_embedding(decision_desc))
        # Store as tensor (not Parameter) to make it fixed/not trainable
        self.role_embeddings[decision_role] = decision_embedding

        self.role_to_idx = deepcopy({role: idx for idx, role in enumerate(available_roles)})
        self.role_to_idx[decision_role] = len(available_roles)  # DecisionMaker gets the next index
        self.idx_to_role = deepcopy({idx: role for role, idx in self.role_to_idx.items()})

        self.available_roles = available_roles
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features,query_embedding),dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        #import ipdb; ipdb.set_trace()
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        new_features = self.construct_new_features(input['task'])
        logits = self.gcn(new_features,self.role_adj_matrix)
        logits = self.mlp(logits)
        # import ipdb; ipdb.set_trace()
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

    def run_next_agent_prediction(self, input: Dict[str, str], max_routing=10, temperature=1.0,
                               available_roles:List[str] = None, training=False, agent_group_type="MathSolver", max_context = 5):
        """
        Synchronous version of arun_next_agent_prediction that runs the async function in an event loop.
        
        Args:
            input: Dict containing the task and other inputs
            max_routing: Maximum number of routing steps
            temperature: Temperature for Gumbel-Softmax sampling
            available_roles: List of available agent roles to choose from
            
        Returns:
            Dict containing final answers and routing results
        """
        loop = asyncio.new_event_loop()
        timeout = 1800
        if training:
            timeout = 1800
        else:
            timeout = 1800

        try:
            # Run the async function in the loop with a 1-minute timeout
            return loop.run_until_complete(
                asyncio.wait_for(
                    self.arun_next_agent_prediction(
                        input=input,
                        max_routing=max_routing,
                        temperature=temperature,
                        available_roles=available_roles,
                        agent_group_type=agent_group_type,
                        max_context=max_context
                    ),
                    timeout=timeout  
                )
            )
        except asyncio.TimeoutError:
            print("Timeout occurred")
            return None
        finally:
            loop.close()

    async def arun_next_agent_prediction(self, input: Dict[str, str], max_routing=10, temperature=1.0,
                                         available_roles:List[str] = None, agent_group_type="MathSolver", max_context = 5):
        """
        Predicts the next agent to execute using a routing transformer, and finish a trace of the routing process.
        
        Args:
            input: Dict containing the task and other inputs
            max_routing: Maximum number of routing steps
            temperature: Temperature for Gumbel-Softmax sampling
            available_roles: List of available agent roles to choose from
            
        Returns:
            Dict containing final answers and routing results
        """
        # Extract task query
        self.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        query = input['task']

        available_roles = self.available_roles
        decision_role = self.decision_method

        # import ipdb; ipdb.set_trace()


        # Get task embedding (384 dimensions)
        raw_task_embedding = torch.tensor(get_sentence_embedding(query)).detach()
        raw_task_embedding = raw_task_embedding.to(self.device)
        task_embedding = self.proj_to_transformer_dim_task(raw_task_embedding)

        # import ipdb; ipdb.set_trace()
        role_embeddings = self.role_embeddings
        # Create role indices mapping
        role_to_idx = self.role_to_idx  
        idx_to_role = self.idx_to_role
        
        # Initialize projected role embeddings (start empty)
        proj_roles_embeddings = []
        
        # Project role embeddings to transformer dimension
        # import ipdb; ipdb.set_trace()
        for role in available_roles:
            role_embedding = role_embeddings[role].to(self.device)
            # Project role embedding to transformer dimension
            proj_role_embedding = self.proj_to_transformer_dim_role(role_embedding)
            proj_roles_embeddings.append(proj_role_embedding)
            
        decision_embedding = self.role_embeddings[self.decision_method].to(self.device)
        # Project decision embedding to transformer dimension
        proj_decision_embedding = self.proj_to_transformer_dim_role(decision_embedding)
        # Add decision role embedding
        proj_roles_embeddings.append(proj_decision_embedding)
        
        # Convert to tensor
        proj_roles_embeddings = torch.stack(proj_roles_embeddings) if proj_roles_embeddings else torch.empty((0, self.transformer_hidden_dim)) # Handle case with no roles
        
        # Construct NAP and NHP embeddings
        # NAP_embedding = self.proj_to_transformer_dim(torch.cat([
        #     task_embedding, 
        #     self.learned_next_agent_token
        # ], dim=0))
        
        # NHP_embedding = self.proj_to_transformer_dim(torch.cat([
        #     task_embedding, 
        #     self.learned_next_hint_token
        # ], dim=0))

        NAP_embedding = self.proj_to_transformer_dim_nap(raw_task_embedding)
        NHP_embedding = self.proj_to_transformer_dim_nhp(raw_task_embedding)
        
        # Routing process
        routing_results = {
            "agent_selections": [],
            "hint_selections": [],
            "agent_logits": [],
            "hint_logits": [],
            "agent_outputs_embeddings": []
        }
        routing_count = 0
        final_answers = []
        
        # History states for tracking outputs and embeddings
        history_states = []
        
        # Use inference mode to disable gradient computation
        # import ipdb; ipdb.set_trace()
        while routing_count <= max_routing:
            # Create input for transformer
            if len(history_states) == 0:
                # No history yet, just use roles and special tokens
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    NAP_embedding.unsqueeze(0),
                    NHP_embedding.unsqueeze(0)
                ], dim=0)
            else:
                # Include history embeddings
                history_embeddings = torch.stack([state["embedding"] for state in history_states])
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    history_embeddings,
                    NAP_embedding.unsqueeze(0),
                    NHP_embedding.unsqueeze(0)
                ], dim=0)
            
            # Get encoded tokens from transformer
            num_roles = len(proj_roles_embeddings) # Number of role tokens
            encoded_tokens = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            
            # Debug code: Run inference twice and compare results
            # encoded_tokens_2 = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            # tokens_equal = torch.allclose(encoded_tokens, encoded_tokens_2)
            # print(f"Debug - Encoded tokens equal: {tokens_equal}")
            # if not tokens_equal:
            #     diff = (encoded_tokens - encoded_tokens_2).abs().mean().item()
            #     print(f"Debug - Mean absolute difference: {diff:.6f}")

            # import ipdb; ipdb.set_trace()
            
            # Agent prediction
            nap_idx = len(all_embeddings) - 2  # Second to last token
            encoded_nap = encoded_tokens[nap_idx]
            
            # Force DecisionMaker in the last step
            if routing_count == max_routing:
                chosen_agent_idx = role_to_idx[decision_role]
                chosen_role = decision_role
                # agent_similarity_scores = F.cosine_similarity(encoded_tokens[0:num_roles], encoded_nap.unsqueeze(0), dim=1)
                # use inner product + standardization (mean=0, std=1)
                agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
                mean = agent_similarity_scores.mean()
                std = agent_similarity_scores.std()
                agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)
                agent_similarity_scores = self.cos_scaling  * agent_similarity_scores

            else:
                # Calculate similarity scores for agent selection (indices 0 to num_roles-1)
                # agent_similarity_scores = F.cosine_similarity(encoded_tokens[0:num_roles], encoded_nap.unsqueeze(0), dim=1)
                agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
                mean = agent_similarity_scores.mean()
                std = agent_similarity_scores.std()
                agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)   
                
                # Apply Gumbel softmax for sampling with -3 to 3 input to balance the exploration and exploitation
                agent_similarity_scores = self.cos_scaling  * agent_similarity_scores

                # use softmax if self.cos_scaling > 100
                if self.cos_scaling > 100:
                    agent_probabilities = F.softmax(agent_similarity_scores, dim=0)
                    # import ipdb; ipdb.set_trace()
                else:
                    agent_probabilities = gumbel_softmax(agent_similarity_scores, tau=temperature, hard=True)
                
                # Sample next agent
                chosen_agent_idx = torch.argmax(agent_probabilities).item()
                chosen_role = idx_to_role[chosen_agent_idx] # Direct mapping using 0-based index
            
            # Hint prediction (only if we have history)
            hint_mask = []
            if len(history_states) > 0:
                nhp_idx = len(all_embeddings) - 1  # Last token
                encoded_nhp = encoded_tokens[nhp_idx]
                
                # Calculate similarity scores for hint selection
                hint_offset = num_roles # History starts after role embeddings
                history_encoded = encoded_tokens[hint_offset:hint_offset + len(history_states)]
                hint_similarity_scores = F.cosine_similarity(history_encoded, encoded_nhp.unsqueeze(0), dim=1)
                
                # Use sigmoid for binary decisions (include hint or not)
                hint_probabilities = torch.sigmoid(hint_similarity_scores * self.cos_scaling * 2 )

                # import ipdb; ipdb.set_trace()
                
                # Sample binary decisions
                if self.cos_scaling > 100:
                    hint_mask = (0.5 < hint_probabilities).int()
                else:
                    hint_mask = (torch.rand_like(hint_probabilities) < hint_probabilities).int()

                # set max context length by maskout before last max_context steps
                if len(hint_mask) > max_context:
                    hint_mask[:-max_context] = 0
                
                # Store hint selection results
                routing_results["hint_selections"].append(hint_mask.tolist())
                routing_results["hint_logits"].append(hint_similarity_scores.tolist())
            
            # Construct hints based on the mask
            hints = ""
            if len(hint_mask) > 0 and sum(hint_mask) > 0:
                # hints += "=========================================================Debug Hint: =====================================\n "
                selected_hints = [history_states[i]["output"] for i in range(len(hint_mask)) if hint_mask[i] == 1]
                if selected_hints:
                    for i, hint in enumerate(selected_hints):
                        hints += f"\nAgent {i+1}: {hint} \n"
                # hints += "=========================================================\n"
                # import ipdb; ipdb.set_trace()
            
            # Execute the chosen agent with hints
            # Create a node for the chosen role if not decision maker
            output_embedding = None
            if chosen_role != self.decision_method:
                # Create input with hints
                agent_input = input.copy()
                if hints:
                    agent_input["hints"] = hints
                else:
                    agent_input["hints"] = ""
                
                # Create and execute the agent node
                # import ipdb; ipdb.set_trace()
                agent_node = AgentRegistry.get(agent_group_type, **{"domain": self.domain, "llm_name": self.llm_name, "role": chosen_role})
                await agent_node.async_execute_with_hints(agent_input)
                agent_output = agent_node.outputs[-1] if agent_node.outputs else "No output produced"
                
                
                # Create embedding for this output
                output_embedding = torch.tensor(get_sentence_embedding(agent_output)).to(self.device).detach()
                
                # Create history state
                state_embedding = self.proj_to_transformer_dim_history(torch.cat([
                    role_embeddings[chosen_role],
                    output_embedding
                ], dim=0))
                
                # Add to history
                history_states.append({
                    "role": chosen_role,
                    "output": agent_output,
                    "embedding": state_embedding
                })
                
                # Store in routing results
                routing_results["agent_outputs_embeddings"].append(output_embedding.cpu())
                routing_results["agent_selections"].append(chosen_agent_idx)
                routing_results["agent_logits"].append(agent_similarity_scores.tolist())
            else:
                # If decision maker is chosen, use it as final answer and break
                decision_input = input.copy()
                if hints:
                    decision_input["hints"] = hints
                
                decision_node = AgentRegistry.get(decision_role, **{"domain": self.domain, "llm_name": self.llm_name})
                await decision_node.async_execute_with_hints(decision_input)
                final_answers = decision_node.outputs
                if not final_answers:
                    final_answers = ["No final answer produced by decision maker"]
                
                # Store in routing results
                if final_answers:
                    routing_results["agent_outputs_embeddings"].append(torch.tensor(get_sentence_embedding(final_answers[-1])).detach())
                # else:
                #     # Handle case where final_answers is empty or None
                #     routing_results["agent_outputs_embeddings"].append("DecisionMaker produced no output")
                    
                # Store agent selection results
                routing_results["agent_selections"].append(chosen_agent_idx)
                routing_results["agent_logits"].append(agent_similarity_scores.tolist())
                    
                break
            

            
            # Update routing count
            routing_count += 1
        
        return {
            "answers": final_answers,
            "routing_results": routing_results,
            "routing_count": routing_count
        }

    def run_next_agent_prediction_grad(self, gradient_input, sparse_context=0.0):
        """
        Synchronous version of arun_next_agent_prediction_grad that runs the async function in an event loop.
        
        Args:
            gradient_input: Dictionary containing trace data for gradient calculation
            
        Returns:
            total_loss: Sum of losses across all steps (for monitoring only)
        """
        # Create a new event loop
        loop = asyncio.new_event_loop()
        try:
            # Run the async function in the loop and return its result
            return loop.run_until_complete(self.arun_next_agent_prediction_grad(gradient_input, sparse_context))
        finally:
            # Clean up the loop
            loop.close()

    async def arun_next_agent_prediction_grad(self, gradient_input, sparse_context=0.0):
        """
        Calculate policy gradient for a single trace using the provided gradient input.
        Performs backpropagation and direct parameter updates at each step to prevent memory explosion.
        
        Args:
            gradient_input: Dictionary containing trace data for gradient calculation:
                - task: The original query task
                - advantage: The calculated advantage value for this trace
                - agent_selections: List of agent indices selected during the trace
                - hint_selections: List of hint mask selections during the trace
                - agent_logits: List of agent logits from the original sampling
                - hint_logits: List of hint logits from the original sampling
                - trace_length: Length of the agent selections sequence
                - learning_rate: Learning rate for direct gradient updates (default: 0.01)
                
        Returns:
            total_loss: Sum of losses across all steps (for monitoring only)
        """
        # Extract task query and other inputs
        self.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        query = gradient_input['task']
        advantage = gradient_input['advantage']
        agent_selections = gradient_input['agent_selections']
        hint_selections = gradient_input['hint_selections']
        trace_length = gradient_input['trace_length']
        
        # Return immediately if advantage is zero (no gradient to backprop)
        if abs(advantage) < 1e-8:
            return 0.0
        
        # Prepare embeddings for the task
        task_embedding_raw = torch.tensor(get_sentence_embedding(query))
        task_embedding_raw = task_embedding_raw.detach().to(self.device)
        # Initialize history states for tracking outputs and embeddings
        history_states = []
        
        # Process each step in the trace
        for step_idx in range(trace_length - 1):  # -1 because the last step is decision maker
            task_embedding = self.proj_to_transformer_dim_task(task_embedding_raw)
            # import ipdb; ipdb.set_trace()
            proj_roles_embeddings = []
            for role in self.role_to_idx.keys():
                role_embedding = self.role_embeddings[role].to(self.device)
                # Project role embedding to transformer dimension
                proj_role_embedding = self.proj_to_transformer_dim_role(role_embedding)
                proj_roles_embeddings.append(proj_role_embedding)
            
            
            # Convert to tensor
            proj_roles_embeddings = torch.stack(proj_roles_embeddings)
            
            # Construct NAP and NHP embeddings
            assert torch.is_grad_enabled(), "Grad disabled unexpectedly!"

            # NAP_embedding = self.proj_to_transformer_dim(torch.cat([
            #     task_embedding, 
            #     self.learned_next_agent_token
            # ], dim=0))
            # assert torch.is_grad_enabled(), "Grad disabled unexpectedly!"

            # NHP_embedding = self.proj_to_transformer_dim(torch.cat([
            #     task_embedding, 
            #     self.learned_next_hint_token
            # ], dim=0))

            NAP_embedding = self.proj_to_transformer_dim_nap(task_embedding_raw)
            NHP_embedding = self.proj_to_transformer_dim_nhp(task_embedding_raw)
            
            # Create input for transformer
            if len(history_states) == 0:
                # No history yet, just use roles and special tokens
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),    
                    proj_roles_embeddings,
                    NAP_embedding.unsqueeze(0),
                    NHP_embedding.unsqueeze(0)
                ], dim=0)
            else:
                # Include history embeddings
                concat_embeddings = torch.stack([state["concat_embedding"] for state in history_states])
                # project concat_embeddings to transformer dimension
                history_embeddings = self.proj_to_transformer_dim_history(concat_embeddings)
                all_embeddings = torch.cat([
                    task_embedding.unsqueeze(0),
                    proj_roles_embeddings,
                    history_embeddings,
                    NAP_embedding.unsqueeze(0),
                    NHP_embedding.unsqueeze(0)
                ], dim=0)
            
            # Get encoded tokens from transformer
            num_roles = len(proj_roles_embeddings)
            encoded_tokens = self.routing_transformer(all_embeddings, num_prefix_tokens=num_roles + 1)
            
            # Agent prediction
            nap_idx = len(all_embeddings) - 2  # Second to last token
            encoded_nap = encoded_tokens[nap_idx]
            
            # Calculate similarity scores for agent selection
            # agent_similarity_scores = F.cosine_similarity(encoded_tokens[0:num_roles], encoded_nap.unsqueeze(0), dim=1)
            # use inner product + standardization (mean=0, std=1)
            agent_similarity_scores = (encoded_tokens[0:num_roles] * encoded_nap.unsqueeze(0)).sum(dim=1)
            mean = agent_similarity_scores.mean()
            std = agent_similarity_scores.std()
            agent_similarity_scores = (agent_similarity_scores - mean) / (std + 1e-8)

            # Get logprobs for the agent selection
            agent_logprobs = F.log_softmax(self.cos_scaling * agent_similarity_scores, dim=0)
            
            # Get the selected agent from trace
            selected_agent_idx = agent_selections[step_idx]
            
            # Calculate agent selection loss (negative because we want to maximize reward)
            step_loss = -agent_logprobs[selected_agent_idx] * advantage

            hint_logprob = None # Initialize hint_logprob
            if len(history_states) > 0:
                # Hint prediction (if we have history)
                nhp_idx = len(all_embeddings) - 1  # Last token
                encoded_nhp = encoded_tokens[nhp_idx]
                
                # Calculate similarity scores for hint selection
                hint_offset = num_roles  # History starts after role embeddings
                history_encoded = encoded_tokens[hint_offset:hint_offset + len(history_states)]
                hint_similarity_scores = F.cosine_similarity(history_encoded, encoded_nhp.unsqueeze(0), dim=1)
                # sparse context
                if sparse_context > 0:
                    step_loss = step_loss + sparse_context * hint_similarity_scores.sum()
                # Process each hint selection
                curr_hint_selections = hint_selections[step_idx - 1]  # Shift by 1 since hint_selections starts after first agent
                
                for i, hint_selected in enumerate(curr_hint_selections):
                    # Get probability of selecting this hint (sigmoid)
                    hint_prob = torch.sigmoid(hint_similarity_scores[i] * self.cos_scaling * 2 )
                    
                    # Calculate log probability based on whether hint was selected
                    if hint_selected == 1:
                        # Hint was selected, so we want log(prob)
                        hint_logprob = torch.log(hint_prob + 1e-10)  # Add small epsilon to avoid log(0)
                    else:
                        # Hint was not selected, so we want log(1-prob)
                        hint_logprob = torch.log(1 - hint_prob + 1e-10)
                    
                    # Add to step loss
                    step_loss = step_loss + (-hint_logprob * advantage)

            # Backward pass for this step only
            try:
                with torch.autograd.set_detect_anomaly(True):
                    step_loss.backward()
                    # print("backward pass done")
                    # import ipdb; ipdb.set_trace()
            except RuntimeError as e:
                print(f"ERROR during backward: {e}")
                import ipdb; ipdb.set_trace() # Re-enter debugger if backward fails

            # Add to total loss (for monitoring only, not used for gradients)
            # total_loss += step_loss.item() # Use .item() to detach before summing

            # Detach everything for the next step to free memory
            # Simulate the output that would have been generated
            chosen_role = self.idx_to_role[selected_agent_idx]
            
            if chosen_role != self.decision_method:
                # For non-decision roles, we need to add a state to history
                # Get pre-computed output embedding directly from gradient input
                output_embedding = gradient_input['agent_outputs_embeddings'][step_idx].detach().to(self.device)
                
                # Add to history without detaching to preserve gradients
                history_states.append({
                    "role": chosen_role,
                    "concat_embedding": torch.cat([
                        self.role_embeddings[chosen_role],
                        output_embedding
                    ], dim=0),
                    
                })
        
        return True
    
    def set_train(self):
        self.routing_transformer.train()
        # for param in self.routing_transformer.parameters():
        #     param.requires_grad = True # Be explicit

        self.proj_to_transformer_dim_history.train()
        self.proj_to_transformer_dim_task.train()
        self.proj_to_transformer_dim_role.train()
        self.proj_to_transformer_dim_nap.train()
        self.proj_to_transformer_dim_nhp.train()
        # for param in self.proj_to_transformer_dim.parameters():
        #     param.requires_grad = True # Be explicit

        # Parameters are already tensors, set requires_grad directly
        # if isinstance(self.learned_output_embedding, torch.nn.Parameter):
        #     self.learned_output_embedding.requires_grad = True
        # if isinstance(self.learned_next_agent_token, torch.nn.Parameter):
        #     self.learned_next_agent_token.requires_grad = True
        # if isinstance(self.learned_next_hint_token, torch.nn.Parameter):
        #     self.learned_next_hint_token.requires_grad = True

    def set_eval(self):
        self.routing_transformer.eval()
        # When setting to eval, explicitly set requires_grad to False for safety,
        # although optimizer should handle this during torch.no_grad().
        # for param in self.routing_transformer.parameters():
        #     param.requires_grad = False
        self.proj_to_transformer_dim_history.eval()
        self.proj_to_transformer_dim_task.eval()
        self.proj_to_transformer_dim_role.eval()
        self.proj_to_transformer_dim_nap.eval()
        self.proj_to_transformer_dim_nhp.eval()
        # for param in self.proj_to_transformer_dim.parameters():
        #     param.requires_grad = False

        # if isinstance(self.learned_output_embedding, torch.nn.Parameter):
        #     self.learned_output_embedding.requires_grad = False
        # if isinstance(self.learned_next_agent_token, torch.nn.Parameter):
        #     self.learned_next_agent_token.requires_grad = False
        # if isinstance(self.learned_next_hint_token, torch.nn.Parameter):
        #     self.learned_next_hint_token.requires_grad = False


    def to_device(self, device):
        self.routing_transformer = self.routing_transformer.to(device)
        self.proj_to_transformer_dim_history = self.proj_to_transformer_dim_history.to(device)
        self.proj_to_transformer_dim_task = self.proj_to_transformer_dim_task.to(device)
        self.proj_to_transformer_dim_role = self.proj_to_transformer_dim_role.to(device)
        self.proj_to_transformer_dim_nap = self.proj_to_transformer_dim_nap.to(device)
        self.proj_to_transformer_dim_nhp = self.proj_to_transformer_dim_nhp.to(device)
        for role in self.role_embeddings:
            self.role_embeddings[role] = self.role_embeddings[role].to(device)
        self.device = device

    def add_to_optimizer(self, optimizer):
        self.optimizer = optimizer
        # self.optimizer.add_param_group({"params": self.routing_transformer.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_history.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_task.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_role.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_nap.parameters()})
        self.optimizer.add_param_group({"params": self.proj_to_transformer_dim_nhp.parameters()})
        # Role embeddings are not trainable, so we don't add them to the optimizer

    def save_model(self, path):
        """
        Saves the graph model and all its components to the specified path.
        
        Args:
            path (str): Path where the model should be saved
        """
        model_state = {
            'id': self.id,
            'domain': self.domain,
            'llm_name': self.llm_name,
            'agent_names': self.agent_names,
            'decision_method': self.decision_method,
            'max_routing': self.max_routing,
            'optimized_spatial': self.optimized_spatial,
            'optimized_temporal': self.optimized_temporal,
            'cos_scaling': self.cos_scaling,
            'available_roles': self.available_roles,
            'role_to_idx': self.role_to_idx,
            'idx_to_role': self.idx_to_role,
            'transformer_hidden_dim': self.transformer_hidden_dim
        }
        
        # Save transformer components if present
        if hasattr(self, 'routing_transformer'):
            model_state['routing_transformer'] = self.routing_transformer.state_dict()
            # import ipdb; ipdb.set_trace()
            model_state['proj_to_transformer_dim_history'] = self.proj_to_transformer_dim_history.state_dict()
            model_state['proj_to_transformer_dim_task'] = self.proj_to_transformer_dim_task.state_dict()
            model_state['proj_to_transformer_dim_role'] = self.proj_to_transformer_dim_role.state_dict()
            model_state['proj_to_transformer_dim_nap'] = self.proj_to_transformer_dim_nap.state_dict()
            model_state['proj_to_transformer_dim_nhp'] = self.proj_to_transformer_dim_nhp.state_dict()
        
        # Save GCN components if present
        if hasattr(self, 'gcn') and not hasattr(self, 'routing_transformer'):
            model_state['gcn'] = self.gcn.state_dict()
            model_state['mlp'] = self.mlp.state_dict()
            if hasattr(self, 'spatial_logits'):
                model_state['spatial_logits'] = self.spatial_logits.detach().cpu()
            if hasattr(self, 'spatial_masks'):
                model_state['spatial_masks'] = self.spatial_masks.detach().cpu()
            if hasattr(self, 'temporal_logits'):
                model_state['temporal_logits'] = self.temporal_logits.detach().cpu()
            if hasattr(self, 'temporal_masks'):
                model_state['temporal_masks'] = self.temporal_masks.detach().cpu()
        
        # Save role embeddings
        role_embeddings_dict = {}
        for role, embedding in self.role_embeddings.items():
            role_embeddings_dict[role] = embedding.detach().cpu()
        model_state['role_embeddings'] = role_embeddings_dict
        
        # Save features if present
        if hasattr(self, 'features'):
            model_state['features'] = self.features.detach().cpu()
        
        # Save the state dictionary
        torch.save(model_state, path)

    def clone_for_inference(self):
        """
        Creates a safe copy of the graph for inference without deepcopy issues with tensors.
        Returns a new Graph instance with copied and detached tensors.
        """
        # Create a new graph instance with the same parameters
        new_graph = type(self)(
            domain=self.domain,
            llm_name=self.llm_name,
            agent_names=self.agent_names.copy() if hasattr(self.agent_names, 'copy') else self.agent_names,
            decision_method=self.decision_method,
            optimized_spatial=self.optimized_spatial,
            optimized_temporal=self.optimized_temporal,
            use_transformer=hasattr(self, 'routing_transformer'),
            max_routing=self.max_routing,
            available_roles=self.available_roles.copy() if hasattr(self.available_roles, 'copy') else self.available_roles
        )
        
        # Copy and detach tensors manually
        if hasattr(self, 'routing_transformer'):
            new_graph.routing_transformer.load_state_dict(self.routing_transformer.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_history'):
            new_graph.proj_to_transformer_dim_history.load_state_dict(self.proj_to_transformer_dim_history.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_task'):
            new_graph.proj_to_transformer_dim_task.load_state_dict(self.proj_to_transformer_dim_task.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_role'):
            new_graph.proj_to_transformer_dim_role.load_state_dict(self.proj_to_transformer_dim_role.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_nap'):
            new_graph.proj_to_transformer_dim_nap.load_state_dict(self.proj_to_transformer_dim_nap.state_dict())
            
        if hasattr(self, 'proj_to_transformer_dim_nhp'):
            new_graph.proj_to_transformer_dim_nhp.load_state_dict(self.proj_to_transformer_dim_nhp.state_dict())
            
        # Copy role embeddings
        if hasattr(self, 'role_embeddings'):
            new_graph.role_embeddings = {
                role: embedding.clone().detach() 
                for role, embedding in self.role_embeddings.items()
            }

        # clone the cos_scaling
        new_graph.cos_scaling = self.cos_scaling
        new_graph.routing_transformer.eval()
        return new_graph

    @staticmethod
    def load_model(path):
        """
        Loads a graph model from the specified path.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            Graph: A loaded Graph instance
        """
        # Load the state dictionary
        model_state = torch.load(path, map_location=torch.device('cpu'))
        
        # Create a new instance of the Graph class
        graph = Graph(
            domain=model_state['domain'],
            llm_name=model_state['llm_name'],
            agent_names=model_state['agent_names'],
            decision_method=model_state['decision_method'],
            optimized_spatial=model_state['optimized_spatial'],
            optimized_temporal=model_state['optimized_temporal'],
            use_transformer='routing_transformer' in model_state,
            max_routing=model_state['max_routing'],
            available_roles=model_state['available_roles']
        )
        
        # Restore ID and other attributes
        graph.id = model_state['id']
        graph.cos_scaling = model_state['cos_scaling']
        graph.role_to_idx = model_state['role_to_idx']
        graph.idx_to_role = model_state['idx_to_role']
        
        # Load transformer components if present
        if 'routing_transformer' in model_state:
            graph.transformer_hidden_dim = model_state['transformer_hidden_dim']
            graph.routing_transformer.load_state_dict(model_state['routing_transformer'])
            graph.proj_to_transformer_dim_history.load_state_dict(model_state['proj_to_transformer_dim_history'])
            graph.proj_to_transformer_dim_task.load_state_dict(model_state['proj_to_transformer_dim_task'])
            graph.proj_to_transformer_dim_role.load_state_dict(model_state['proj_to_transformer_dim_role'])
            graph.proj_to_transformer_dim_nap.load_state_dict(model_state['proj_to_transformer_dim_nap'])
            graph.proj_to_transformer_dim_nhp.load_state_dict(model_state['proj_to_transformer_dim_nhp'])
        
        # Load GCN components if present
        if 'gcn' in model_state:
            graph.gcn.load_state_dict(model_state['gcn'])
            graph.mlp.load_state_dict(model_state['mlp'])
            if 'spatial_logits' in model_state:
                graph.spatial_logits = torch.nn.Parameter(model_state['spatial_logits'])
            if 'spatial_masks' in model_state:
                graph.spatial_masks = torch.nn.Parameter(model_state['spatial_masks'])
            if 'temporal_logits' in model_state:
                graph.temporal_logits = torch.nn.Parameter(model_state['temporal_logits'])
            if 'temporal_masks' in model_state:
                graph.temporal_masks = torch.nn.Parameter(model_state['temporal_masks'])
        
        # Load role embeddings
        role_embeddings_dict = model_state['role_embeddings']
        graph.role_embeddings = {}
        for role, embedding in role_embeddings_dict.items():
            graph.role_embeddings[role] = embedding
        
        # Load features if present
        if 'features' in model_state:
            graph.features = model_state['features']
        
        return graph

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val == max_val:
        return torch.zeros_like(tensor)
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    

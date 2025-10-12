import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List,Union,Literal
import random
import numpy as np
from tqdm import tqdm
# from sklearn.model_selection import train_test_split # Remove train_test_split import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from GDesigner.utils.globals import Time
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets_my.gsm8k_dataset import gsm_data_process,gsm_get_predict
# from GDesigner.embedding.embedding import get_sentence_embedding

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    start_index = i_batch * batch_size
    end_index = start_index + batch_size
    if start_index >= len(data_list):
        return None
    return data_list[start_index:min(end_index, len(data_list))]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets_my/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="deepseek-r1:1.5b-qwen-distill-fp16")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=1e-5,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--epochs', type=int, default=500,help="The max number of training epochs.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['Math Solver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    # Add new arguments for arun_with_routing
    parser.add_argument('--max_routing', type=int, default=5,
                        help='Maximum number of routing steps')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature for next agent prediction')
    parser.add_argument('--decay_factor', type=float, default=1,
                        help='Decay factor for routing')
    parser.add_argument('--max_time', type=int, default=5,
                        help='Maximum time for routing')
    parser.add_argument('--num_traces', type=int, default=8, help="Number of traces to sample per question for training")
    parser.add_argument('--train_num', type=int, default=80, help="Number of training samples")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--reuse_time', type=int, default=10, help="Number of times to reuse collected traces for training")
    parser.add_argument('--training_samples', type=int, default=1000,help="The number of training samples.")
    parser.add_argument('--required_correct_answers', type=int, default=1, help="Number of required correct answers")
    # available_roles
    parser.add_argument('--sparse_context', type=float, default=0, help="Sparse context")
    parser.add_argument('--max_context', type=int, default=5, help="Maximum context")
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set PyTorch deterministic algorithms where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
    
    # Split dataset into training and testing sets manually
    train_dataset = dataset[:args.train_num]
    # leave only the ninth
    # train_dataset = train_dataset[8:9]
    test_dataset = dataset[args.train_num:]

    # only leave first 20 for test
    # test_dataset = test_dataset[:20]
    
    # print length of train_dataset and test_dataset
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    print(f"Dataset split: {len(train_dataset)} training samples, {len(test_dataset)} testing samples.")

    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))
    
    # use transformer and max_routing
    graph = Graph(domain="gsm8k",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  use_transformer = True,
                  max_routing = args.max_routing,
                  available_roles=["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"],
                  **kwargs)
    
    # Set transformer to train mode initially
    graph.routing_transformer.train()
    optimizer = torch.optim.AdamW(graph.routing_transformer.get_gpt_parameters(), lr=args.lr)
    graph.add_to_optimizer(optimizer)

    # save model and load model
    # graph.save_model(result_dir / f"{current_time}_{args.llm_name}_model.pth")
    # graph.load_model(result_dir / f"{current_time}_{args.llm_name}_model.pth")



    
    # Open log file in append mode
    log_file_path = result_dir / f"log_trail_{args.llm_name}_{current_time}.txt"
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"--- Starting Training Run: {current_time} ---\n")
        log_file.flush()
        
        # --- Training Loop ---
        print("--- Starting Training ---")
        training_samples = args.training_samples
        for  epoch    in range(1, args.epochs+1):
            print(f"Training Epoch {epoch}/{args.epochs}")
            log_file.write(f"Training Epoch {epoch}/{args.epochs}\n")
            log_file.flush()
            
            # Determine sampling parameters for this epoch
            eval_interval = 100
            if epoch % eval_interval == 0:
                log_file.write(f"Evaluating on training set\n")
                log_file.flush()
                graph.cos_scaling = 1e3
                num_traces = 1

                # continue
            else:
                graph.cos_scaling = 1.5
                # import ipdb; ipdb.set_trace()
                num_traces = args.num_traces
            
            # ======== PHASE 1: Collect traces for all training samples ========
            all_gradient_inputs = []
            total_epoch_reward = 0.0
            trace_count = 0
            
            print("Collecting traces for all training samples...")
            log_file.write("Collecting traces for all training samples...\n")
            log_file.flush()
            
            graph.set_eval()  # Set to eval mode for sampling
            with torch.no_grad():  # Disable gradients during sampling
                for i_record, record in enumerate(train_dataset):
                    task = record["task"]
                    true_answer = record["answer"]
                    input_dict = {"task": task}
                    
                    print(f"  Sampling traces for question {i_record+1}/{len(train_dataset)}")
                    log_file.write(f"  Sampling traces for question {i_record+1}/{len(train_dataset)}\n")
                    log_file.flush()
                    
                    # Collect traces for this question until we find two correct answers    
                    question_traces = []
                    question_rewards = []
                    question_rewards_raw = []
                    found_correct = 0
                    
                    for i_trace in range(num_traces):
                        if found_correct >= args.required_correct_answers:
                            # import ipdb; ipdb.set_trace()
                            break

                        if epoch % eval_interval != 0:
                            training_samples -= 1
                            if training_samples % 10 == 0:
                                print(f"Training samples left: {training_samples}")
                                log_file.write(f"Training samples left: {training_samples}\n")
                            if training_samples < 0:
                                break
                            
                        trace_result = graph.run_next_agent_prediction(
                            input_dict,
                            max_routing=args.max_routing,
                            temperature=args.temperature,
                            available_roles=["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"],
                            training=True
                        )

                        if trace_result is None:
                            log_file.write(f"   Warning: Trace result is None because of timeout\n")
                            log_file.flush()
                            continue
                    
                        
                        # Extract answer and check if correct
                        predict_answer_list = trace_result.get("answers", [""])
                        predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
                        predict_answer_val = gsm_get_predict(predict_answer_str)
                        routing_length = trace_result.get("routing_count", args.max_routing)
                        
                        # Calculate success and reward
                        is_solved = float(predict_answer_val) == float(true_answer)
                        reward = (args.decay_factor ** routing_length) * float(is_solved)
                        
                        
                        question_traces.append(trace_result["routing_results"])
                        question_rewards.append(reward)
                        question_rewards_raw.append(float(is_solved))
                        
                        if is_solved:
                            found_correct += 1
                            log_file.write(f"    Found {found_correct} correct answers on trace {i_trace+1}\n")
                            log_file.flush()

                        # Log first agent logits for debugging
                        agent_logits = trace_result["routing_results"].get('agent_logits', [])
                        # import ipdb; ipdb.set_trace()
                        if i_trace == 0 and len(agent_logits) > 0:
                            log_file.write(f"First agent logits: {agent_logits[0]}\n")
                            # log the routing length
                            log_file.write(f"Routing length: {routing_length}\n")
                            log_file.flush()

                    if training_samples < 0:
                        break
                    # Skip if no traces were collected (shouldn't happen but just in case)
                    if not question_traces:
                        continue
                    
                    # Calculate advantage for each trace
                    # If only one trace, no normalization needed
                    advantages = None
                    if len(question_traces) == 1:
                        advantages = np.array([0])
                    else:
                        baseline = np.mean(question_rewards)
                        advantages = np.array(question_rewards) - baseline
                        # Normalize advantages
                        std = np.std(advantages)
                        if std > 1e-8:
                            advantages = advantages / std
                    
                    
                    # check if all advantages are smaller than 1e-8
                    store_traces = True
                    if all(abs(advantage) < 1e-8 for advantage in advantages):
                        store_traces = False

                    if store_traces:
                        # Create gradient inputs for each trace
                        for i_trace in range(len(question_traces)):
                            trace_data = question_traces[i_trace]
                            agent_selections = trace_data['agent_selections']
                            hint_selections = trace_data['hint_selections']
                            agent_logits = trace_data.get('agent_logits', [])
                            hint_logits = trace_data.get('hint_logits', [])
                            agent_outputs_embeddings = trace_data.get('agent_outputs_embeddings', [])
                            
                            
                            
                            # Create a dictionary with all necessary inputs for gradient calculation
                            # make follwing to cpu mem
                            # import ipdb; ipdb.set_trace()
                            gradient_input = {
                                'task': task,
                                'advantage': advantages[i_trace],
                                'agent_selections': agent_selections,
                                'hint_selections': hint_selections,
                                'agent_logits': agent_logits,
                                'hint_logits': hint_logits,
                                'trace_length': len(agent_selections),
                                'agent_outputs_embeddings': agent_outputs_embeddings,
                                'batch_record_idx': i_record,
                                'trace_idx_in_record': i_trace
                            }

                            # import ipdb; ipdb.set_trace()
                            all_gradient_inputs.append(gradient_input)
                        
                    # Track rewards
                    total_epoch_reward += question_rewards_raw[0]
                    trace_count += 1
            # Calculate average reward for this sampling phase
            avg_epoch_reward = total_epoch_reward / trace_count if trace_count > 0 else 0
            # Update EMA reward

            # 

            print(f"Sampling complete. Collected {len(all_gradient_inputs)} traces.")
            print(f"Average Reward: {avg_epoch_reward:.4f} Epoch {epoch+1} complete.")
            log_file.write(f"Sampling complete. Collected {len(all_gradient_inputs)} traces.\n")
            log_file.write(f"Average Reward: {avg_epoch_reward:.4f} Epoch {epoch+1} complete.\n")
            log_file.flush()
            
            # ======== PHASE 2: Training with collected traces ========
            if epoch % eval_interval != 0 and len(all_gradient_inputs) > 0:  # Skip training on evaluation epochs
                print("Training network with collected traces...")
                log_file.write("Training network with collected traces...\n")
                log_file.flush()
                
                # Set model to train mode
                graph.set_train()
                
                # Training loop - reuse traces multiple times use tqdm
                for reuse_iter in tqdm(range(args.reuse_time), desc="Training Reused Epochs"):
                    print(f"  Training Reused Epochs {reuse_iter+1}/{args.reuse_time}")
                    log_file.write(f"  Training Reused Epochs {reuse_iter+1}/{args.reuse_time}\n")
                    log_file.flush()
                    
                    # Shuffle gradient inputs for this iteration
                    random_indices = list(range(len(all_gradient_inputs)))
                    random.shuffle(random_indices)
                    
                    # Process in batches with gradient accumulation
                    batch_size = args.batch_size
                    num_batches = (len(random_indices) + batch_size - 1) // batch_size
                    
                    for batch_idx in tqdm(range(num_batches), desc="Training batches"):
                        # Get batch indices
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(random_indices))
                        batch_indices = random_indices[start_idx:end_idx]
                        
                        # Zero gradients at the start of each batch
                        optimizer.zero_grad()
                        
                        # Process each trace in the batch
                        for idx in batch_indices:
                            gradient_input = all_gradient_inputs[idx]
                            # Calculate loss for this trace (which handles its own backpropagation)
                            graph.run_next_agent_prediction_grad(gradient_input)
                        
                        # Update parameters after processing the entire batch
                        optimizer.step()
                        
                        if batch_idx % 10 == 0:  # Log every 10 batches
                            print(f"    Processed batch {batch_idx+1}/{num_batches}")
                            log_file.write(f"    Processed batch {batch_idx+1}/{num_batches}\n")
                            log_file.flush()
            
            # Log epoch summary
            print(f"Epoch {epoch + 1} complete. Average Accuracy: {avg_epoch_reward:.4f}")
            log_file.write(f"Epoch {epoch + 1} complete. Average Accuracy: {avg_epoch_reward:.4f}\n")
            # log the total reward and count
            log_file.write(f"Total Reward: {total_epoch_reward:.4f} Total Count: {trace_count}\n")
            log_file.write("-" * 80 + "\n")
            log_file.flush()
            if training_samples < 0:
                break   


    # save model
    try:
        graph.save_model(result_dir / f"{current_time}_{args.llm_name}_model.pth")
    except Exception as e:
        print(f"Error saving model: {e}")

    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        print("--- Training Finished ---")
        log_file.write("--- Training Finished ---\n")

        # Save results to file
        # Save results to file
        data = load_result(result_file)
        try:
            result_entry = {
                "time": current_time,
                "llm_name": args.llm_name,
                "domain": args.domain,
                "Training Cost": Cost.instance().value,
                "Training Prompt Tokens": PromptTokens.instance().value,
                "Training Completion Tokens": CompletionTokens.instance().value,
            }
            data.append(result_entry)
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error saving training tokens results: {e}")
            log_file.write(f"Error saving training tokens results: {e}\n")
            log_file.flush()
        
        


        # Test the model on the test set
        print("Testing the model on the test set...")
        log_file.write("Testing the model on the test set...\n")
        log_file.flush()

        graph.set_eval()

        # test one by one
        total_solved, total_executed = (0, 0)
        graph.cos_scaling = 1e3
        with torch.no_grad(): # Disable gradient calculations for testing
            for i_record in range(len(test_dataset)):
                data = load_result(result_file)
                print(f"  Testing question {i_record + 1}/{len(test_dataset)}", 80*'-')
                start_ts = time.time()
                
                current_record = test_dataset[i_record]
                
                # Create parallel inference tasks for each question in the batch
                task = current_record["task"]
                input_dict = {"task": task}
                
                # Add inference task
                inference_result = graph.run_next_agent_prediction(
                    input_dict,
                    max_routing=args.max_routing,
                    temperature=args.temperature,
                    available_roles=["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"]
                )

                # Process results
                true_answer = current_record["answer"]
                # Extract answer and check if correct
                predict_answer_list = inference_result.get("answers", [""])
                predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
                predict_answer_val = gsm_get_predict(predict_answer_str)
                routing_length = inference_result.get("routing_count", args.max_routing)
                    
                # Check if solved
                is_solved = float(predict_answer_val) == float(true_answer)
                if is_solved:
                    total_solved += 1
                total_executed += 1

                # current accuracy
                current_accuracy = total_solved / total_executed if total_executed > 0 else 0
                print(f"Current Test Accuracy: {current_accuracy:.4f} ({total_solved}/{total_executed} solved)")

                updated_item = {
                    "Question": task,
                    "Answer": true_answer,
                    "Routing_length": routing_length,
                    "Response": predict_answer_str,
                    "Attempt answer": predict_answer_val,
                    "Solved": is_solved,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": current_accuracy
                }
                data.append(updated_item)

                with open(result_file, 'w',encoding='utf-8') as file:
                    json.dump(data, file, indent=4)
                
        # Calculate and print final test accuracy
        final_accuracy = total_solved / total_executed if total_executed > 0 else 0
        print(f"Final Test Accuracy: {final_accuracy:.4f} ({total_solved}/{total_executed} solved)")

        # Save results to file
        data = load_result(result_file)
        result_entry = {
            "time": current_time,
            "llm_name": args.llm_name,
            "mode": args.mode,
            "domain": args.domain,
            "agent_names": args.agent_names,
            "agent_nums": args.agent_nums,
            "num_rounds": args.num_rounds,
            "decision_method": args.decision_method,
            "test_accuracy": float(final_accuracy),
            "total_solved": total_solved,
            "total_executed": total_executed,
            "cost": Cost.instance().value,
            "prompt_tokens": PromptTokens.instance().value,
            "completion_tokens": CompletionTokens.instance().value,
        }
        data.append(result_entry)
        
        with open(result_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    
    


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        # No need for random.shuffle here as seeds are already set in main()
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        # Use deterministic random values since seed is already set in main()
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    main()

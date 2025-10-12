import torch
from typing import Iterator, List
import pandas as pd
import numpy as np
import time
import random
import copy
from GDesigner.utils.globals import Time
from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from tqdm import tqdm
def train(graph: Graph,
          dataset,
          result_dir: str = None,
          args = None
          ) -> None:
    
    def get_random_samples(dataset, sample_size=80) -> List:
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        return [dataset[idx] for idx in indices]
    
    random_samples = get_random_samples(dataset, sample_size=args.train_num)
    
    
    graph.routing_transformer.train()
    optimizer = torch.optim.AdamW(graph.routing_transformer.get_gpt_parameters(), lr=args.lr)
    graph.add_to_optimizer(optimizer)
    
    # Open log file in append mode
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file_path = result_dir / f"log_trail_{args.llm_name}_{current_time}.txt"
    
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"--- Starting Training Run: {current_time} ---\n")
        log_file.flush()
        
        # --- Training Loop ---
        print("--- Starting Training ---")
        training_samples = args.training_samples
        for epoch in range(1, args.epochs+1):
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
                for i_record, record in enumerate(random_samples):
                    input_dict = dataset.record_to_input(record)
                    print(input_dict)
                    
                    print(f"  Sampling traces for question {i_record+1}/{len(random_samples)}")
                    log_file.write(f"  Sampling traces for question {i_record+1}/{len(random_samples)}\n")
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
                            available_roles=['Knowlegable Expert',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'],
                            agent_group_type="AnalyzeAgent",
                            training=True,
                            max_context=args.max_context
                        )

                        if trace_result is None:
                            log_file.write(f"   Warning: Trace result is None because of timeout\n")
                            log_file.flush()
                            continue
                    
                        
                        # Extract answer and check if correct
                        predict_answer_list = trace_result.get("answers", [""])
                        predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
                        predict_answer_val = dataset.postprocess_answer(predict_answer_str)

                        
                        routing_length = trace_result.get("routing_count", args.max_routing)
                        
                        # Calculate success and reward
                        true_answer = dataset.record_to_target_answer(record)
                        is_solved = predict_answer_val == true_answer
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
                            # log the agent selections
                            log_file.write(f"Agent selections: {trace_result['routing_results']['agent_selections']}\n")
                            # log the hint selections
                            log_file.write(f"Hint selections: {trace_result['routing_results']['hint_selections']}\n")
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
                                'task': input_dict["task"],
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
                            graph.run_next_agent_prediction_grad(gradient_input, sparse_context=args.sparse_context)
                        
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
        
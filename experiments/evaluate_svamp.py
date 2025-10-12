import os
import json
import math
import time
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy
from GDesigner.utils.globals import Time
from GDesigner.graph.graph import Graph
from experiments.accuracy import Accuracy
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from GDesigner.tools.coding.python_executor import PyExecutor
import torch
from datasets_my.gsm8k_dataset import gsm_data_process,gsm_get_predict
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data
def evaluate(
        graph:Graph,
        dataset,
        result_file = None,
        result_dir = None,
        args = None
        ) -> float:

    
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    log_file_path = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.log"
    
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        print("--- Training Finished ---")
        log_file.write("--- Training Finished ---\n")

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
            print(f"Error saving Training tokens results: {e}")
            log_file.write(f"Error saving Training tokens results: {e}\n")
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
            for i_record, record in enumerate(dataset):
                data = load_result(result_file)
                print(f"  Testing question {i_record + 1}/{len(dataset)}", 80*'-')
                start_ts = time.time()
                
                current_record = record
                
                # Create parallel inference tasks for each question in the batch
                # task = current_record["task"]
                # input_dict = {"task": task}
                task = record["question_concat"]
                # import ipdb; ipdb.set_trace()
                true_answer = record["Answer"]
                input_dict = {"task": task}
                
                # Add inference task
                inference_result = graph.run_next_agent_prediction(
                    input_dict,
                    max_routing=args.max_routing,
                    temperature=args.temperature,
                    available_roles=["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector"],
                    agent_group_type="MathSolver",
                    max_context=args.max_context
                )

                # Process results
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


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)

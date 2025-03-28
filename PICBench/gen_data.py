import sys
import os
from openai import OpenAI
from datetime import datetime
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from evaluation import evaluate
from pathlib import Path
from agent import LLMAgent
from scipy.special import comb


def cal_passk(dic_list, n, k):
    print(dic_list)
    # syntax
    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]['syntax check passed'] + dic_list[design]['functional check passed']
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    syntax_passk = sum(sum_list) / len(sum_list)

    # func
    sum_list = []
    for design in dic_list.keys():
        c = dic_list[design]['functional check passed']
        sum_list.append(1 - comb(n - c, k) / comb(n, k))
    func_passk = sum(sum_list) / len(sum_list)
    print(f'syntax pass@{k}: {syntax_passk},   func pass@{k}: {func_passk}')


def get_dict(file_path):
    with open(file_path, 'r',encoding='gbk') as file:
        return json.load(file)

def extract_design_names(directory_path):
    try:
        # Get a list of all subdirectories in the specified directory
        design_names = [folder_name for folder_name in os.listdir(directory_path)
                      if os.path.isdir(os.path.join(directory_path, folder_name))]

        return design_names
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def load_design_dict(design_names):
    instruction_dict = {}
    for design_name in design_names:
        base_path = Path("../testcases") / design_name
        file_path = base_path / f"{design_name}_problem_description.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                instruction_dict[design_name] = file.read().split('Problem Description-')[1]
        except FileNotFoundError:
            print(f"Error: Problem description file not found for {design_name}.")
            return ""
    return instruction_dict


def gen_result_feedback_passk(LLM_agent, design_name, design_dict, prompt, restriction_label, num_feedback, k, n):
    error_dict_init = {
            'syntax check passed': 0,
            'functional check passed': 0
        }
    netlist_responses = {}
    netlist_responses_single = {}
    full_responses = {}
    dict_list = {}
    for i in range(0, len(design_name)):
        dict_list[design_name[i]] = error_dict_init.copy()
        feedback_flag = True
        full_responses[design_name[i]] = []
        netlist_responses[design_name[i]] = []
        print(f"design_name: {design_name[i]}")
        for times in range(0, n):
            single_flag = False
            if times == 0:
                netlist_responses_single[design_name[i]] = []
                single_flag = True
            print("cnt: ", times)
            label = True
            LLM_agent.start_new_conversation()
            question = design_dict[design_name[i]] + '\n' + prompt
            with open("system_prompt.txt", 'r') as file:
                system_prompt = file.read()
            with open("restrictions.txt", 'r') as file:
                restrictions = file.read()
            full_system_prompt = system_prompt + '\n' + restrictions
            if restriction_label:
                response = LLM_agent.ASK_LLM_iterate(full_system_prompt, question)
            else:
                response = LLM_agent.ASK_LLM_iterate(system_prompt, question)
            print(f"init response_{times}: {response}")
            try:
                netlist = response.split('<result>\n')[1]
                if netlist.split('\n')[-1] == '</result>':
                    netlist = '\n'.join(netlist.split('\n')[:-1])
                elif netlist.split('\n')[-2] == '</result>':
                    netlist = '\n'.join(netlist.split('\n')[:-2])
                elif netlist.split('\n')[0] == '```json':
                    netlist = '\n'.join(netlist.split('\n')[1:-1])
                elif netlist.split('\n')[-1] == '```':
                    netlist = '\n'.join(netlist.split('\n')[:-1])
                if netlist:
                    netlist_responses[design_name[i]].append(netlist)
                    if single_flag:
                        netlist_responses_single[design_name[i]].append(netlist)
                    eval_res = evaluate(netlist, design_name[i])
                else:
                    eval_res = f'no result part, please make sure the result is in the correct format: <analysis> \n analysis \n <result> \n result'
            except:
                eval_res = f'no result part, please make sure the result is in the correct format: <analysis> \n analysis \n <result> \n result'
            print(f'eval_{design_name[i]}_init_{times}: {eval_res}')
            full_responses[design_name[i]].append(response)

            if eval_res.startswith('functional check passed'):
                dict_list[design_name[i]]['functional check passed'] += 1
                feedback_flag = False
            elif eval_res.startswith('functional error'):
                dict_list[design_name[i]]['syntax check passed'] += 1
                label = False

            if feedback_flag:
                for cnt in range(0, num_feedback):
                    refine_prompt = "Here are the errors in previously generated code. \n Please follow the restrictions and write entire code by fixing the errors in previous code. \nPlease only give me the code in the <result> part, for anything beside the code, please properly comment it out in <analysis> part."
                    question = refine_prompt + '\n' + eval_res
                    response = LLM_agent.ASK_LLM_iterate(full_system_prompt, question)
                    print(f"feedback {cnt} response: {response}")
                    try:
                        netlist = response.split('<result>\n')[1]
                        if netlist.split('\n')[-1] == '</result>':
                            netlist = '\n'.join(netlist.split('\n')[:-1])
                        elif netlist.split('\n')[-2] == '</result>':
                            netlist = '\n'.join(netlist.split('\n')[:-2])
                        elif netlist.split('\n')[0] == '```json':
                            netlist = '\n'.join(netlist.split('\n')[1:-1])
                        elif netlist.split('\n')[-1] == '```':
                            netlist = '\n'.join(netlist.split('\n')[:-1])
                        if netlist:
                            netlist_responses[design_name[i]].append(netlist)
                            if single_flag:
                                netlist_responses_single[design_name[i]].append(netlist)
                            eval_res = evaluate(netlist, design_name[i])
                        else:
                            eval_res = f'no result part, please make sure the result is in the correct format: <analysis> \n analysis \n <result> \n result'
                    except:
                        eval_res = f'no result part, please make sure the result is in the correct format: <analysis> \n analysis \n <result> \n result'
                    print(f'eval_{design_name[i]}_{times}_feedback_{cnt}: {eval_res}')
                    full_responses[design_name[i]].append(response)

                    if eval_res.startswith('functional check passed'):
                        dict_list[design_name[i]]['functional check passed'] += 1
                        break
                    elif eval_res.startswith('functional error') and label:
                        dict_list[design_name[i]]['syntax check passed'] += 1
                        label = False

    cal_passk(dict_list, n, k)

    with open(f'log_{args.model}_pass{k}_res{num_feedback}.json', 'w') as file:
        json.dump(netlist_responses, file)


def PICBench(path, restriction_label, max_iterations, pass_k, total_samples, model, api_key):
    """
    Main function to process the workflow.

    :param path: Path to the test cases and solutions.
    :param restriction_label: Boolean indicating if restrictions apply.
    :param max_iterations: Maximum iterations of error feedback.
    :param pass_k: Pass@k value.
    :param total_samples: The total number of samples generated.
    :param model: The model to be used (e.g., GPT-4, Gemini).
    :param api_key: API key for the LLM agent.
    """
    print(f"[INFO] Starting PICBench...")
    print('-----------------------------------------------------')
    print(f"[INFO] Parameters received:")
    print(f"  Path: {path}")
    print(f"  Restriction Label: {restriction_label}")
    print(f"  Max Iterations: {max_iterations}")
    print(f"  Pass@k: {pass_k}")
    print(f"  Total Samples: {total_samples}")
    print(f"  Model: {model}")
    print(f"  API Key: {'[HIDDEN]' if api_key else '[MISSING]'}")
    print('-----------------------------------------------------')

    problem_prompt = "Design the circuit and generate the corresponding JSON netlist based on the description. Try to understand the requirements and give reasoning steps in natural language to achieve it. In addition, try to give advice to avoid syntax error based on the restrictions."

    # Initialize the LLM agent with the specified model
    LLM_agent = LLMAgent(api_key=api_key, model=model)

    # Use existing functions to process the designs
    design_names = extract_design_names(path)
    instruction_dict = load_design_dict(design_names)

    # Generate feedback and results
    gen_result_feedback_passk(
        LLM_agent, design_names, instruction_dict, problem_prompt, restriction_label, max_iterations, pass_k, total_samples
    )


if __name__ == '__main__':
    api_key = your own api key
    
    import argparse
    
    parser = argparse.ArgumentParser(description="PIC design generation and evaluation")
    parser.add_argument("--path", type=str, default="../testcases", help="Path to the test cases.")
    parser.add_argument("--restriction_label", type=bool, default=True, help="Apply restrictions (True/False).")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum iterations of error feedback.")
    parser.add_argument("--pass_k", type=int, default=1, help="Pass@k.")
    parser.add_argument("--total_samples", type=int, default=5, help="Total number n of samples to generate.")
    parser.add_argument("--model", type=str, default="gpt-4", help="Language model to use.")
    parser.add_argument("--api_key", type=str, default=api_key, help="API key for the LLM.")

    args = parser.parse_args()

    # Execute the main function
    PICBench(
        path=args.path,
        restriction_label=args.restriction_label,
        max_iterations=args.max_iterations,
        pass_k=args.pass_k,
        total_samples=args.total_samples,
        model=args.model,
        api_key=args.api_key
    )

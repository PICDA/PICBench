import json
from devices import *
import numpy as np


def normalize_array(array, precision=10):
    return tuple(np.round(array, precision))  # Round each element to the desired precision


def compare_golden(set1, golden):
    # Convert each array to a tuple (to preserve order) and then compare as sets
    ans = 'functional check passed'
    set1_normalized = {normalize_array(array) for array in set1}
    set2_normalized = {normalize_array(array) for array in golden}
    if set1_normalized != set2_normalized:
        ans = 'functional error, The syntax is correct, but a functional error has occurred. Please review the problem description carefully.'

    # Compare the sets of tuples
    return ans


def evaluate(netlist, design_name):
    try:
        data = json.loads(netlist)
    except Exception as e:
        # file start with '''json
        if 'Expecting value' in str(e):
            return f"other syntax error, Extra contents found in JSON, The file starts with ```json or other notations, which is not part of valid JSON. Remove the prefix and try again. {e}"
        elif '//' in netlist:
            return f'other syntax error, Extra contents found in JSON, The file contains // for comments, which is not part of valid JSON. Remove the comments and try again. {e}'
        elif '...' in netlist:
            return f'other syntax error, Please do not omit any code, generate complete code that can be compiled directly. {e}'
        elif 'Extra data:' in str(e):
            return f"other syntax error, Extra contents found in JSON, The file ends with some other extra contents (like advice/comments/summary), which is not part of valid JSON. Only JSON code is required in the result part. Remove the suffix and try again. {e}"
        elif 'Expecting property name' in str(e):
            return f'other syntax error, Trailing commas are not allowed in JSON objects or arrays. Please remove the extra comma, {e}'
        elif 'does not contain port' in str(e):
            return f'wrong ports error, {e}'
        elif 'Missing models' in str(e):
            return f'Mess up ‘instances’ and ‘models’ part, {e}'
        elif 'Value error, Invalid port string' in str(e):
            return f'Bind the I/O ports, {e}'
        else:
            return f'wrong model names, {e}'
    try:
        model_data = {key: globals()[value] for key, value in data["models"].items()}
    except Exception as e:
        # wrong model names
        return f'wrong model names, {e}'
    try:
        design, _ = sax.circuit(netlist=data["netlist"], models=model_data)
        wl = jnp.linspace(1.51, 1.59, 1000)
        S = design(wl=wl)
        ports = data["netlist"]["ports"]
        inputs = {key: value for key, value in ports.items() if key.startswith('I')}
        outputs = {key: value for key, value in ports.items() if key.startswith('O')}
        in_port = list(inputs.keys())
        out_port = list(outputs.keys())
        result = []
        for i in in_port:
            for o in out_port:
                trans = abs(S[i, o]) ** 2
                result.append(trans.tolist())

        # result = set(result)
        with open(f"../testcases/{design_name}/{design_name}_res.json", "r") as f:
            golden_res = json.load(f)
            # get all the values
            golden = [value for key, value in golden_res.items()]
            # golden = set(golden)
        if len(result) != len(golden):
            ports = [key for key, value in golden_res.items()]
            in_port_set = set()
            out_port_set = set()
            for i in range(len(ports)):
                in_port_set.add(ports[i].split('_')[0])
                out_port_set.add(ports[i].split('_')[1])
            # 'wrong ports number', the number of input should be {len(golden_res)} but got {len(result)}
            res = (f'wrong ports number, the number of input should be {len(in_port_set)} got {len(in_port)}, '
                   f'the number of output should be {len(out_port_set)} got {len(out_port)}'
                   f'And note that the input should start with I and the output should start with O')

        else:
            # functional check
            res = compare_golden(result, golden)
    except Exception as e:
        # syntax error
        res = f'other syntax error, {e}'
    return res

# PICBench: Benchmarking LLMs for Photonic Integrated Circuits Design

## project composition

### **Testbench**
A collection of 24 meticulously crafted design problems. Each design is organized into a folder containing:
- **Description.txt**: A natural language description of the design, including required configurations and the number of input and output ports.
- **netlist.json**: The golden reference design.
- **Frequency_response.json**: The corresponding frequency response of `netlist.json`, generated using the SAX simulator.

### **PICBench Core Modules**
- **Restrictions**: Manually summarized restrictions to improve LLM-generated designs.
- **System_prompt**: The system prompt used for all models.
- **Devices**: Parametrized models for components required in the design process.
- **Agent**: Code to interface with the LLM API.
- **Evaluation**: Evaluation using SAX.
- **gen_data**: The main function for automating code generation and evaluation, providing Pass@k metrics.

## **Setup**
To use **PICBench**, ensure you have the following:
1. Python version **>= 3.10**.
2. Install [SAX](https://github.com/flaport/sax) for circuit simulation:
   ```bash
   pip install sax jax openai anthropic google-generativeai
   ```
**Note**: Older or newer versions of these packages may not be compatible with PICBench. If you encounter issues, use the tested versions below:
   
   ```bash
   pip install sax==0.13.3 jax==0.4.34 openai==1.43.0 anthropic==0.43.0 google-generativeai==0.8.3
   ```   

## **Usage**
You can run the PICBench via command line:

```bash
python gen_data.py \
    --path "<path-to-testcases>" \
    --restriction_label <True/False> \
    --max_iterations <number> \
    --pass_k <number> \
    --total_samples <number> \
    --model "<model-name>" \
    --api_key "<api-key>"
```

You can also use the tool programmatically:

```bash
from gen_data import PICBench

PICBench(
    path="../testcases",
    restriction_label=True,
    max_iterations=3,
    pass_k=1,
    total_samples=5,
    model="gpt-4",
    api_key="your-api-key"
)
```

### **Parameters**
| Parameter             | Type    | Default                       | Description                                                                |
|-----------------------|---------|-------------------------------|----------------------------------------------------------------------------|
| `--path`              | string  | '../testcases'                | Path to the directory containing test cases and solutions.                 |
| `--restriction_label` | boolean | 'True'                        | Whether to apply restrictions to the generation process.                   |
| `--max_iterations`    | int     | '3'                           | Maximum iterations for error feedback.                                     |
| `--pass_k`            | int     | '1'                           | k for Pass@k.                                                              |
| `--total_samples`     | int     | '5'                           | Total number of samples to generate.                                       |
| `--model`             | string  | '"gpt-4"'                     | Name of the language model to use for reasoning steps.                     |
| `--api_key`           | string  | Required                      | API key to authenticate with the language model service.                   |

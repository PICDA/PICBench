import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Optional


class LLMAgent:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize the LLM Agent with API key and model type.
        Supports GPT (OpenAI), Claude (Anthropic), and Gemini (Google).
        """
        self.api_key = api_key
        self.model = model
        self.model_class = self._determine_model_class(model)
        self.client = self._initialize_client()
        self.current_conversation_history = []

    def _determine_model_class(self, model: str) -> str:
        """
        Determine the model class (gpt, claude, gemini) from the model string.
        """
        if model == "o1-mini":
            return "gpt"
        return model.split("-")[0]

    def _initialize_client(self):
        """
        Initialize the appropriate client based on the model class.
        """
        if self.model_class == "gpt":
            return OpenAI(api_key=self.api_key)
        elif self.model_class == "claude":
            return Anthropic(api_key=self.api_key)
        elif self.model_class == "gemini":
            genai.configure(api_key=self.api_key, transport="rest")
            return genai.GenerativeModel(model_name=self.model)
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")

    def _call_api(self, messages: List[dict]):
        if self.model_class == 'gpt':
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error calling API: {e}"
        elif self.model_class == 'gemini':
            try:
                response = self.client.generate_content(contents=messages)
                return response.text
            except Exception as e:
                return f"Error calling API: {e}"
        elif self.model_class == 'claude':
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=messages
                )
            except Exception as e:
                return f"Error calling API: {e}"
            return response.content

    def _format_message(self, system_prompt, example_a, example_q, user_question):
        """
        Initialize the appropriate messages format based on the model class.
        """
        if self.model_class == 'gpt':
            if self.model == 'o1-mini':
                messages = [
                    {"role": "user", "content": system_prompt},
                    {"role": "user", "content": example_q},
                    {"role": "assistant", "content": example_a},
                    {"role": "user", "content": user_question},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example_q},
                    {"role": "assistant", "content": example_a},
                    {"role": "user", "content": user_question},
                ]
        elif self.model_class == 'gemini':
            messages = [
                {"role": "user", "parts": system_prompt},
                {"role": "user", "parts": example_q},
                {"role": "model", "parts": example_a},
                {"role": "user", "parts": user_question},
            ]
        elif self.model_class == 'claude':
            messages = [
                {"role": "user", "content": system_prompt},
                {"role": "user", "content": example_q},
                {"role": "assistant", "content": example_a},
                {"role": "user", "content": user_question},
            ]
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")
        return messages

    def ASK_LLM(self, system_prompt, user_question: str):
        """
        A single standalone call without any history tracking.
        """
        example_q = "Create a Mach-Zehnder interferometer (MZI) with a single input and output, featuring a path length difference of L. Use the built-in multimode interferometer (MMI) component. \nParameters:\nL = 10 microns \n\nDesign the circuit and generate the corresponding JSON netlist based on the description. Try to understand the requirements and give reasoning steps in natural language to achieve it."
        example_a = "<analysis>\nTo create a Mach-Zehnder Interferometer (MZI) with one input and one output using the built-in MMI component, we need the following steps:\n\n1. **Components Needed**:\n   - Two MMIs (acting as splitters and combiners).\n   - Two waveguides (one with default length, one with an additional 10 microns to create the length difference).\n\n2. **Instances**:\n   - `mmi1`: First MMI splitter.\n   - `waveguide_top`: Waveguide with length = default length + 10 microns.\n   - `waveguide_bottom`: Waveguide with default length.\n   - `mmi2`: Second MMI combiner.\n\n3. **Connections**:\n   - Connect output ports of `mmi1` to inputs of the two waveguides.\n   - Connect outputs of the waveguides to inputs of `mmi2`.\n\n4. **Ports**:\n   - Define the input port connected to `mmi1`.\n   - Define the output port from `mmi2`.\n\n5. **Length Difference**:\n   - Set `waveguide_top` length to default length + 10 microns.\n\n<result>\n{\n  \"netlist\":{\n    \"instances\": {\n      \"mmi1\": \"mmi\",\n      \"waveguide_top\": {\"component\": \"waveguide\", \"settings\": {\"length\": 20}},\n      \"waveguide_bottom\": \"waveguide\",\n      \"mmi2\": \"mmi\"\n    },\n    \"connections\": {\n      \"mmi1,O1\": \"waveguide_bottom,I1\",\n      \"waveguide_bottom,O1\": \"mmi2,I1\",\n      \"mmi1,O2\": \"waveguide_top,I1\",\n      \"waveguide_top,O1\": \"mmi2,I1\"\n    },\n    \"ports\": {\n      \"I1\": \"mmi1,I1\",\n      \"O1\": \"mmi2,O1\"\n    }\n  },\n  \"models\":{\n    \"mmi\": \"mmi1x2\",\n    \"waveguide\": \"straight\"\n  }\n}"

        messages = self._format_message(system_prompt, example_a, example_q, user_question)
        return self._call_api(messages)

    def ASK_LLM_iterate(self, system_prompt, user_question: str, clear_context: bool = False):
        """
        Handles iterative calls with optional context clearing.
        If `clear_context` is True, it starts a new conversation.
        """
        example_q = "Create a Mach-Zehnder interferometer (MZI) with a single input and output, featuring a path length difference of L. Use the built-in multimode interferometer (MMI) component. \nParameters:\nL = 10 microns \n\nDesign the circuit and generate the corresponding JSON netlist based on the description. Try to understand the requirements and give reasoning steps in natural language to achieve it."
        example_a = "<analysis>\nTo create a Mach-Zehnder Interferometer (MZI) with one input and one output using the built-in MMI component, we need the following steps:\n\n1. **Components Needed**:\n   - Two MMIs (acting as splitters and combiners).\n   - Two waveguides (one with default length, one with an additional 10 microns to create the length difference).\n\n2. **Instances**:\n   - `mmi1`: First MMI splitter.\n   - `waveguide_top`: Waveguide with length = default length + 10 microns.\n   - `waveguide_bottom`: Waveguide with default length.\n   - `mmi2`: Second MMI combiner.\n\n3. **Connections**:\n   - Connect output ports of `mmi1` to inputs of the two waveguides.\n   - Connect outputs of the waveguides to inputs of `mmi2`.\n\n4. **Ports**:\n   - Define the input port connected to `mmi1`.\n   - Define the output port from `mmi2`.\n\n5. **Length Difference**:\n   - Set `waveguide_top` length to default length + 10 microns.\n\n<result>\n{\n  \"netlist\":{\n    \"instances\": {\n      \"mmi1\": \"mmi\",\n      \"waveguide_top\": {\"component\": \"waveguide\", \"settings\": {\"length\": 20}},\n      \"waveguide_bottom\": \"waveguide\",\n      \"mmi2\": \"mmi\"\n    },\n    \"connections\": {\n      \"mmi1,O1\": \"waveguide_bottom,I1\",\n      \"waveguide_bottom,O1\": \"mmi2,I1\",\n      \"mmi1,O2\": \"waveguide_top,I1\",\n      \"waveguide_top,O1\": \"mmi2,I1\"\n    },\n    \"ports\": {\n      \"I1\": \"mmi1,I1\",\n      \"O1\": \"mmi2,O1\"\n    }\n  },\n  \"models\":{\n    \"mmi\": \"mmi1x2\",\n    \"waveguide\": \"straight\"\n  }\n}"

        if clear_context or not self.current_conversation_history:
            # Start a new conversation
            self.current_conversation_history = self._format_message(system_prompt, example_a, example_q, user_question)

        if self.model_class == 'gpt':
            # Add user question to the conversation history
            self.current_conversation_history.append({"role": "user", "content": user_question})

            # Call the API with the current conversation history
            response = self._call_api(self.current_conversation_history)

            # Append assistant's response to the conversation history
            self.current_conversation_history.append({"role": "assistant", "content": response})

        elif self.model_class == 'gemini':
            self.current_conversation_history.append({"role": "user", "parts": user_question})
            response = self._call_api(self.current_conversation_history)
            self.current_conversation_history.append({"role": "model", "parts": response})

        elif self.model_class == 'claude':
            self.current_conversation_history.append({"role": "user", "content": user_question})
            response = self._call_api(self.current_conversation_history)
            self.current_conversation_history.append({"role": "assistant", "content": response})
        else:
            raise ValueError(f"Unsupported model class: {self.model_class}")
        return response

    def start_new_conversation(self):
        """
        Resets the conversation history for a new problem context.
        """
        self.current_conversation_history = []

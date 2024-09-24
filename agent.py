from transformers import AutoTokenizer, Phi3ForCausalLM
import torch
import re
import ast
from termcolor import colored

import os
import transformers
import logging

import warnings
from tools.Tools import *

# Ensure that accelerate is imported to enable its features
from accelerate import Accelerator

# Ignore all warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
transformers.logging.set_verbosity_error()
# Suppress specific warnings
warnings.filterwarnings("ignore", message="Special tokens have been added")
warnings.filterwarnings("ignore", message="A new version of the following files was downloaded")
import gc

class Agent:
    def __init__(self, system_prompt, model_name, tools, stop=None):
        """
        Initializes the agent with a model.

        Parameters:
        system_prompt (str): The system's prompt template.
        model_name (str): The name of the model to use.
        stop (list): Optional stop tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, verbose=False)
        self.model = Phi3ForCausalLM.from_pretrained(model_name)
        self.system_prompt_template = '<|system|>{}<|end|>'.format(system_prompt)
        self.tools = tools
        self.history = []

    def process_prompt(self, prompt):
        """
        Processes the user prompt and generates a response.

        Parameters:
        prompt (str): The user input.

        Returns:
        tuple: (response_text, image_path or None)
        """
        prompt_template = '<|user|>{}<|end|>'.format(prompt)
        self.history.append(self.system_prompt_template)
        self.history.append(prompt_template)
        input_history = self.history + ['<|assistant|>']

        input_string = ''.join(input_history)

        inputs = self.tokenizer.encode(input_string, return_tensors='pt')
        outputs = self.model.generate(
            inputs,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            max_new_tokens=100
        )
        # Output processing
        output_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = output_decoded.split('<|assistant|>')[-1]

        self.history = input_history
        response_out = response.split('<|end|>')[0]

        # Execute tools if any
        image_path = None
        for tool in self.tools.keys():
            if tool in response_out:
                try:
                    # Prepare the response for evaluation
                    output_corrected = re.sub(r'(\w+)', r'"\1"', response_out)
                    output_dict = ast.literal_eval(output_corrected)
                    tool_name = output_dict['tool']
                    arguments = output_dict['arguments']
                    cleaned_arguments = [arg.strip('"') for arg in arguments]

                    # Assert that the tool name and arguments are present
                    assert tool_name in self.tools, f"Tool {tool_name} is not recognized."
                    assert len(cleaned_arguments) == 3, "Incorrect number of arguments provided."

                    # Call the function by looking it up in the dictionary
                    response_tool = self.tools[tool_name](*cleaned_arguments)
                    response_out = response_tool[0]
                    parent_path = os.getcwd()
                    image_path = os.path.join(parent_path, 'saves', 'plot.png')
                    describer = ImageDescriber()
                    image_description = describer.caption_image(image_path)
                    response_out = f"{response_out} {image_description}"
                except Exception as e:
                    print(f"Error: {e}")
                    response_out = 'There was an error in running the tool.'

        self.history[-1] = self.history[-1] + str(response_out) + '<|end|>'
        formatted_response = ".\n".join(response_out.split(". "))

        return formatted_response, image_path

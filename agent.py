from transformers import AutoTokenizer,Phi3ForCausalLM
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




class Agent:
    def __init__(self, system_prompt, model_name,tools, stop=None):
        """
        Initializes the agent with a model.

        Parameters:
        system_prompt (str): The system's prompt template.
        model_name (str): The name of the model to use.
        stop (list): Optional stop tokens.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,verbose=False)
        self.model = Phi3ForCausalLM.from_pretrained(model_name)
        self.system_prompt_template = '<|system|>{}<|end|>'.format(system_prompt)
        self.tools = tools
        

    def chat(self):
        history = []
        while (prompt := input("Enter a prompt (q to quit): ")) != "q":
            prompt_template = '''<|user|>{}<|end|>'''
            prompt_processed = prompt_template.format(prompt)
            history.append(self.system_prompt_template)
            
            history.append(prompt_processed)
            input_history = history + ['<|assistant|>']

            input_string = ''
            for string in input_history:
                input_string = input_string + string

            inputs = self.tokenizer.encode(input_string, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                max_new_tokens=100
            )
            # output processing
            output_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            response = output_decoded.split('<|assistant|>')[-1]
            
            print(colored(prompt, 'cyan'))
            history = input_history
            response_out = response.split('<|end|>')[0]
            print(colored(response_out,'green'))
            
            
            # block for execute function
           
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
                        response_tool = self.tools[tool_name](cleaned_arguments[0], cleaned_arguments[1], cleaned_arguments[2])
                        response_out = response_tool[0]
                        parent_path = os.getcwd()
                        image_path = os.path.join(parent_path,'saves','plot.png')
                        describer = ImageDescriber()
                        image_description = describer.caption_image(image_path)
                        response_out = response_out + ' ' + image_description
                        #print(image_path)
                    except Exception as e:
                        print(f"Error: {e}")
                        response_out = 'there was an error in running the tool'

            input_history[-1] = input_history[-1] + str(response_out) + '<|end|>'
            formatted_response = ".\n".join(response_out.split(". "))

            # Print the formatted response in red
            print(colored(formatted_response, 'red'))
            #print(colored(response_out, 'red'))
        return history
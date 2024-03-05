from typing import List, Dict, Callable
import openai

import re


def find_first_integer_digit(string):
    match = re.search(r'\d+', string)
    if match:
        integer = int(match.group())
        first_digit = int(str(integer)[0])
        return first_digit
    else:
        return None


class PromptBuilder:
    """
    Utility class for building prompts for prompt checking.
    """
    def __init__(self):
        self.prompt = ""

    def add_task(self, task_description):
        self.prompt += f"### Task\n{task_description}\n\n"
        return self

    def add_header(self, header_text):
        self.prompt += f"### {header_text}\n"
        return self

    def add_content(self, content):
        self.prompt += f"{content}\n\n"
        return self

    def add_content_to_analyse(self, response):
        self.prompt += f"### Content to analyze\n```\n{response}\n```\n\n"
        return self

    def add_what_to_return(self, what_to_return):
        self.prompt += f"### What to return\n{what_to_return}\n\n"
        return self

    def add_output_example(self, output_example):
        self.prompt += f"### Example of output to follow\n{output_example}"
        return self
        
    def build(self):
        return self.prompt


def call_gpt(model_name: str, prompt: str) -> str:
    """
    Generate a response using an OpenAI GPT model.

    Args:
        model_name (str): The name of the GPT model to use.
        prompt (str): The prompt for generating the response.

    Returns:
        str: The generated response.
    """
    # Define the prompt (value of the key "content")
    message_text = [{"role":"system","content":prompt}]

    # Call the OpenAI API to generate an answer to the prompt in message_text
    completion = openai.ChatCompletion.create(
    engine=model_name,
    messages = message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )

    return completion.choices[0]["message"]["content"]


def prompt_checking(num_tests: int, func: Callable, prompt: str, feature_prompt: dict, PromptBuilder: PromptBuilder, **kwargs) -> list:
    """
    Perform prompt checking for a specified number of tests.

    Args:
        num_tests (int): The number of tests to perform.
        func (function): The function taking the user query as input and returning the final response.
        prompt (str): The prompt to use for the initial GPT call.
        feature_prompt (dict): A dictionary containing the feature prompt details.
            It should include the following keys:
                - "task": The task description.
                - "header": The header text.
                - "content": The content to be added.
                - "return": The description of what to return.
                - "examples": The example of output to follow.
        PromptBuilder (class): The PromptBuilder class to use for constructing prompts.

    Returns:
        list: A list of results obtained from the prompt checking, converted to integers.

    """
    list_results = []
    for i in range(num_tests):
        response = func(prompt=prompt, **kwargs)

        builder = PromptBuilder()

        # Construct the checker prompt
        checker_prompt = builder.add_task(feature_prompt["task"]) \
            .add_header(feature_prompt["header"]) \
            .add_content(feature_prompt["content"]) \
            .add_content_to_analyse(response) \
            .add_what_to_return(feature_prompt["return"]) \
            .add_output_example(feature_prompt["examples"]) \
            .build()

        # Perform GPT call with the checker prompt
        result = func(prompt=checker_prompt, **kwargs)

        # Convert the result to an integer and append to the list
        list_results.append(int(find_first_integer_digit(result)))

    return list_results
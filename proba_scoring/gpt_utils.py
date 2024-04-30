from typing import Callable
import openai
from proba_scoring.prompt_builder import PromptBuilder
from proba_scoring.schema import CheckerPrompt

import re
import json


def find_first_integer_digit(string: str) -> int:
    """
    Finds the first integer digit in a string. Used for double-checking the output of the checker-LLM.

    Args:
        string (str): The input string.

    Returns:
        int: The first integer digit found in the string. Returns None if no integer digit is found.

    Example:
        >>> find_first_integer_digit("abc123def456")
        1
    """
    match = re.search(r'\d+', string)

    if match:
        integer = int(match.group())
        first_digit = int(str(integer)[0])
        return first_digit
    else:
        return None


def find_first_float_digit(string: str) -> int:
    """
    Finds the first integer or decimal digit in a string. Used for double-checking the output of the checker-LLM.

    Args:
        string (str): The input string.

    Returns:
        float: The first integer or decimal digit found in the string. Returns None if no integer or decimal digit is found.

    Example:
        >>> find_first_integer_digit("abc123def456")
        1
        >>> find_first_integer_digit("abc0.5def789")
        0.5
    """
    match = re.search(r'(\d+(\.\d+)?)', string)

    if match:
        return float(match.group())
    else:
        return None


def call_gpt(model_name: str, prompt: str) -> str:
    """
    Generate a response using an OpenAI GPT model.

    Args:
        model_name (str): The name of the GPT model to use.
        prompt (str): The prompt for generating the response.

    Returns:
        str: The generated response.

    Example:
        >>> call_gpt(model_name = gpt_model, prompt = user_question)
        Response from the OpenAI GPT to the user question.
    """
    # Define the prompt (value of the key "content")
    message_text = [{"role":"system","content":prompt}]

    # Call the OpenAI API to generate an answer to the prompt in message_text
    # pip library version is openai==0.28.0. Later versions might require a change here
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


def llm_checking(
        num_tests: int,
        func_call: Callable, 
        func_checker: Callable, 
        prompt: str,
        system_prompt: str = None,
        checker_prompt: str = None,
        **kwargs
        ) -> list:
    """
    Perform prompt checking for a specified number of tests.

    Args:
        num_tests (int): The number of tests to perform.
        func_call (function): The function taking the user query as input (can be a simple GPT API call or a whole pipeline).
        func_checker (function): The function taking the first user response as input and returning the final response (can be a simple GPT API call or a whole pipeline).
        prompt (str): The prompt to use for the initial LLM call.
        system_prompt (str): Either the question to compare, the expected answer, or anything that the checker prompt will compare to the intiial prompt 
        checker_prompt (str): The checker prompt
        **kwargs: Additional keyword arguments for the model API. (e.g., `max_tokens`, `temperature`)

    Returns:
        list: A list of results obtained from the prompt checking, converted to integers.

    Example:
        >>> kwargs = {"model_name": gpt_model}
        >>> llm_checking(num_tests = 3, func_call = call_gpt, func_checker = call_gpt, prompt = user_question, feature_prompt = feature_prompt_dict, **kwargs)
    """
    list_results = []
    for i in range(num_tests):
        response = func_call(prompt=prompt, **kwargs)

        # Perform GPT call with the checker prompt
        if system_prompt is not None:
            result = func_checker(prompt=checker_prompt.format(system_prompt=system_prompt, statement_to_rate=response), **kwargs)
        else:
            result = func_checker(prompt=checker_prompt.format(statement_to_rate=response), **kwargs)

        print(response,'score: ',int(json.loads(result)['rating']))

        # Convert the result to an integer and append to the list
        list_results.append(int(json.loads(result)['rating']))

    return list_results
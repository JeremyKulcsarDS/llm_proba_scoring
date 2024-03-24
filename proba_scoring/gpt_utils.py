from typing import Callable
import openai
from proba_scoring.prompt_builder import PromptBuilder
from proba_scoring.schema import CheckerPrompt

import re

PROMPT_BUILDER_PREFIX_SUFFIX = {
        "task": ["### Task\n","\n\n"],
        "header": ["### ","\n"],
        "content": ["","\n\n"],
        "initial_prompt": ["### Initial prompt\n```\n","\n```\n\n"],
        "response": ["### Response\n```\n","\n```\n\n"],
        "content_to_analyse": ["### Content to analyse\n```\n","\n```\n\n"],
        "to_return": ["### What to return\n","\n\n"],
        "example": ["### Example of output to follow\n",""]
    }


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


def build_checker_prompt(
        builder: PromptBuilder, 
        prompt: str, 
        response:str, 
        checker_prompt_features: dict,
        need_initial_prompt:bool,
        need_expected_answer:bool, # to add
        prefix_suffix_dict: dict = PROMPT_BUILDER_PREFIX_SUFFIX
        ):

    if need_initial_prompt:
        for key in checker_prompt_features.keys():
            if key != "to_return":
                builder.add_line(
                    prefix_suffix_dict[key][0],
                    checker_prompt_features[key],
                    prefix_suffix_dict[key][1]
                )
            else:
                builder.add_line(
                    prefix_suffix_dict["initial_prompt"][0],
                    prompt,
                    prefix_suffix_dict["initial_prompt"][1]
                )
                builder.add_line(
                    prefix_suffix_dict["response"][0],
                    response,
                    prefix_suffix_dict["response"][1]
                )
                builder.add_line(
                    prefix_suffix_dict[key][0],
                    checker_prompt_features[key],
                    prefix_suffix_dict[key][1]
                )
    else:
        for key in checker_prompt_features.keys():
            if key != "to_return":
                builder.add_line(
                    prefix_suffix_dict[key][0],
                    checker_prompt_features[key],
                    prefix_suffix_dict[key][1]
                )
            else:
                builder.add_line(
                    prefix_suffix_dict["content_to_analyse"][0],
                    response,
                    prefix_suffix_dict["content_to_analyse"][1]
                )
                builder.add_line(
                    prefix_suffix_dict[key][0],
                    checker_prompt_features[key],
                    prefix_suffix_dict[key][1]
                )
    
    return builder.build()


def llm_checking(
        num_tests: int,
        func_call: Callable, 
        func_checker: Callable, 
        prompt: str,
        need_initial_prompt: bool,
        checker_prompt_features: CheckerPrompt = None,
        **kwargs
        ) -> list:
    """
    Perform prompt checking for a specified number of tests.

    Args:
        num_tests (int): The number of tests to perform.
        func_call (function): The function taking the user query as input (can be a simple GPT API call or a whole pipeline).
        func_checker (function): The function taking the first user response as input and returning the final response (can be a simple GPT API call or a whole pipeline).
        prompt (str): The prompt to use for the initial LLM call.
        feature_prompt (dict): A dictionary containing the feature prompt details for the checker LLM.
            It should include the following keys:
                - "task": The task description.
                - "header": The header text.
                - "content": The content to be added.
                - "to_return": The description of what to return (should always return 0 or 1).
                - "examples": The example of output to follow.
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

        # Construct the checker prompt
        builder = PromptBuilder()
        checker_prompt = build_checker_prompt(
            builder = builder,
            prompt = prompt,
            response = response,
            need_initial_prompt = need_initial_prompt,
            checker_prompt_features = checker_prompt_features)

        # Perform GPT call with the checker prompt
        result = func_checker(prompt=checker_prompt, **kwargs)

        print(response,'score: ',int(find_first_integer_digit(result)))

        # Convert the result to an integer and append to the list
        list_results.append(int(find_first_integer_digit(result)))

        # If it doesn't return 0 or 1, it will throw an error. Need to take care of this case.

    return list_results


def llm_checking_expectancy_tmp(
        num_tests: int,
        func_call: Callable, 
        func_checker: Callable, 
        prompt: str,
        need_initial_prompt: bool,
        expected_answer: str,
        checker_prompt_features: CheckerPrompt = None,
        **kwargs
        ) -> list:
    """
    WORK IN PROGRESS
    """
    list_results = []
    for i in range(num_tests):
        response = func_call(prompt=prompt, **kwargs)

        # Construct the checker prompt
        checker_prompt = f"""
        ### Task
        Your job is to measure whether the two following statements included in the ``` contain the same information with accuracy
        
        ### Expected answer
        ```
        {expected_answer}
        ```
        
        ### Generated answer
        ```
        {response}
        ```
        
        ### What to return
        Return an integer respecting the following rule:
        '1' if they contain the same information,
        '0.5' if they partially share information.
        '0' if they share no information or the given information is different (for instance different numbers).
        
        Do not include anything else other than the integer in the response.
        
        ### Example of output to follow
        1
        """
        
        print(expected_answer)
        print(response)

        # Perform GPT call with the checker prompt
        result = func_checker(prompt=checker_prompt, **kwargs)

        print(response,'score: ',float(find_first_float_digit(result)))

        # Convert the result to an integer and append to the list
        list_results.append(float(find_first_float_digit(result)))

        # If it doesn't return 0 or 1, it will throw an error. Need to take care of this case.

    return list_results
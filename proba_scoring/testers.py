from typing import List, Dict, Callable, Tuple

import copy

import gzip
import numpy as np

from proba_scoring.scoring import compute_proportion_positivies, compute_conditional_proba_score, compute_unconditional_proba_score, append_proba_scores
from proba_scoring.gpt_utils import llm_checking


class CorrectnessTester():
    """
    Correctness tester.

    Sends an additional requests to a LLM, asking the model to judge whether 
    the response answers the question properly.
    
    This score can help gauge if the LLM is able to understand its questions, 
    which can be a point of interest when the user asks very complex or 
    domain-specific questions.
    
    The Correctness score is the fraction of 
    "yes" responses, divided by the total number of responses.
    """
    def __init__(
              self
              ) -> None:
        self.__checker_prompt = {
             "task":"Your job is to measure whether the following statement included in the ``` replies the initial question with accuracy.",
             "to_return":"Return an integer respecting the following rule:\n'1' if the answer replies the initial question,\n'0' if the answer does not reply the initial question.\nDo not include anything else other than the integer in the response.\nIf the statement announces that it cannot fulffill the request for whatever reason or anything else, return '0' anyway.",
             "example":"1"
             }
        
        with open("artifacts/prompt_correctness_checker.txt", "r") as file:
            self.__checker_prompt = file.read()


    def prompt_checking(
            self, 
            num_tests: int,
            func_call: Callable, 
            func_checker: Callable, 
            prompt: str,
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
            checker_prompt (str): The checker prompt
            **kwargs: Additional keyword arguments for the model API. (e.g., `max_tokens`, `temperature`)

        Returns:
            list: A list of results obtained from the prompt checking, converted to integers.

        Example:
            >>> kwargs = {"model_name": gpt_model}
            >>> prompt_checking(num_tests = 3, func_call = call_gpt, func_checker = call_gpt, prompt = user_question, feature_prompt = feature_prompt_dict, **kwargs)
        """
        if checker_prompt is None:
                checker_prompt = self.__checker_prompt

        list_results = llm_checking(
            num_tests = num_tests,
            func_call = func_call,
            func_checker = func_checker,
            prompt = prompt,
            system_prompt = prompt,
            checker_prompt = checker_prompt,
            **kwargs
            )

        return list_results
    

    def score(
              self,
              checker_llm_results: List[int]
              ) -> float:
        """
        Calculates the score based on the provided list of checker LLM results.
        It will sum all the 1s and divide by the number of responses.

        Args:
            checker_llm_results (List[int]): The list of checker LLM results.

        Returns:
            float: The computed score.
        """
        return compute_proportion_positivies(checker_llm_results)
    

    def append_scores(
              self,
              queries_dict: Dict[str, dict],
              num_tests: int,
              func_call: Callable, 
              func_checker: Callable, 
              checker_prompt: str = None,
              **kwargs
            ) -> Dict[str, dict]:
        
        queries_dict_copy = copy.deepcopy(queries_dict)
        
        for query in queries_dict_copy.values():
            checker_llm_results = self.prompt_checking(
                 num_tests = num_tests, 
                 func_call = func_call, 
                 func_checker = func_checker, 
                 prompt = query["question"], 
                 checker_prompt = checker_prompt, 
                 **kwargs
                 )
            
            query["correctness_score"] = self.score(checker_llm_results)
            del checker_llm_results

        return queries_dict_copy
    

class ExpectancyTester():
    """
    Expectancy tester.

    Sends an additional requests to a LLM, asking the model to judge whether 
    the response was the expected one. 
    
    This score helps check if the model answers what is expected. It requires
    the construction of expected answers from a human, and can serve for looking
    if the model needs more prompt engineering or fine-tuning.
    
    The Expectancy score is the fraction of 
    "yes" responses, divided by the total number of responses.
    """
    def __init__(
              self
              ) -> None:
        with open("artifacts/prompt_expectancy_checker.txt", "r") as file:
            self.__checker_prompt = file.read()


    def prompt_checking(
            self, 
            num_tests: int,
            func_call: Callable, 
            func_checker: Callable, 
            prompt: str,
            expected_answer: str,
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
            checker_prompt (str): The checker prompt
            **kwargs: Additional keyword arguments for the model API. (e.g., `max_tokens`, `temperature`)

        Returns:
            list: A list of results obtained from the prompt checking, converted to integers.

        Example:
            >>> kwargs = {"model_name": gpt_model}
            >>> prompt_checking(num_tests = 3, func_call = call_gpt, func_checker = call_gpt, prompt = user_question, feature_prompt = feature_prompt_dict, **kwargs)
        """
        if checker_prompt is None:
                checker_prompt = self.__checker_prompt

        list_results = llm_checking(
            num_tests = num_tests,
            func_call = func_call,
            func_checker = func_checker,
            prompt = prompt,
            system_prompt = expected_answer,
            checker_prompt = checker_prompt,
            **kwargs
            )

        return list_results
    

    def score(
              self,
              checker_llm_results: List[int]
              ) -> float:
        """
        Calculates the score based on the provided list of checker LLM results.
        It will sum all the 1s and divide by the number of responses.

        Args:
            checker_llm_results (List[int]): The list of checker LLM results.

        Returns:
            float: The computed score.
        """
        return compute_proportion_positivies(checker_llm_results)
    

    def append_scores(
              self,
              queries_dict: Dict[str, dict],
              num_tests: int,
              func_call: Callable, 
              func_checker: Callable, 
              checker_prompt: str = None,
              **kwargs
            ) -> Dict[str, dict]:
        
        queries_dict_copy = copy.deepcopy(queries_dict)
        
        for query in queries_dict_copy.values():
            checker_llm_results = self.prompt_checking(
                 num_tests = num_tests, 
                 func_call = func_call, 
                 func_checker = func_checker, 
                 prompt = query["question"], 
                 expected_answer = query["expected_answer"],
                 checker_prompt = checker_prompt, 
                 **kwargs
                 )
            
            query["expectancy_score"] = self.score(checker_llm_results)
            del checker_llm_results

        return queries_dict_copy
    

class ConsistencyTester():
    """
    Consistency tester.

    Sends an additional requests to a LLM, asking the model to judge whether 
    the response answers the question consistently.
    
    This score can help gauge if the LLM is able to understand the question and 
    provide a consistent answer.

    Note: This test isn't that useful for deterministic-aimed configurations
    such as temperature = 0 + top_k = 1.
    """
    def __init__(
              self
              ) -> None:
        pass


    def compute_ncd(
            self, 
            str1: str, 
            str2: str
            ) -> float:
        """
        Computes the Normalized Compression Distance (NCD) between two strings.

        Args:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            float: The NCD between the two strings.
        """
        c_str1 = len(gzip.compress(str1.encode()))
        c_str2 = len(gzip.compress(str2.encode()))
        c_str1_str2 = len(gzip.compress(" ".join([str1, str2]).encode()))
        return (c_str1_str2 - min(c_str1, c_str2)) / max(c_str1, c_str2)
    

    def build_baseline(
            self, 
            list_replies: List[str]
            ) -> float:
        """
        Builds the baseline value by computing the NCD for each reply against itself and returning the minimum value.

        Args:
            list_replies (List[str]): A list of replies.

        Returns:
            float: The baseline value.
        """
        baseline_elements = []
        for reply in list_replies:
            baseline_elements.append(self.compute_ncd(reply, reply))
        return min(baseline_elements)
    

    def score(
            self, 
            list_replies: List[str]
            ) -> float:
        """
        Computes the score based on the NCD values between pairs of replies, normalized by the baseline value.

        Args:
            list_replies (List[str]): A list of replies.

        Returns:
            float: The score.
        """
        baseline = self.build_baseline(list_replies)

        ncd_list = []
        for i in range(len(list_replies)):
            for j in range(i + 1, len(list_replies)):
                ncd_list.append(self.compute_ncd(list_replies[i], list_replies[j]) - baseline)

        return 1 - np.mean(ncd_list)


    def extract_info(
            self,
            llm_reply: str
            ) -> List[str]:
        prompt = \
        """
        #### TASK
        Your task is to extract the information from the statement in the ```
        Reorganise the information as small factual sentences.
        Order them alphabteically.

        """
        
        return


class ToxicityTester():
    """
    Toxicity tester.

    Send an additional request to an LLM, asking the model to judge whether the response was toxic. 
    The request is done multiple times in order to get a score by calculating the fraction of "yes" responses. 
    A general score and a question-specific score are deduced. Difference with human expected judgement of 
    toxicity can be made to add the human bias perspective into the score.
    """
    def __init__(
              self
              ) -> None:
        with open("artifacts/prompt_toxicity_checker.txt", "r") as file:
            self.__checker_prompt = file.read()


    def prompt_checking(
            self, 
            num_tests: int,
            func_call: Callable, 
            func_checker: Callable, 
            prompt: str,
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
            checker_prompt (str): The checker prompt
            **kwargs: Additional keyword arguments for the model API. (e.g., `max_tokens`, `temperature`)

        Returns:
            list: A list of results obtained from the prompt checking, converted to integers.

        Example:
            >>> kwargs = {"model_name": gpt_model}
            >>> prompt_checking(num_tests = 3, func_call = call_gpt, func_checker = call_gpt, prompt = user_question, checker_prompt = checker_prompt, **kwargs)
        """
        if checker_prompt is None:
                checker_prompt = self.__checker_prompt

        list_results = llm_checking(
             num_tests = num_tests,
            func_call = func_call,
            func_checker = func_checker,
            prompt = prompt,
            checker_prompt = checker_prompt,
            **kwargs
            )

        return list_results
    

    def score(
              self,
              checker_llm_results: List[int]
              ) -> float:
        """
        Calculates the score based on the provided list of checker LLM results.
        It will sum all the 1s and divide by the number of responses.

        Args:
            checker_llm_results (List[int]): The list of checker LLM results.

        Returns:
            float: The computed score.
        """
        return compute_proportion_positivies(checker_llm_results)

    
    def conditional_proba_score(
              self,
              checker_llm_results: List[int],
              human_eval: float
              ) -> Tuple[float, float]:
        # Returns the conditional proba score and the similarity score to the human evaluation
         return compute_conditional_proba_score(checker_llm_results, human_eval)
    
    def unconditional_proba_score(
              self,
              list_results_all_questions: List[List[float]]
              ) -> List[float]:
         return compute_unconditional_proba_score(list_results_all_questions)
    
    def append_scores(
              self,
              queries_dict: Dict[str, dict],
              num_tests: int,
              func_call: Callable, 
              func_checker: Callable, 
              checker_prompt: str = None,
              **kwargs
            ) -> Dict[str, dict]:
        
        queries_dict_copy = copy.deepcopy(queries_dict)
        
        for query in queries_dict_copy.values():
            checker_llm_results = self.prompt_checking(
                 num_tests = num_tests, 
                 func_call = func_call, 
                 func_checker = func_checker, 
                 prompt = query["question"], 
                 checker_prompt = checker_prompt, 
                 **kwargs
                 )
            
            query["toxocity_checks"] = checker_llm_results
            del checker_llm_results

        return append_proba_scores(queries_dict_copy, returns_field_name = "toxocity_checks")
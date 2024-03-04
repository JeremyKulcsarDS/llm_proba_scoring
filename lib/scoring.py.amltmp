from typing import List, Dict, Tuple


def compute_unconditional_proba_score(list_results_all_questions: List[List[float]]) -> List[float]:
    """
    Computes the unconditional probability score for each question based on the list of results.

    Args:
        list_results_all_questions (List[List[float]]): A list of lists containing the results for each question.

    Returns:
        List[float]: A list of probability scores for each question.
    """

    # Set n and m
    num_questions = len(list_results_all_questions)
    num_results = len(list_results_all_questions[0])

    # Compute the final probability score for the question
    proba_e = sum(element for sublist in list_results_all_questions for element in sublist) / (num_questions*num_results)
    
    return proba_e


def compute_conditional_proba_score(list_results_one_question: List[float], human_eval: float) -> Tuple[float, float]:
    """
    Computes the conditional probability score and similarity score for a given question.

    Args:
        list_results_one_question (List[float]): List of probability scores for a question.
            Each score represents the probability of a certain event E given the question Q.
        human_eval (float): The human evaluation score for the question.

    Returns:
        float: The computed conditional probability score.
            The score represents the conditional probability of event E given the question Q.
            It is calculated as the average probability score in the list_results_one_question.

    """

    # Set m
    num_results = len(list_results_one_question)

    # Compute the conditional probability score for the question
    proba_e_knowing_q = sum(element for element in list_results_one_question) / (num_results)
    
    # Returns the conditional proba score and the similarity score to the human evaluation
    return proba_e_knowing_q, 1 - abs(human_eval - proba_e_knowing_q)


def append_proba_scores(checking_elements: Dict[str, dict]) -> Dict[str, dict]:
    """
    Computes the conditional and unconditional probability scores for a set of checking elements and appends them to each query.

    Args:
        checking_elements (Dict[str, dict]): Dictionary containing checking elements.
            Each key represents a query with the following structure:
                {
                    "question": str,  # The question associated with the query
                    "returns": List[float],  # List of probability scores for the query
                    "human_eval": int  # The human evaluation score for the query
                }

    Returns:
        Dict[str, dict]: The updated checking_elements dictionary with appended scores.
    """

    # Extract the lists of results from the dictionary of responses
    list_results_all_questions = [query["returns"] for query in checking_elements.values()]
    n = len(list_results_all_questions)

    # Compute the unconditional score
    S_u = compute_unconditional_proba_score(list_results_all_questions)

    for query in checking_elements.values():
        # Compute the conditional score for each question and add it to the dictionary
        query["conditional_score"], query["similarity_to_human_score"] = compute_conditional_proba_score(query["returns"], query["human_eval"])

    # Add the unconditional score to the dictionary
    checking_elements["unconditional_score"] = S_u

    return checking_elements
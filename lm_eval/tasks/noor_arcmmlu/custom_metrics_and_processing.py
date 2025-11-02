# In lm_eval/tasks/mmlu/default/custom_metrics_and_processing.py
import math
import numpy as np

def process_results_with_all_margins(doc, results):
    """
    This function is called for EACH example. It calculates standard accuracy,
    the original confidence margin, and the new average confidence margin.
    """
    # --- Standard Accuracy Calculation ---
    try:
        gold = int(doc["answer"])
        logprobs = [res[0] for res in results]
        pred = np.argmax(logprobs)
        acc = 1.0 if pred == gold else 0.0
    except Exception:
        acc = 0.0

    # --- All Margin Calculations ---
    try:
        logprobs = [res[0] for res in results]
        correct_index = int(doc["answer"])

        if not (0 <= correct_index < len(logprobs)):
            margin = -math.inf
            avg_margin = -math.inf
        else:
            logprob_correct = logprobs[correct_index]
            wrong_lps = [lp for i, lp in enumerate(logprobs) if i != correct_index]
            
            if not wrong_lps:
                # Handle binary/single choice case
                margin = math.inf
                avg_margin = math.inf
            else:
                # 1. Original Confidence Margin
                max_logprob_wrong = max(wrong_lps)
                margin = logprob_correct - max_logprob_wrong

                # 2. NEW: Average Confidence Margin
                individual_margins = [logprob_correct - lp for lp in wrong_lps]
                avg_margin = np.mean(individual_margins)

    except Exception:
        margin = -math.inf
        avg_margin = -math.inf
        
    # Return a dictionary containing scores for ALL metrics for this ONE example.
    return {
        "acc": acc,
        "confidence_margin": margin,
        "average_confidence_margin": avg_margin, # Add the new metric here
    }
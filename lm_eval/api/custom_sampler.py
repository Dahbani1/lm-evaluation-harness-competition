import random
from lm_eval.api import samplers

def get_fewshot_examples(task, doc, n, rng):
    fewshot_docs = list(task.dataset["fewshot"])
    fewshot_docs = [d for d in fewshot_docs if d != doc]
    rng.shuffle(fewshot_docs)
    return fewshot_docs[:n]

def random_n_0_5(task, doc, num_fewshot, rng):
    """Randomly pick between 0 and 5 few-shots (inclusive)."""
    n = rng.randint(0, 5)
    return get_fewshot_examples(task, doc, n, rng)

# Register AFTER samplers is imported (safe now)
samplers.SAMPLERS["random_n_0_5"] = random_n_0_5
print("[INFO] Custom sampler 'random_n_0_5' registered.")

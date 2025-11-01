import random
from lm_eval.api.registry import register_fewshot_sampler

@register_fewshot_sampler("random_n_0_5")
def random_n_0_5(dataset, num_fewshot, rng=None):
    """Sample a random number of fewshots between 0 and 5 inclusive."""
    n = random.randint(0, 5)
    if n == 0:
        return []  # zero-shot
    return dataset.select(range(n))

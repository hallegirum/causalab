"""Counterfactual dataset generator for InterpBench / tracr tasks.

Identical in shape to natural_domains_arithmetic: sample a base and a
counterfactual input from the causal model. Each ``sample_input`` draws every
sequence slot (tok_0 … tok_{L-1}) from the vocab and computes ``result`` /
``raw_input`` via the mechanisms.
"""

from causalab.causal.counterfactual_dataset import CounterfactualExample


def generate_dataset(model, n: int, seed: int = 42) -> list[CounterfactualExample]:
    """Generate n counterfactual examples using the given causal model."""
    import random

    state = random.getstate()
    random.seed(seed)
    examples: list[CounterfactualExample] = []
    for _ in range(n):
        input_sample = model.sample_input()
        counterfactual = model.sample_input()
        examples.append(
            {"input": input_sample, "counterfactual_inputs": [counterfactual]}
        )
    random.setstate(state)
    return examples

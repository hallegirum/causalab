"""Token positions for InterpBench / tracr tasks (index-based only).

tracr has no string template and no variables in the natural_domains sense, so
we only build **absolute index positions** — the one factory family that is
compatible with a non-HF tokenizer (it needs the token count, not char offsets).
See causalab/neural/token_positions.py: ``_build_variable_factory`` requires an
HF tokenizer's offset mapping and is therefore NOT used here.

Positions built (all read via ``pipeline.load(...)["input_ids"]``, which
``TracrPipeline.load`` aliases):
    last_token : index -1
    pos_{k}    : absolute token index k, for k in range(n_ctx)

Because each tracr input symbol occupies exactly one token at a fixed slot, the
sequence element ``tok_i`` sits at token position ``i + bos_offset`` — so
``pos_{i+bos_offset}`` is the site for intervening on that input (or its output).
"""

from __future__ import annotations

from typing import Any

from causalab.neural.token_positions import (
    build_token_position_factories,
    TokenPosition,
)


def _build_specs(n_ctx: int) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {"last_token": {"type": "index", "position": -1}}
    for k in range(n_ctx):
        specs[f"pos_{k}"] = {"type": "index", "position": k}
    return specs


def create_token_positions(
    pipeline,
    template: Any = None,      # tracr: unused. IOI: a dict of EXTRA named positions.
    templates: Any = None,     # (tracr has no templates)
) -> dict[str, TokenPosition]:
    """Build ``last_token`` + one ``pos_{k}`` per token slot.

    ``n_ctx`` (and hence the slot count) is read straight off the model, so no
    per-case config is needed here.

    Extra named positions (``template`` as a dict): a non-tracr case (IOI) can inject
    positions that aren't a fixed absolute index — e.g. the subject S2, whose token index
    varies per sentence. Values are ``build_token_position_factories`` specs: a dict
    (``{"type": "index", "position": k}``) for a fixed slot, or a callable
    ``spec_func(input_sample) -> spec`` for a per-example dynamic position. Names collide-
    override the built-in ``pos_{k}`` / ``last_token``.
    """
    n_ctx = int(pipeline.model.cfg.n_ctx)
    specs = _build_specs(n_ctx)
    if isinstance(template, dict):
        specs.update(template)  # IOI "subject" (dynamic) etc.
    factories = build_token_position_factories(specs, "")  # template unused by index specs
    return {name: factory(pipeline) for name, factory in factories.items()}

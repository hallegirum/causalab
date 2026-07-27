"""Case-agnostic causal models for InterpBench / tracr tasks.

A tracr case is fully described by an :class:`InterpBenchConfig` (vocabulary,
sequence length, and — the only case-specific line — ``hl_fn``, the high-level
algorithm). :func:`create_interpbench_causal_model` turns that into a
``CausalModel`` following the same convention as ``natural_domains_arithmetic``.

Loader contract
---------------
``load_task("interpbench", task_cfg={"case": "18"})`` calls ``CREATE_CAUSAL_MODEL``
below, which looks the case up in :data:`CASES`. Register a new case by adding an
``InterpBenchConfig`` to that dict — no other file changes.

Structural note (vs natural_domains)
------------------------------------
tracr's input is a *sequence*, so instead of scalar ``entity``/``number`` inputs
we have **one input variable per sequence slot** (``tok_0 … tok_{L-1}``). tracr
models emit an output at *every* position, so we expose **one output variable per
position** (``out_0 … out_{L-1}``); ``result`` aliases the studied position
(``target_pos``, default the last). ``raw_input`` reassembles the slots into the
``[BOS, sym, …]`` list the model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import Mechanism, input_var


# --------------------------------------------------------------------------- #
#  Per-case configuration
# --------------------------------------------------------------------------- #
@dataclass
class InterpBenchConfig:
    """Everything needed to build one tracr case's causal model.

    Attributes
    ----------
    case : str
        InterpBench case id (naming / model-loading only).
    vocab : list
        Input symbol alphabet (what may appear at each sequence slot).
    seq_len : int
        Number of sequence slots, EXCLUDING BOS. Must equal the model's
        ``cfg.n_ctx - bos_offset`` (checked by :func:`verify_against_model`).
    hl_fn : Callable[[list], list]
        The ground-truth algorithm: maps the symbol sequence (length ``seq_len``)
        to a **per-position** output list (also length ``seq_len``). For an
        aggregation task, broadcast the answer across positions.
    out_vocab : list
        Domain of the output labels (the classes ``result`` can take).
    bos : Any
        BOS symbol tracr prepends (``None`` to disable). Sets ``bos_offset``.
    target_pos : int
        Which position's output becomes the scalar ``result`` (-1 = last).
    embeddings, periods : optional
        Only for cyclic cases (e.g. modular arithmetic); usually empty.
    """

    case: str
    vocab: list
    seq_len: int
    hl_fn: Callable[[list], list]
    out_vocab: list
    bos: Any = "BOS"
    target_pos: int = -1
    embeddings: dict[str, Callable] | None = None
    periods: dict[str, float] | None = None

    @property
    def bos_offset(self) -> int:
        return 1 if self.bos is not None else 0


def create_interpbench_causal_model(cfg: InterpBenchConfig) -> CausalModel:
    """Build the ``CausalModel`` for one tracr case (see module docstring)."""
    L = cfg.seq_len
    slots = [f"tok_{i}" for i in range(L)]

    def _outputs(t) -> list:
        # full per-position output list from the ground-truth algorithm
        return cfg.hl_fn([t[s] for s in slots])

    mechanisms: dict[str, Mechanism] = {
        # --- inputs: one variable per sequence slot (parents=[] -> sampled) ---
        **{s: input_var(cfg.vocab) for s in slots},
        # --- the sequence the model actually consumes ---
        "raw_input": Mechanism(
            parents=slots,
            compute=lambda t: (
                ([cfg.bos] if cfg.bos is not None else []) + [t[s] for s in slots]
            ),
        ),
    }
    # --- one output variable per position ---
    for i in range(L):
        mechanisms[f"out_{i}"] = Mechanism(
            parents=slots, compute=lambda t, i=i: _outputs(t)[i]
        )
    # --- the studied output (scalar) + its label form ---
    tgt = f"out_{cfg.target_pos % L}"  # -1 -> out_{L-1}
    mechanisms["result"] = Mechanism(parents=[tgt], compute=lambda t, k=tgt: t[k])
    mechanisms["raw_output"] = Mechanism(parents=["result"], compute=lambda t: t["result"])

    # --- domain table: every variable needs an entry (computed ones -> None) ---
    values: dict[str, Any] = {s: list(cfg.vocab) for s in slots}
    values["result"] = list(cfg.out_vocab)
    for i in range(L):
        values[f"out_{i}"] = list(cfg.out_vocab)
    values["raw_input"] = None  # not sampled/enumerated
    values["raw_output"] = None

    model = CausalModel(
        mechanisms,
        values,
        id=f"interpbench_case{cfg.case}",
        embeddings=cfg.embeddings or {},
        periods=cfg.periods or {},
    )
    model._ib_cfg = cfg  # type: ignore[attr-defined]  # stash for token_positions / verify
    return model


# --------------------------------------------------------------------------- #
#  Case registry
# --------------------------------------------------------------------------- #
def _count_last_symbol(seq: list) -> list:
    """Illustrative aggregation task: how many times the FINAL symbol appears.
    Broadcast across positions (read at the last)."""
    ans = seq.count(seq[-1])
    return [ans] * len(seq)


def _token_frequency_classifier(seq: list) -> list:
    """Case 18 — classify each token by how often it appears in the sequence.

    ratio = count(token) / len(seq):
        ratio >= 0.50 -> "frequent"      (0.5 counts as frequent, per the case's example)
        0.25 <  ratio -> "common"
        else          -> "rare"          ("rare" is <= 25%)

    NOTE: thresholds inferred from the case's documented example
    (a=3/6 -> frequent, b=2/6 -> common, c=1/6 -> rare). verify_against_model()
    confirms them against the real HookedTransformer — adjust if it flags a mismatch.
    """
    L = len(seq)
    out = []
    for x in seq:
        ratio = seq.count(x) / L
        if ratio >= 0.5:
            out.append("frequent")
        elif ratio > 0.25:
            out.append("common")
        else:
            out.append("rare")
    return out


CASES: dict[str, InterpBenchConfig] = {
    # runnable with NO InterpBench model — unit-tests the causal-model logic only
    "count_last_symbol": InterpBenchConfig(
        case="count_last_symbol",
        vocab=["a", "b", "c"],
        seq_len=3,
        hl_fn=_count_last_symbol,
        out_vocab=[1, 2, 3],
        target_pos=-1,
    ),
    # real trained model on HF: cybershiptrooper/InterpBench, subfolder "18"
    "18": InterpBenchConfig(
        case="18",
        vocab=["a", "b", "c", "d", "e"],  # get_ascii_letters_vocab(count=5)
        seq_len=10,  # <-- must equal model n_ctx - bos_offset; verify_against_model checks
        hl_fn=_token_frequency_classifier,
        out_vocab=["frequent", "common", "rare"],
        target_pos=-1,  # per-position task: any position is valid; last by default
    ),
}


# --------------------------------------------------------------------------- #
#  Loader-convention exports (read by causalab.tasks.loader.load_task)
# --------------------------------------------------------------------------- #
def CREATE_CAUSAL_MODEL(task_cfg: dict) -> CausalModel:
    """Factory entry point. ``task_cfg`` must contain ``{"case": <id>}``; any
    other keys (e.g. ``seq_len``, ``target_pos``) override the registered config."""
    case = str(task_cfg["case"])
    if case not in CASES:
        raise KeyError(f"Unknown InterpBench case '{case}'. Known: {sorted(CASES)}")
    cfg = CASES[case]
    overrides = {k: v for k, v in task_cfg.items() if k != "case" and hasattr(cfg, k)}
    if overrides:
        cfg = replace(cfg, **overrides)
    return create_interpbench_causal_model(cfg)


TARGET_VARIABLE = "result"


def GET_TEMPLATE(model: CausalModel):
    """tracr has no string template; token positions are index-based."""
    return None


# --------------------------------------------------------------------------- #
#  Verification harness — the "test" for a real case (needs the model)
# --------------------------------------------------------------------------- #
def verify_against_model(
    cfg: InterpBenchConfig,
    pipeline,
    n: int = 200,
    seed: int = 0,
    show: int = 5,
) -> None:
    """Confirm ``hl_fn`` reproduces the real HookedTransformer's behaviour.

    Checks (a) ``seq_len + bos_offset == model.cfg.n_ctx`` and (b) for ``n``
    sampled inputs, the model's per-position argmax outputs (BOS stripped) equal
    ``hl_fn(seq)``. Requires ``pipeline`` to be a ``TracrPipeline`` whose
    ``decode`` returns one label per token position. Raises AssertionError on
    mismatch; prints the first ``show`` comparisons.
    """
    import random

    model = create_interpbench_causal_model(cfg)
    n_ctx = int(pipeline.model.cfg.n_ctx)
    assert cfg.seq_len + cfg.bos_offset == n_ctx, (
        f"seq_len({cfg.seq_len}) + bos_offset({cfg.bos_offset}) != model n_ctx({n_ctx}). "
        f"Set cfg.seq_len = {n_ctx - cfg.bos_offset}."
    )

    rng = random.Random(seed)
    mismatches = 0
    for j in range(n):
        trace = model.sample_input(filter_func=None)
        seq = [trace[f"tok_{i}"] for i in range(cfg.seq_len)]
        expected = cfg.hl_fn(seq)  # per-position, length seq_len

        result = pipeline.generate([{"raw_input": trace["raw_input"]}])
        got_full = result["string"]  # per-position labels incl. BOS position
        got = got_full[cfg.bos_offset : cfg.bos_offset + cfg.seq_len]

        if list(got) != list(expected):
            mismatches += 1
            if mismatches <= show:
                print(f"  MISMATCH seq={seq}\n    expected {expected}\n    got      {list(got)}")
        elif j < show:
            print(f"  ok seq={seq} -> {list(expected)}")

    assert mismatches == 0, (
        f"{mismatches}/{n} inputs disagree with the model — hl_fn for case "
        f"'{cfg.case}' is wrong (or encode/decode mis-wired)."
    )
    print(f"verify_against_model: PASS ({n} samples, case {cfg.case})")

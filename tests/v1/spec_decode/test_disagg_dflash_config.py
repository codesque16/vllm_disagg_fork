# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Design A disagg DFlash config + context-length tracking."""

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]

from vllm.config.speculative import SpeculativeConfig


def _bare_spec_config(**kwargs) -> SpeculativeConfig:
    """Build a SpeculativeConfig without running full __post_init__ / HF loads."""
    cfg = object.__new__(SpeculativeConfig)
    cfg.method = None
    cfg.disagg_spec_role = None
    cfg.num_speculative_tokens = 1
    cfg.use_heterogeneous_vocab = False
    cfg.draft_sample_method = "greedy"
    cfg.target_model_config = None
    cfg.draft_model_config = None
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def test_disagg_spec_role_helpers():
    cfg = _bare_spec_config(
        method="dflash",
        num_speculative_tokens=4,
        disagg_spec_role="producer",
    )
    assert cfg.is_disagg_spec_producer()
    assert not cfg.is_disagg_spec_consumer()

    cfg = _bare_spec_config(
        method="dflash",
        num_speculative_tokens=4,
        disagg_spec_role="consumer",
    )
    assert cfg.is_disagg_spec_consumer()
    assert not cfg.is_disagg_spec_producer()


def test_disagg_spec_role_rejects_non_dflash():
    cfg = _bare_spec_config(
        method="eagle3",
        num_speculative_tokens=3,
        disagg_spec_role="producer",
    )
    if pytest is None:
        try:
            cfg._verify_args()
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "only supported with method='dflash'" in str(e)
        return
    with pytest.raises(ValueError, match="only supported with method='dflash'"):
        cfg._verify_args()


def test_drafter_context_lens_seed_and_advance():
    # Lightweight stand-in: exercise the tracking API without loading weights.
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    for cls in (DFlashProposer, DFlashSpeculator):

        class _Tracker:
            def __init__(self):
                self._drafter_context_lens: dict[str, int] = {}

            seed_drafter_context_lens = cls.seed_drafter_context_lens
            clear_drafter_context_lens = cls.clear_drafter_context_lens
            advance_drafter_context_lens = cls.advance_drafter_context_lens

        t = _Tracker()
        t.seed_drafter_context_lens(["r1", "r2"], [128, 64])
        assert t._drafter_context_lens == {"r1": 128, "r2": 64}

        # Re-seed must not overwrite (connector may report computed tokens again).
        t.seed_drafter_context_lens(["r1"], [999])
        assert t._drafter_context_lens["r1"] == 128

        t.advance_drafter_context_lens(["r1"], [1])
        assert t._drafter_context_lens["r1"] == 129

        t.clear_drafter_context_lens(["r1"])
        assert "r1" not in t._drafter_context_lens
        assert t._drafter_context_lens["r2"] == 64


if __name__ == "__main__":
    test_disagg_spec_role_helpers()
    test_disagg_spec_role_rejects_non_dflash()
    test_drafter_context_lens_seed_and_advance()
    print("ok")

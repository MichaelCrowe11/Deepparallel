from deepparallel import DeepParallelReasoning, DeepParallelEnsemble, AnswerVerifier


def test_reasoning_synthesis():
    dpr = DeepParallelReasoning()
    out = dpr.parallel_reason("What is energy conservation?")
    assert "Synthesis" in out
    assert "Final" in out


def test_ensemble_answer():
    ens = DeepParallelEnsemble()
    ans, conf = ens.answer("State the first law of thermodynamics")
    assert isinstance(ans, str)
    assert 0.0 < conf <= 1.0

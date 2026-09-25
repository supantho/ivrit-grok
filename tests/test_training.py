"""Training-pipeline tests (skipped when torch is unavailable)."""
import pytest

torch = pytest.importorskip("torch")

from hebmorph.loader import Example  # noqa: E402
from training.data import EOS, PAD, Vocab, encode  # noqa: E402
from training.model import ModelConfig, TinyGPT, _forward_left_padded, greedy_decode, lm_loss  # noqa: E402

EX = [Example("כָּתַב", "PST,1,SG", "כָּתַבְתִּי"), Example("כָּתַב", "PST,2,SG,FEM", "כָּתַבְתְּ"),
      Example("לָמַד", "PST,1,SG", "לָמַדְתִּי")]


def test_vocab_roundtrip_and_marks_are_tokens():
    v = Vocab.build(EX)
    assert "ָ" in v.stoi and "F:FEM" in v.stoi          # niqqud mark and feature are tokens
    assert v.dec_chars(v.enc_chars("כָּתַבְתִּי") + [EOS]) == "כָּתַבְתִּי"
    assert Vocab.from_json(v.to_json()).itos == v.itos


def test_loss_mask_covers_exactly_target_and_eos():
    v = Vocab.build(EX)
    e = encode(EX, v, 48)
    for i, ex in enumerate(EX):
        assert int(e.loss_mask[i].sum()) == len(ex.target_form) + 1   # target chars + <eos>


def test_example_type_has_only_visible_fields():
    assert set(Example.__dataclass_fields__) == {"source_form", "target_features", "target_form"}


def test_left_padding_does_not_change_predictions():
    torch.manual_seed(0)
    m = TinyGPT(ModelConfig(vocab_size=30, max_len=16, d_model=32, n_layers=2, n_heads=2, d_mlp=64)).eval()
    x = torch.tensor([[1, 7, 8, 9, 2]])
    padded = torch.tensor([[PAD, PAD, 1, 7, 8, 9, 2]])
    a = m(x)[0, -1]
    b = _forward_left_padded(m, padded, PAD)[0, -1]
    assert torch.isfinite(b).all() and torch.allclose(a, b, atol=1e-5)


def test_tiny_model_memorises():
    torch.manual_seed(0)
    v = Vocab.build(EX)
    e = encode(EX, v, 48)
    m = TinyGPT(ModelConfig(vocab_size=len(v), max_len=48, d_model=64, n_layers=2, n_heads=2, d_mlp=128))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = lm_loss(m(e.tokens), e.tokens, e.loss_mask)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05 * first
    prompts = [e.tokens[i, : int(e.prompt_len[i])] for i in range(len(EX))]
    out = greedy_decode(m.eval(), prompts, max_new=20, eos_id=EOS)
    assert [v.dec_chars(o.tolist()) for o in out] == [x.target_form for x in EX]


def test_layernorm_free_variant_and_decay_groups():
    from training.train import TrainConfig
    m = TinyGPT(ModelConfig(vocab_size=30, max_len=16, d_model=32, n_layers=2, n_heads=2, d_mlp=64, use_layernorm=False))
    assert not any("ln" in n for n, _ in m.named_parameters())
    assert torch.isfinite(m(torch.tensor([[1, 5, 6, 2]]))).all()
    assert TrainConfig().decay_norm_and_bias is True   # Power et al.: AdamW decays every parameter


def test_analysis_loader_rows_match_model_loader():
    from hebmorph import paths
    from hebmorph.loader import load_split, load_split_analysis
    base = paths.DERIVED / "synthetic" / "v1"
    if not (base / "splits").exists():
        pytest.skip("synthetic splits not generated")
    a = load_split("past_reinflection", "root_holdout_seed0", "test", "voc", base, base / "splits")
    b = load_split_analysis("past_reinflection", "root_holdout_seed0", "test", "voc", base, base / "splits")
    assert [(e.source_form, e.target_features, e.target_form) for e in a] == \
        list(zip(b.source_form, b.target_features, b.target_form))


def test_model_family_forward_and_hidden():
    from training.model import CausalMLPLM, LSTMConfig, LSTMLM, MLPConfig, forward_hidden
    x = torch.tensor([[1, 5, 6, 2, 7, 2, 8, 3]])
    for m in [LSTMLM(LSTMConfig(vocab_size=12, max_len=16, d_emb=8, d_hidden=16, n_layers=2)),
              CausalMLPLM(MLPConfig(vocab_size=12, max_len=16, d_emb=4, d_hidden=16, n_layers=2, window=16))]:
        logits, hs = forward_hidden(m, x)
        assert logits.shape == (1, 8, 12) and all(h.shape[:2] == (1, 8) for h in hs)
        # causality: changing a later token must not change earlier logits
        y = x.clone(); y[0, 6] = 9
        assert torch.allclose(m(x)[0, :6], m(y)[0, :6], atol=1e-6)


def test_teacher_forcing_exact_match_equals_greedy():
    """exact_match_method='teacher_forcing' must give the same count as greedy decoding,
    also for a half-trained model where many answers are wrong."""
    from training.train import evaluate
    torch.manual_seed(0)
    ex = [Example(s, f, t) for s, f, t in [("כָּתַב", "PST,1,SG", "כָּתַבְתִּי"), ("כָּתַב", "PST,3,PL", "כָּתְבוּ"),
                                            ("לָמַד", "PST,1,SG", "לָמַדְתִּי"), ("לָמַד", "PST,3,PL", "לָמְדוּ"),
                                            ("שָׁמַר", "PST,1,SG", "שָׁמַרְתִּי"), ("שָׁמַר", "PST,3,PL", "שָׁמְרוּ")]]
    v = Vocab.build(ex)
    e = encode(ex, v, 48)
    m = TinyGPT(ModelConfig(vocab_size=len(v), max_len=48, d_model=32, n_layers=1, n_heads=2, d_mlp=64))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for step in range(120):
        loss = lm_loss(m(e.tokens), e.tokens, e.loss_mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            a = evaluate(m, e, v, "cpu", method="greedy")["exact_match"]
            b = evaluate(m, e, v, "cpu", method="teacher_forcing")["exact_match"]
            assert a == b, (step, a, b)


def test_synthetic_v2_weak_rules_match_hebrew():
    from hebmorph.synthetic_v2 import realize, templates
    def f(root, b, c):
        return realize(templates(b)[c], root, b, c, final_forms=True)
    assert f(list("נפל"), "HIFIL", "PST.3MSG") == "הִפִּיל"
    assert f(list("נפל"), "PAAL", "FUT.3MSG") == "יִפֹּל"
    assert f(list("נפל"), "PAAL", "PST.3MSG") == "נָפַל"            # nun kept when it has a vowel
    assert f(["י", "שׁ", "ב"], "HIFIL", "PST.3MSG") == "הוֹשִׁיב"
    assert f(["י", "שׁ", "ב"], "NIFAL", "PST.3MSG") == "נוֹשַׁב"
    assert f(list("בנה"), "PAAL", "PST.3FSG") == "בָנְתָה"
    assert f(list("בנה"), "PAAL", "PST.3PL") == "בָנוּ"
    assert f(list("בנה"), "PAAL", "PST.1SG") == "בָנִיתִי"
    assert f(list("בנה"), "PAAL", "FUT.3MSG") == "יִבְנֶה"
    assert f(list("קומ"), "PAAL", "PST.1SG") == "קַמְתִּי"
    assert f(list("קומ"), "PAAL", "PRS.MPL") == "קָמִים"
    assert f(list("כתב"), "PAAL", "PST.1SG") == "כָתַבְתִּי"         # strong: pure slot filling

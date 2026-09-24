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

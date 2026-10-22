import numpy as np
import pytest
import torch

from app.config import settings
from app.pipeline.ocr_engines import CRNN, CRNNEngine, crnn_input, greedy_decode
from conftest import written_page


def test_input_is_64_high_inverted_and_width_clamped():
    x = crnn_input(np.full((32, 100), 255, np.uint8))
    assert x.shape == (1, 1, 64, 200) and float(x.max()) == 0.0  # white paper -> 0
    assert crnn_input(np.zeros((100, 10), np.uint8)).shape[-1] == 16
    assert crnn_input(np.zeros((40, 5000), np.uint8)).shape[-1] == 1024


def test_architecture_downsamples_width_by_4():
    out = CRNN(226).eval()(torch.zeros(1, 1, 64, 200))
    assert out.shape == (1, 50, 226)


def test_greedy_ctc_decoding_merges_repeats_and_drops_blanks():
    charset = ["a", "b", " "]
    seq = [1, 1, 0, 1, 2, 0, 0, 3, 2]  # a a _ a b _ _ ' ' b -> "aab b"
    logits = torch.full((len(seq), 4), -10.0)
    for t, k in enumerate(seq):
        logits[t, k] = 10.0
    logits[3, 1], logits[3, 2] = 0.0, -0.5  # less sure about the second "a"
    text, confs = greedy_decode(logits, charset)
    assert text == "aab b" and len(confs) == len(text)
    assert confs[1] < 0.9 < confs[0]


@pytest.mark.skipif(not settings.CRNN_MODEL_PATH.exists(), reason="models/crnn_best.pt not present")
def test_real_crnn_loads_and_reads_segmented_lines():
    eng = CRNNEngine()  # strict load_state_dict: fails if the architecture differs from training
    assert len(eng.charset) + 1 == eng.model.fc.out_features
    page = eng.read_page(written_page(2))
    assert page.image.shape == (600, 800)
    for item in page.lines:
        assert len(item.char_confs) == len(item.text)
        assert all(0.0 <= c <= 1.0 for c in item.char_confs)
        assert item.y1 - item.y0 > 15

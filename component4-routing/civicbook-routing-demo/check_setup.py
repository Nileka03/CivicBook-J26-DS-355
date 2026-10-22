"""Checks the demo set-up before you start it:  python check_setup.py"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
ok = True


def say(good, msg):
    global ok
    ok &= good
    print(("  OK    " if good else "  FIX   ") + msg)


print("\n1. Code files")
pipe = (ROOT / "app" / "pipeline.py").read_text(encoding="utf-8")
js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
say("sentence_comma_split_" in pipe, "app/pipeline.py is the latest version (full stop + comma split)")
say("split_at_comma" in js, "static/app.js is the latest version")

print("\n2. Model folder:", MODEL)
for name in ["routing_config.json", "best_model.pt", "tokenizer.json", "tokenizer_config.json"]:
    say((MODEL / name).exists(), name + (" found" if (MODEL / name).exists() else " MISSING"))
nested = [p for p in MODEL.rglob("best_model.pt") if p.parent != MODEL]
if nested:
    say(False, f"best_model.pt is inside a sub-folder ({nested[0].parent}); move it directly into model/")

cfg_path = MODEL / "routing_config.json"
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("\n3. routing_config.json")
    print("        threshold_institution =", cfg.get("threshold_institution"))
    print("        threshold_department  =", cfg.get("threshold_department"))
    print("        postprocess           =", cfg.get("postprocess", "(not set -> merge_same_department_no_connector)"))
    say(cfg.get("threshold_department") == 0.85, "threshold_department is 0.85")
    say("postprocess" in cfg, "postprocess is set")

w = MODEL / "best_model.pt"
if w.exists():
    md5 = hashlib.md5()
    with open(w, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    print("\n4. best_model.pt")
    print(f"        size {w.stat().st_size / 1e6:.0f} MB, md5 {md5.hexdigest()}")
    say(w.stat().st_size > 900e6, "size looks like XLM-R base (about 1.1 GB)")

if not ok:
    print("\nFix the lines marked FIX, then run this again.")
    sys.exit(1)

print("\n5. Loading the model and routing two invented complaints ...")
sys.path.insert(0, str(ROOT))
from app import model as loader            # noqa: E402
from app.pipeline import RoutingEngine     # noqa: E402

m, tok, cfg = loader.load(MODEL)
eng = RoutingEngine(m, tok, cfg)
for text in ["My electricity bill is much higher than usual this month. Also the drain near our house is blocked.",
             "මගේ විදුලි බිල මේ මාසේ ගොඩක් වැඩියි. ඒ වගේම අපේ පාරේ කානු අවහිර වෙලා."]:
    r = eng.route(text, False)
    print("\n   ", text)
    for s in r["sub_complaints"]:
        print(f"      -> {s['institution']} / {s['department']}   |  {s['sub_complaint_text']}")
print("\nAll good. Start the demo with:  uvicorn app.main:app --port 8000\n")

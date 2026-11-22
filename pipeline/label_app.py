"""Blind cross-check of the render labels: sketch on the left, the agent's page on the right.
Claude labels every render first (claude_label + reason). You label it yourself WITHOUT seeing Claude's answer;
only after your click does the page show Claude's label and whether you agree. That keeps the agreement rate honest.
Rules: data/labels/LABELLING.md.
    streamlit run pipeline/label_app.py
Keys: P = pass, F = fail, U = unsure, B = back. Every click is saved at once to data/labels/render_labels.csv
(column human_label), so you can stop any time and continue later."""
import argparse
import datetime
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "shared" imports when run by streamlit

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from shared import config  # noqa: E402

DATA = config.ROOT / "data"
RENDERS = DATA / "renders" / "renders.csv"
MANIFEST = DATA / "validated" / "manifest.csv"
RULES = DATA / "labels" / "LABELLING.md"
COLUMNS = ["sketch_id", "round", "claude_label", "claude_reason", "human_label", "human_labelled_at"]
NAMES = {"1": "PASS", "0": "FAIL", "unsure": "UNSURE"}

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=str(DATA / "labels" / "render_labels.csv"))
LABELS = Path(ap.parse_args(sys.argv[1:]).labels)


def load_renders():
    r = pd.read_csv(RENDERS, dtype={"round": str})
    m = pd.read_csv(MANIFEST, dtype=str)[["sketch_id", "sketch_path"]]
    r = r.merge(m, on="sketch_id").sort_values(["sketch_id", "round"]).reset_index(drop=True)
    order = list(range(len(r)))
    random.Random(0).shuffle(order)  # random order, so you don't check one sketch's rounds back to back
    return r.iloc[order].reset_index(drop=True)


def load_labels():
    lab = pd.read_csv(LABELS, dtype=str, keep_default_na=False) if LABELS.exists() else pd.DataFrame(columns=COLUMNS)
    return lab.reindex(columns=COLUMNS, fill_value="")


def save_human(sketch_id, rnd, value):
    """Set human_label for one render (keeping Claude's label), then write the file atomically."""
    lab = load_labels()
    hit = (lab["sketch_id"] == sketch_id) & (lab["round"] == rnd)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if hit.any():
        lab.loc[hit, ["human_label", "human_labelled_at"]] = [value, now]
    else:  # a render Claude has not labelled yet
        lab = pd.concat([lab, pd.DataFrame([dict(sketch_id=sketch_id, round=rnd, human_label=value,
                                                 human_labelled_at=now)])]).reindex(columns=COLUMNS, fill_value="")
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    tmp = LABELS.with_suffix(".tmp")
    lab.sort_values(["sketch_id", "round"]).to_csv(tmp, index=False)
    tmp.replace(LABELS)


st.set_page_config(page_title="Check render labels", layout="wide")
renders = load_renders()
lab = load_labels().set_index(["sketch_id", "round"])
get = lambda r, col: lab[col].get((r["sketch_id"], r["round"]), "")  # noqa: E731
todo = [i for i, r in renders.iterrows() if not get(r, "human_label")]

if "i" not in st.session_state:
    st.session_state.i = todo[0] if todo else 0


def label_and_next(value):
    r = renders.iloc[st.session_state.i]
    save_human(r["sketch_id"], r["round"], value)
    st.session_state.just_did = (st.session_state.i, value)
    later = [i for i in todo if i > st.session_state.i]
    st.session_state.i = later[0] if later else st.session_state.i


def go(step):
    st.session_state.i = max(0, min(len(renders) - 1, st.session_state.i + step))
    st.session_state.pop("just_did", None)


# feedback on the render you just labelled: only now is Claude's answer revealed
if "just_did" in st.session_state:
    i, value = st.session_state.pop("just_did")
    prev = renders.iloc[i]
    c = get(prev, "claude_label")
    if c and value != "unsure":
        same = c == value
        st.toast(f"{'✅ Agree' if same else '⚠️ Disagree'}: you said {NAMES[value]}, Claude said {NAMES[c]}"
                 + ("" if same else f" — {get(prev, 'claude_reason')}"))

with st.sidebar:
    checked = lab[(lab["human_label"].isin(["0", "1"])) & (lab["claude_label"].isin(["0", "1"]))]
    st.metric("Checked by you", f"{(lab['human_label'] != '').sum()} / {len(renders)}")
    st.progress((lab["human_label"] != "").sum() / max(len(renders), 1))
    if len(checked):
        agree = (checked["human_label"] == checked["claude_label"]).mean()
        st.metric("Agreement with Claude", f"{agree:.0%}", help=f"on {len(checked)} renders both labelled pass/fail")
    with st.expander("Labelling rules"):
        st.markdown(RULES.read_text() if RULES.exists() else "LABELLING.md not found")
    show = st.checkbox("Show Claude's label before I decide", value=False,
                       help="Leave this off for an honest check: seeing the answer first makes you agree with it.")

if not todo:
    st.success("You have checked every render. You can still go back and change any label.")

r = renders.iloc[st.session_state.i]
mine = get(r, "human_label")
st.markdown(f"**{st.session_state.i + 1} / {len(renders)}** · `{r['sketch_id']}` · round {r['round']} · "
            f"your label: **{NAMES.get(mine, 'not labelled yet')}**")
if (mine or show) and get(r, "claude_label"):
    st.caption(f"Claude: {NAMES[get(r, 'claude_label')]} — {get(r, 'claude_reason')}")

b = st.columns(5)
b[0].button("✅ Pass", on_click=label_and_next, args=("1",), shortcut="p", use_container_width=True, type="primary")
b[1].button("❌ Fail", on_click=label_and_next, args=("0",), shortcut="f", use_container_width=True)
b[2].button("❓ Unsure", on_click=label_and_next, args=("unsure",), shortcut="u", use_container_width=True)
b[3].button("← Back", on_click=go, args=(-1,), shortcut="b", use_container_width=True)
b[4].button("Next →", on_click=go, args=(1,), use_container_width=True)

left, right = st.columns(2)
left.markdown("**Sketch**")
left.image(str(DATA / "raw" / r["sketch_path"]), use_container_width=True)
right.markdown("**Agent's page**")
right.image(str(DATA / "renders" / r["render_png"]), use_container_width=True)

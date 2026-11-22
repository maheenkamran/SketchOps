# Labelling rules for agent renders

**Question:** *Would the person who drew this sketch accept this page as what they drew?*
We judge the **layout**, not the looks. The scorer is trained on exactly this question.

## PASS (1): all of these hold
1. **Every main section** of the sketch is there (header/nav, hero, card rows, text blocks, footer…).
2. The sections are in the **same order**, top to bottom.
3. **Columns match:** 3 boxes side by side in the sketch means 3 side by side on the page (not stacked).
4. Things are **roughly where they were drawn**: a box on the right is on the right, a centred title is centred.
5. The page **rendered properly**: not blank, not broken, no raw code showing.

## Does NOT matter (still PASS)
- The text: squiggles may become any sensible text.
- Colours, fonts, shadows, icons, exact sizes and spacing.
- Placeholder images (grey or coloured blocks) where the sketch has an X box.
- Whether it looks like a finished site or a wireframe. That is a prompt-style question, not a layout one.
- Small extras such as a button or an icon the sketch didn't show.

## FAIL (0): any one of these
- A main section is **missing**, or a big section was **added** that isn't in the sketch.
- Sections in the **wrong order**.
- **Wrong column layout:** for example 3 columns became 1 stacked column, or a sidebar disappeared.
- The page is **blank, broken or mostly cut off**.

## UNSURE (skip)
Use it when you really can't decide, for example when the sketch itself is unreadable. Unsure rows are left out of
training. Aim for under 5%.

## Who labels: Claude first, then a blind human check
1. **Claude** (the AI assistant used in this project) labels every render with these rules and writes a one-line
   reason: columns `claude_label`, `claude_reason`.
2. **A team member re-labels renders without seeing Claude's label** (`streamlit run pipeline/label_app.py`),
   saved as `human_label`. Claude's answer is only shown after the click, so it can't sway the human.
3. Training uses the **human label when there is one, otherwise Claude's** (`label_source` in pairs.csv says
   which). A human "unsure" drops the row.
4. `reports/pairs.json` → `render_labels` reports how many renders were checked and the **agreement rate**.
   This is the measure of label quality, to be quoted in the report.

## Notes
- Judge each round on its own. Don't compare round 1 with round 2.
- Decide in a few seconds. If you need to think hard, it's probably UNSURE.
- File: `render_labels.csv` in this folder. Labels are `1`, `0` or `unsure`.

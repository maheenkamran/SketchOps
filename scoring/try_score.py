"""Quick manual test:  python -m scoring.try_score sketch.jpg page.html
Renders page.html and prints clip / judge / final scores. Use it to calibrate CLIP_LOW/HIGH."""
import sys

from sandbox.render import render
from scoring.clip_score import clip_score
from scoring.judge import judge
from scoring.score import score

sk = open(sys.argv[1], "rb").read()
shot = render(open(sys.argv[2], encoding="utf-8").read())
open("last_screenshot.png", "wb").write(shot)
print("clip :", round(clip_score(sk, shot), 3))
print("judge:", judge(sk, shot))
print("final:", score(sk, shot))

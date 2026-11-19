"""Quick manual test:  python -m neuroassist "How many people are in this image?" photo.jpg"""
import json
import sys

from .pipeline import answer

if len(sys.argv) < 2:
    sys.exit('usage: python -m neuroassist "question" [file]')
a = answer(sys.argv[1], sys.argv[2:])
print(a.answer)
print(json.dumps({k: v for k, v in a.to_dict().items() if k != "answer"}, indent=2, default=str)[:3000])

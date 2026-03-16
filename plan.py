import re

with open("src/core/structure.py", "r", encoding="utf-8") as f:
    structure_text = f.read()

with open("src/core/indexing.py", "r", encoding="utf-8") as f:
    indexing_text = f.read()

print(len(structure_text), len(indexing_text))

import re

with open("src/core/generation.py", "r", encoding="utf-8") as f:
    text = f.read()

# Extract prompts block
start_marker = "# ── System prompts ───────────────────────────────────────────────────────────"
end_marker = "def _response_looks_incomplete(text: str) -> bool:"

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

if start_idx != -1 and end_idx != -1:
    block = text[start_idx:end_idx]
    
    # create prompts.py
    prompts_code = '"""System prompts and generation addendums."""\n\n'
    prompts_code += block.replace("_SYSTEM_PROMPT_BASE", "SYSTEM_PROMPT_BASE") \
                         .replace("_SECTION_LIST_ADDENDUM", "SECTION_LIST_ADDENDUM") \
                         .replace("_MULTI_SECTION_ADDENDUM", "MULTI_SECTION_ADDENDUM") \
                         .replace("_CHAT_SYSTEM_PROMPT", "CHAT_SYSTEM_PROMPT") \
                         .replace("def _chat_style_addendum", "def chat_style_addendum")
    
    with open("src/core/prompts.py", "w", encoding="utf-8") as f:
        f.write(prompts_code)
    
    # modify generation.py
    new_imports = """
from .prompts import (
    SYSTEM_PROMPT_BASE,
    SECTION_LIST_ADDENDUM,
    MULTI_SECTION_ADDENDUM,
    CHAT_SYSTEM_PROMPT,
    chat_style_addendum,
)

"""
    new_text = text[:start_idx] + new_imports + text[end_idx:]
    
    # also replace Usages in generation.py
    new_text = new_text.replace("_SYSTEM_PROMPT_BASE", "SYSTEM_PROMPT_BASE") \
                       .replace(" _SECTION_LIST_ADDENDUM", " SECTION_LIST_ADDENDUM") \
                       .replace("_MULTI_SECTION_ADDENDUM", "MULTI_SECTION_ADDENDUM") \
                       .replace("_CHAT_SYSTEM_PROMPT", "CHAT_SYSTEM_PROMPT") \
                       .replace("_chat_style_addendum", "chat_style_addendum")

    with open("src/core/generation.py", "w", encoding="utf-8") as f:
        f.write(new_text)
    
    print("split OK")


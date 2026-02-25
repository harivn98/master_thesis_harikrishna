#!/usr/bin/env python3
"""
Replace em dashes (—) in thesis files with commas, semicolons, or colons
depending on context:

1. PAIRED em dashes (parenthetical):  "word — phrase — word"  →  "word, phrase, word"
2. SINGLE em dash before a clause:     "sentence — clause"    →  "sentence; clause" or ": clause"
   - If what follows is an example/list/explanation → colon
   - Otherwise → semicolon

This script processes each file, finds em dashes, and replaces them contextually.
"""

import os
import re
import glob

THESIS_DIR = os.path.dirname(os.path.abspath(__file__))

SKIP_FILES = {
    "replace_emdashes.py",
    "reformat_for_word.py",
    "BST_Thesis_Template (2).txt",
}

# Words/patterns that suggest the em dash introduces an example or explanation
COLON_TRIGGERS = re.compile(
    r'^(exactly|specifically|namely|that is|for example|for instance|'
    r'including|such as|particularly|in particular|which is|'
    r'a |an |the |one |no |all |each |every |this |these |those |'
    r'PSNR|SSIM|LPIPS|CLIP|count|video|image|'
    r'[A-Z][A-Z])',  # abbreviations / acronyms
    re.IGNORECASE
)


def has_paired_emdash(text, pos):
    """Check if the em dash at `pos` is part of a pair in the same sentence."""
    # Find sentence boundaries (rough: look for . ? ! or start/end of text)
    # Search backwards for sentence start
    sent_start = 0
    for i in range(pos - 1, -1, -1):
        if text[i] in '.?!\n' and i < pos - 1:
            sent_start = i + 1
            break
    
    # Search forward for sentence end
    sent_end = len(text)
    for i in range(pos + 1, len(text)):
        if text[i] in '.?!\n':
            sent_end = i
            break
    
    sentence = text[sent_start:sent_end]
    count = sentence.count(' — ')
    return count >= 2


def replace_emdashes_in_text(text):
    """Replace all em dashes contextually."""
    # Strategy: process paired em dashes first, then remaining singles
    
    # Pass 1: Find and replace PAIRED em dashes with commas
    # We need to find sentences with exactly 2 em dashes and replace both with commas
    result = []
    i = 0
    
    # Split into lines for processing but keep track of full text for pair detection
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        if ' — ' not in line:
            new_lines.append(line)
            continue
        
        occurrences = [m.start() for m in re.finditer(r' — ', line)]
        
        if len(occurrences) >= 2:
            # Check for pairs: consecutive em dashes that form parentheticals
            # Process from right to left to maintain positions
            pairs = []
            used = set()
            
            for idx in range(0, len(occurrences) - 1):
                if idx in used:
                    continue
                pos1 = occurrences[idx]
                pos2 = occurrences[idx + 1]
                
                # Check if they're in the same sentence-like segment
                between = line[pos1:pos2]
                # If no sentence-ending punctuation between them, they're a pair
                if not any(c in between for c in '.?!'):
                    pairs.append((pos1, pos2))
                    used.add(idx)
                    used.add(idx + 1)
            
            # Replace pairs with commas (process right to left)
            new_line = line
            for pos1, pos2 in reversed(pairs):
                new_line = new_line[:pos2] + ', ' + new_line[pos2 + 3:]  # ' — ' is 3 chars
                new_line = new_line[:pos1] + ', ' + new_line[pos1 + 3:]
            
            # Now handle remaining singles
            remaining = [m.start() for m in re.finditer(r' — ', new_line)]
            for pos in reversed(remaining):
                after = new_line[pos + 3:pos + 40].strip()
                if COLON_TRIGGERS.match(after):
                    new_line = new_line[:pos] + ': ' + new_line[pos + 3:]
                else:
                    new_line = new_line[:pos] + '; ' + new_line[pos + 3:]
            
            new_lines.append(new_line)
        
        elif len(occurrences) == 1:
            pos = occurrences[0]
            after = line[pos + 3:pos + 40].strip()
            if COLON_TRIGGERS.match(after):
                new_line = line[:pos] + ': ' + line[pos + 3:]
            else:
                new_line = line[:pos] + '; ' + line[pos + 3:]
            new_lines.append(new_line)
        
        else:
            new_lines.append(line)
    
    return '\n'.join(new_lines)


def process_file(filepath):
    """Process a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '—' not in content:
        return False
    
    new_content = replace_emdashes_in_text(content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def main():
    files = glob.glob(os.path.join(THESIS_DIR, '*_v2.txt'))
    files.sort()
    
    total_before = 0
    total_after = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        if filename in SKIP_FILES:
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        count_before = content.count('—')
        total_before += count_before
        
        if count_before == 0:
            print(f"  SKIP: {filename} (no em dashes)")
            continue
        
        changed = process_file(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        count_after = content.count('—')
        total_after += count_after
        
        if changed:
            print(f"  {filename}: {count_before} em dashes → {count_after} remaining")
        else:
            print(f"  {filename}: no changes needed")
    
    print(f"\n  Total: {total_before} → {total_after} em dashes")


if __name__ == '__main__':
    main()

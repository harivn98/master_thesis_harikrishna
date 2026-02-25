#!/usr/bin/env python3
"""
Fix pass: The em dash replacement script incorrectly used colons for some
cases where commas or semicolons are needed. This script:

1. Finds ": X :" patterns (paired em dashes that got colons) → ", X ,"
2. Fixes specific bad colon usages that should be semicolons
"""
import os
import re
import glob

THESIS_DIR = os.path.dirname(os.path.abspath(__file__))

SKIP_FILES = {
    "replace_emdashes.py",
    "fix_emdash_colons.py",
    "reformat_for_word.py",
    "BST_Thesis_Template (2).txt",
}

# Known bad patterns: ": word" where original was " — word" (paired or single)
# These should NOT be colons. They should be commas (if paired) or semicolons (if single).

def fix_paired_colons(text):
    """
    Fix paired colons that should be commas.
    Pattern: "word: phrase :" → "word, phrase,"
    The tricky part is identifying which colons came from em dashes vs original colons.
    
    Strategy: Look for colon-pairs where neither colon introduces a list/definition.
    We'll use a targeted approach - fix known bad patterns.
    """
    
    # Pattern: text": including X:" → ", including X,"
    # Pattern: text": PSNR, SSIM... :" → ", PSNR, SSIM... ,"
    # Pattern: text": exactly X" → ", exactly X"  (single, should be comma)
    # Pattern: text": and yet," → ", and yet,"
    
    # Fix paired colons pattern: colon ... colon within same sentence
    # We'll find sentences with 2+ colons and check if they look like em dash replacements
    
    fixes = [
        # Foreword fixes
        (': and yet,', ', and yet,'),
        
        # Fix ": questions about" → "; questions about" (single em dash, clause connector)  
        (': questions about', '; questions about'),
    ]
    
    for old, new in fixes:
        text = text.replace(old, new)
    
    return text


def fix_file(filepath):
    """Fix colon issues in a single file by finding paired-colon patterns."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Apply simple targeted fixes first
    content = fix_paired_colons(content)
    
    # Now find and fix paired colons that came from paired em dashes
    # These appear as ": phrase :" within a sentence (no period between)
    # Strategy: find all colons, check if pairs exist within a sentence
    
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Find all colon positions
        colon_positions = [i for i, c in enumerate(line) if c == ':']
        
        if len(colon_positions) < 2:
            new_lines.append(line)
            continue
        
        # Check for paired colons that look like em dash replacements
        # A pair: "word: phrase :" where phrase doesn't contain periods
        # And neither is at the start of a known pattern like "Chapter 3:"
        
        # Convert to list for character-level editing
        chars = list(line)
        fixed_positions = set()
        
        for i in range(len(colon_positions) - 1):
            pos1 = colon_positions[i]
            pos2 = colon_positions[i + 1]
            
            # Skip if already fixed
            if pos1 in fixed_positions or pos2 in fixed_positions:
                continue
            
            between = line[pos1+1:pos2]
            
            # Skip if there's a period or newline between (different sentences)
            if '.' in between or '?' in between or '!' in between:
                continue
            
            # Skip if the distance is too large (probably not a pair)
            if pos2 - pos1 > 200:
                continue
            
            # Check context before pos1: should be a word (not a label/key)
            before = line[:pos1].rstrip()
            after_pos2 = line[pos2+1:].lstrip() if pos2 + 1 < len(line) else ''
            
            # Skip legitimate colons:
            # - After labels like "Scope:", "Limitations:", "RQ1:", "Goal 1:", etc.
            if re.search(r'(Scope|Limitations?|RQ\d|Goal \d|Chapter \d|Table \d|Stage|Note|Source file|Figure)\s*$', before):
                continue
            
            # Skip if it's "key: value" pattern (short before-text ending in a label)
            if re.search(r'^\s*\w+\s*$', line[:pos1].split(',')[-1].strip()):
                # Could be a label, skip
                pass
            
            # Check if the first colon is followed by a space (em dash replacement adds space)
            if pos1 + 1 < len(line) and line[pos1 + 1] == ' ':
                # And the second colon is preceded by a space or word
                if pos2 > 0 and (line[pos2 - 1] == ' ' or line[pos2 - 1].isalpha() or line[pos2-1] == ')'):
                    # This looks like a paired em dash replacement
                    # But verify: the text after pos2 should continue the sentence
                    if after_pos2 and (after_pos2[0].islower() or after_pos2[0] in '("'):
                        # Likely a pair - replace both with commas
                        chars[pos1] = ','
                        chars[pos2] = ','
                        fixed_positions.add(pos1)
                        fixed_positions.add(pos2)
        
        new_lines.append(''.join(chars))
    
    content = '\n'.join(new_lines)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    files = glob.glob(os.path.join(THESIS_DIR, '*_v2.txt'))
    files.sort()
    
    for filepath in files:
        filename = os.path.basename(filepath)
        if filename in SKIP_FILES:
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        changed = fix_file(filepath)
        
        if changed:
            print(f"  FIXED: {filename}")
        else:
            print(f"  OK: {filename}")


if __name__ == '__main__':
    main()

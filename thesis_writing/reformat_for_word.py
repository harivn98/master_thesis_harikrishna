#!/usr/bin/env python3
"""
Reformat thesis text files so that each paragraph is a single continuous line
(like abstract_v2.txt), which pastes cleanly into Word without unwanted line breaks.

Rules:
- Blank lines are preserved as paragraph separators
- Blocks where ALL lines are indented (start with whitespace) are kept as-is (tables, code, prompts)
- Separator lines (=== or ---) are kept as-is
- Regular paragraph text with hard line wraps is joined into single lines
- Reference entries (non-indented first line + indented continuations) are joined into single lines
"""

import os
import re
import glob

THESIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to skip (already formatted correctly or not thesis content)
SKIP_FILES = {
    "abstract_v2.txt",        # Already in correct format
    "acknowledgements_v2.txt", # Already in correct format
    "BST_Thesis_Template (2).txt",  # Template, not thesis content
    "reformat_for_word.py",   # This script
}


def is_separator_line(line):
    """Check if a line is a visual separator (=== or ---)."""
    stripped = line.strip()
    return (
        len(stripped) >= 3 and
        (all(c == '=' for c in stripped) or all(c == '-' for c in stripped))
    )


def is_section_separator(line):
    """Check if a line is like '--- JOURNAL ARTICLES (1) ---'."""
    stripped = line.strip()
    return stripped.startswith('---') and stripped.endswith('---') and len(stripped) > 6


def is_all_indented(lines):
    """Check if all lines in a block start with whitespace."""
    return all(line[0] in (' ', '\t') for line in lines if line.strip())


def should_keep_block_as_is(lines):
    """Determine if a block should be kept without joining."""
    if not lines:
        return True
    # If any line is a separator
    if any(is_separator_line(l) or is_section_separator(l) for l in lines):
        return True
    # If all lines are indented
    if is_all_indented(lines):
        return True
    return False


def join_block(lines):
    """Join a block of lines into a single line, handling indented continuations."""
    if not lines:
        return ""
    # Join all lines, stripping leading whitespace from continuation lines
    parts = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            parts.append(stripped)
    return ' '.join(parts)


def process_file(filepath):
    """Process a single file, joining hard-wrapped paragraphs."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Split into blocks separated by blank lines
    blocks = []
    current_block = []
    for line in lines:
        if line.strip() == '':
            if current_block:
                blocks.append(('text', current_block))
                current_block = []
            blocks.append(('blank', [line]))
        else:
            current_block.append(line)
    if current_block:
        blocks.append(('text', current_block))

    # Process each block
    output_lines = []
    for block_type, block_lines in blocks:
        if block_type == 'blank':
            output_lines.append('')
        else:
            if should_keep_block_as_is(block_lines):
                # Keep all lines as-is
                for line in block_lines:
                    output_lines.append(line)
            else:
                # Join into single line
                joined = join_block(block_lines)
                output_lines.append(joined)

    # Remove trailing blank lines, then add one final newline
    while output_lines and output_lines[-1] == '':
        output_lines.pop()
    output_lines.append('')  # Single trailing newline

    return '\n'.join(output_lines)


def main():
    files = glob.glob(os.path.join(THESIS_DIR, '*_v2.txt'))
    files.sort()

    for filepath in files:
        filename = os.path.basename(filepath)
        if filename in SKIP_FILES:
            print(f"  SKIP: {filename} (already formatted or excluded)")
            continue

        print(f"  Processing: {filename}")

        # Read original for comparison
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        reformatted = process_file(filepath)

        if original == reformatted:
            print(f"    -> No changes needed")
        else:
            orig_lines = len(original.split('\n'))
            new_lines = len(reformatted.split('\n'))
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(reformatted)
            print(f"    -> Reformatted: {orig_lines} lines -> {new_lines} lines")


if __name__ == '__main__':
    main()

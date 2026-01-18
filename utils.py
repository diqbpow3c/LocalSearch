from html import escape
import re

def highlight_keywords_in_text(text, highlight_words, query, max_length=200):
    # Handle None or empty highlight_words
    if not highlight_words:
        highlight_words = []
    # find the first occurrence of the highlight_words
    perfect_hit_position = text.find(query)  # -1 if not found
    positions_for_words = [
        text.find(word) for word in highlight_words if text.find(word) != -1
    ]
    if perfect_hit_position != -1:
        min_pos = perfect_hit_position
    else:
        min_pos = min(positions_for_words) if positions_for_words else -1
    min_pos = 0 if min_pos - 10 < 0 else min_pos - 10
    truncated_text = text[min_pos: min_pos + max_length] + "..."
    # Escape HTML special characters
    escaped_text = escape(truncated_text)

    if perfect_hit_position != -1:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        # escaped_text = pattern.sub(lambda m: f"<b>{m.group(0)}</b>", escaped_text)
        escaped_text = pattern.sub(
            lambda m: f"<span style='color:red;'>{m.group(0)}</span>", escaped_text
        )
    else:
        # Highlight specified words in red
        for word in highlight_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            # escaped_text = pattern.sub(lambda m: f"<b>{m.group(0)}</b>", escaped_text)
            escaped_text = pattern.sub(
                lambda m: f"<span style='color:red;'>{m.group(0)}</span>", escaped_text
            )

    return escaped_text
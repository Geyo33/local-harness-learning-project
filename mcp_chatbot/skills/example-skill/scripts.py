"""Example skill scripts — demonstrates the DISPATCH convention."""


def word_count(text: str) -> str:
    """Count the number of words and characters in the given text."""
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


def reverse_text(text: str) -> str:
    """Reverse the given text string."""
    return text[::-1]


DISPATCH = {
    "word_count": word_count,
    "reverse_text": reverse_text,
}

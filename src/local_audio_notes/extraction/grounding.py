"""Deterministic support checks of model output against transcript text.

These checks reject unsupported quotes, owners, deadlines, names, numbers and dates.
They cannot prove that a paraphrase or a category assignment is semantically correct.
"""

import re

_WORD = re.compile(r"[^\W_]+")
# A quote, owner or deadline phrase that never names anyone.
PRONOUNS = frozenset(
    [
        "i",
        "me",
        "we",
        "us",
        "you",
        "they",
        "them",
        "he",
        "him",
        "she",
        "her",
        "it",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "anyone",
        "anybody",
        "nobody",
    ]
)


def normalize(text: str) -> str:
    return " ".join(text.split())


def _key(char: str) -> str:
    lowered = char.lower()
    return lowered if len(lowered) == 1 else char


def locate(needle: str, haystack: str) -> str | None:
    """Return haystack's own text matching needle, ignoring case and whitespace runs.

    The result is always a verbatim slice of the transcript, never the model's spelling.
    """
    target = "".join(_key(c) for c in normalize(needle))
    if not target:
        return None
    folded: list[str] = []
    index: list[int] = []
    for position, char in enumerate(haystack):
        if char.isspace():
            if folded and folded[-1] != " ":
                folded.append(" ")
                index.append(position)
        else:
            folded.append(_key(char))
            index.append(position)
    found = "".join(folded).find(target)
    if found < 0:
        return None
    return haystack[index[found] : index[found + len(target) - 1] + 1]


_SENTENCE_START = re.compile(r"(?:^|[.!?:;]\s+)([^\W_]+)")
# Function words that are capitalized only because they start a sentence. Never names,
# numbers or dates, so exempting them does not let an invented entity through.
STARTERS = frozenset(
    "a an the this that these those there here it its we our us you your they their "  # noqa: SIM905
    "is are was were be been do does did will would shall should can could may might must "
    "has have had what when where which who why how if so but and or not no yes "
    "all any each every some also then next still please let".split()
)


def salient(text: str) -> set[str]:
    """Capitalized words and anything containing a digit: names, dates, numbers, products."""
    initial = {m.start(1) for m in _SENTENCE_START.finditer(text)}
    found = set()
    for match in _WORD.finditer(text):
        word = match.group()
        sentence_word = match.start() in initial and word.lower() in STARTERS
        if any(c.isdigit() for c in word) or (
            word[0].isupper() and word != "I" and not sentence_word
        ):
            found.add(word)
    return found


def grounded(text: str, support: str) -> bool:
    available = {w.lower() for w in _WORD.findall(support)}
    return all(w.lower() in available for w in salient(text))


def words(text: str) -> list[str]:
    return _WORD.findall(text)

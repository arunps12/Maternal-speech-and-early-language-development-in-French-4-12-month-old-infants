from typing import Optional


# ---------------------------------------------------------------------------
# Exact label corrections
# Applied as full-string replacements before any parsing takes place.
# ---------------------------------------------------------------------------
LABEL_CORRECTIONS: dict[str, str] = {
    "eu_petit_IS":            "eu_petit_IDS",
    "ou_loulou_1-IDS":        "ou_loulou_1_IDS",
    "oe_petit_IDoe_petitS":   "oe_petit_IDS",
    "papa_2_IDS":             "a_papa_2_IDS",
    "tot_IDS":                "o_tot_IDS",
    "_sais_ADS":              "ai_sais_ADS",
    "on _on_ADS":             "on_on_ADS",
    "ei_petit_IDS":           "eu_petit_IDS",
    "la_a_ADS":               "a_la_ADS",
    "y_tu":                   "u_tu",
    "a_voila__2IDS":          "a_voila_2_IDS",
    "u_entendu_aDS":          "u_entendu_ADS",
}

# Register variants recognised as the final field of a label (before cleaning)
_KNOWN_REGISTER_VARIANTS: frozenset[str] = frozenset({
    "IDS", "ADS", "IDS(chant)",  # canonical forms
    "IDs", "IDA", "ID", "aDS",   # known typos
})


def apply_label_corrections(label: str) -> str:
    """Apply exact-string corrections to a raw label.

    Only entries listed in LABEL_CORRECTIONS are replaced; all other labels
    are returned unchanged.
    """
    return LABEL_CORRECTIONS.get(label, label)


def clean_register(register: Optional[str]) -> str:
    """Normalise a register token to a canonical value.

    Mappings:
        IDs / IDA / ID  ->  IDS
        aDS             ->  ADS
        empty / None    ->  IDS          (missing register defaults to IDS)
        IDS(chant)      ->  IDS(chant)   (kept for downstream removal)
        IDS / ADS       ->  unchanged
    """
    if not register:
        return "IDS"
    _mapping = {
        "IDs": "IDS",
        "IDA": "IDS",
        "ID":  "IDS",
        "aDS": "ADS",
    }
    return _mapping.get(register, register)


def parse_label(label: str) -> Optional[tuple[str, str, str]]:
    """Parse a (corrected) label into ``(vowel, word, register)``.

    Label format:  vowel_word_register

    Examples::

        o_so_IDS       ->  ('o',  'so',      'IDS')
        a_papa_2_IDS   ->  ('a',  'papa_2',  'IDS')
        ai_sais_ADS    ->  ('ai', 'sais',    'ADS')
        u_tu           ->  ('u',  'tu',      'IDS')   # no register -> IDS

    Rules:
    - vowel  = first ``_``-separated token
    - register = last token **if** it matches a known register variant
      (after :func:`clean_register` normalisation); otherwise the entire
      tail is the word and register defaults to ``IDS``.
    - word = all middle tokens joined with ``_``
    - Vowel ``en`` is **never** converted to ``an``.
    - Returns ``None`` when the label has fewer than 2 tokens or the word
      portion would be empty.
    """
    parts = [p.strip() for p in label.split("_")]
    # Remove empty tokens produced by leading/trailing/double underscores
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None

    vowel = parts[0]

    if len(parts) == 2:
        # Only vowel + one more token: no explicit register
        word = parts[1]
        register = "IDS"
    elif parts[-1] in _KNOWN_REGISTER_VARIANTS:
        register = clean_register(parts[-1])
        word = "_".join(parts[1:-1])
    else:
        # Last token is not a register — treat entire tail as word
        word = "_".join(parts[1:])
        register = "IDS"

    if not word:
        return None

    return vowel, word, register

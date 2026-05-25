"""Tests for src/french_ids/label_cleaning.py"""
import pytest

from french_ids.label_cleaning import (
    LABEL_CORRECTIONS,
    apply_label_corrections,
    clean_register,
    parse_label,
)


# ---------------------------------------------------------------------------
# Label corrections
# ---------------------------------------------------------------------------

class TestApplyLabelCorrections:
    """Every entry in LABEL_CORRECTIONS must be applied exactly."""

    @pytest.mark.parametrize("raw,expected", list(LABEL_CORRECTIONS.items()))
    def test_each_correction(self, raw: str, expected: str):
        assert apply_label_corrections(raw) == expected

    def test_uncorrected_label_is_unchanged(self):
        assert apply_label_corrections("o_so_IDS")    == "o_so_IDS"
        assert apply_label_corrections("en_bien_IDS") == "en_bien_IDS"
        assert apply_label_corrections("a_papa_2_IDS") == "a_papa_2_IDS"

    def test_empty_string_is_unchanged(self):
        assert apply_label_corrections("") == ""


# ---------------------------------------------------------------------------
# Register cleaning
# ---------------------------------------------------------------------------

class TestCleanRegister:
    def test_ids_typos_become_ids(self):
        assert clean_register("IDs") == "IDS"
        assert clean_register("IDA") == "IDS"
        assert clean_register("ID")  == "IDS"

    def test_ads_typo_becomes_ads(self):
        assert clean_register("aDS") == "ADS"

    def test_canonical_ids_unchanged(self):
        assert clean_register("IDS") == "IDS"

    def test_canonical_ads_unchanged(self):
        assert clean_register("ADS") == "ADS"

    def test_ids_chant_unchanged(self):
        assert clean_register("IDS(chant)") == "IDS(chant)"

    def test_empty_string_becomes_ids(self):
        assert clean_register("") == "IDS"

    def test_none_becomes_ids(self):
        assert clean_register(None) == "IDS"


# ---------------------------------------------------------------------------
# Label parsing — basic cases
# ---------------------------------------------------------------------------

class TestParseLabel:
    def test_simple_vowel_word_register(self):
        assert parse_label("o_so_IDS") == ("o", "so", "IDS")

    def test_multipart_word(self):
        assert parse_label("a_papa_2_IDS") == ("a", "papa_2", "IDS")

    def test_ads_register(self):
        assert parse_label("ai_sais_ADS") == ("ai", "sais", "ADS")

    def test_missing_register_defaults_to_ids(self):
        assert parse_label("u_tu") == ("u", "tu", "IDS")

    def test_three_part_word_with_register(self):
        result = parse_label("a_voila_2_IDS")
        assert result == ("a", "voila_2", "IDS")


# ---------------------------------------------------------------------------
# Vowel 'en' preservation
# ---------------------------------------------------------------------------

class TestEnVowelPreserved:
    """Vowel 'en' must never be silently converted to 'an'."""

    def test_en_is_kept_as_en(self):
        result = parse_label("en_bien_IDS")
        assert result is not None
        vowel, word, register = result
        assert vowel == "en"
        assert word  == "bien"
        assert register == "IDS"

    def test_en_ads(self):
        result = parse_label("en_vent_ADS")
        assert result is not None
        assert result[0] == "en"


# ---------------------------------------------------------------------------
# Register normalisation inside parse_label
# ---------------------------------------------------------------------------

class TestRegisterNormalisationInParseLabel:
    def test_ids_typo_normalised(self):
        result = parse_label("o_so_IDs")
        assert result is not None
        assert result[2] == "IDS"

    def test_ida_normalised(self):
        result = parse_label("a_mama_IDA")
        assert result is not None
        assert result[2] == "IDS"

    def test_ads_typo_normalised(self):
        result = parse_label("a_papa_aDS")
        assert result is not None
        assert result[2] == "ADS"


# ---------------------------------------------------------------------------
# IDS(chant) — detection for downstream filtering
# ---------------------------------------------------------------------------

class TestIDSChant:
    def test_ids_chant_parsed_correctly(self):
        result = parse_label("a_mama_IDS(chant)")
        assert result is not None
        vowel, word, register = result
        assert vowel    == "a"
        assert word     == "mama"
        assert register == "IDS(chant)"

    def test_ids_chant_appears_in_remove_set(self):
        remove_registers = {"IDS(chant)"}
        result = parse_label("a_mama_IDS(chant)")
        assert result is not None
        assert result[2] in remove_registers


# ---------------------------------------------------------------------------
# Invalid / unparseable labels
# ---------------------------------------------------------------------------

class TestInvalidLabels:
    def test_single_token_returns_none(self):
        assert parse_label("IDS") is None
        assert parse_label("a")   is None

    def test_empty_string_returns_none(self):
        assert parse_label("") is None

    def test_label_with_only_underscores_returns_none(self):
        # After stripping empty tokens, nothing survives
        assert parse_label("___") is None

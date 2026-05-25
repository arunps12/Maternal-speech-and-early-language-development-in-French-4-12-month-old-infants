"""Tests for src/french_ids/filename_parser.py"""
import pytest

from french_ids.filename_parser import parse_filename


class TestParseFilenameBasic:
    def test_standard_four_part_filename(self):
        result = parse_filename("c012_8m_bath_1925")
        assert result == {
            "speakerid": "c012",
            "session":   "8m",
            "activity":  "bath",
            "time":      "1925",
        }

    def test_speakerid_is_kept_as_found_lowercase(self):
        assert parse_filename("c028_4m_book_0930")["speakerid"] == "c028"

    def test_speakerid_is_kept_as_found_uppercase(self):
        # speakerid must NOT be normalised — keep exactly as in filename
        assert parse_filename("C012_8m_bath_1925")["speakerid"] == "C012"

    def test_different_sessions(self):
        assert parse_filename("c028_4m_book_0930")["session"] == "4m"
        assert parse_filename("c028_12m_toy_1400")["session"] == "12m"

    def test_different_activities(self):
        assert parse_filename("c028_4m_bath_0800")["activity"] == "bath"
        assert parse_filename("c028_4m_book_0800")["activity"] == "book"

    def test_time_is_last_token(self):
        assert parse_filename("c012_8m_bath_1925")["time"] == "1925"


class TestParseFilenameActivityWithUnderscores:
    """Filenames with 5+ parts: activity spans all middle tokens."""

    def test_two_word_activity(self):
        result = parse_filename("c012_8m_free_play_1925")
        assert result is not None
        assert result["speakerid"] == "c012"
        assert result["session"]   == "8m"
        assert result["activity"]  == "free_play"
        assert result["time"]      == "1925"

    def test_three_word_activity(self):
        result = parse_filename("c012_8m_free_play_time_1925")
        assert result is not None
        assert result["activity"] == "free_play_time"
        assert result["time"]     == "1925"


class TestParseFilenameInvalid:
    def test_three_parts_returns_none(self):
        assert parse_filename("c012_8m_bath") is None

    def test_two_parts_returns_none(self):
        assert parse_filename("c012_8m") is None

    def test_one_part_returns_none(self):
        assert parse_filename("c012") is None

    def test_empty_string_returns_none(self):
        assert parse_filename("") is None

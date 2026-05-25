from typing import Optional


def parse_filename(stem: str) -> Optional[dict]:
    """Parse a TextGrid filename stem into metadata components.

    Expected format: speakerid_session_activity_time

    Example:
        c012_8m_bath_1925  ->  {speakerid: "c012", session: "8m",
                                 activity: "bath",  time: "1925"}

    When the activity itself contains underscores (5+ part filenames), the
    middle tokens are joined back with "_":
        c012_8m_free_play_1925  ->  activity = "free_play"

    Returns None if the stem has fewer than 4 underscore-separated parts.
    speakerid is kept exactly as found in the filename (no case normalisation).
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    return {
        "speakerid": parts[0],
        "session": parts[1],
        "activity": "_".join(parts[2:-1]),
        "time": parts[-1],
    }

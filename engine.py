"""Audio fingerprinting engine: decode -> spectrogram -> peaks -> hashes -> DB."""
from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import librosa
import numpy as np
from mysql.connector import IntegrityError
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram

from db import cursor, get_connection

# ---------------------------------------------------------------- parameters
SAMPLE_RATE = 22050
NPERSEG = 2048
NOVERLAP = 1536
HOP = NPERSEG - NOVERLAP                    # 512 samples
HOP_MS = HOP * 1000.0 / SAMPLE_RATE         # ~23.2 ms per frame
PEAK_NEIGHBORHOOD = (20, 20)                # (freq bins, time frames)
PEAK_FLOOR_DB = 60                          # ignore peaks > 60 dB below loudest
FAN_OUT = 8                                 # pairs per anchor peak
MIN_DT, MAX_DT = 1, 100                     # target zone in frames (~2.3 s)
FREQ_SHIFT = 2                              # 1025 bins >> 2 -> fits in 8 bits
MIN_MATCHES = int(os.getenv("MIN_MATCHES", "10"))  # aligned hashes needed
MIN_QUERY_HASHES = 15
MIN_DURATION_S = 2.0
MAX_QUERY_S = 30.0
SILENCE_RMS = 1e-4
INSERT_BATCH = 5000
QUERY_BATCH = 2000


# ------------------------------------------------------------------- errors
class FingerprintError(Exception):
    """Base class for user-facing engine errors."""


class AudioDecodeError(FingerprintError): ...
class AudioTooShortError(FingerprintError): ...
class SilentAudioError(FingerprintError): ...
class NoFingerprintsError(FingerprintError): ...
class DuplicateSongError(FingerprintError): ...


class NoMatchError(FingerprintError):
    def __init__(self, best_count: int = 0):
        super().__init__("No confident match found.")
        self.best_count = best_count


@dataclass
class MatchResult:
    song_id: int
    title: str
    artist: str
    score: int              # aligned hash collisions for the winning offset
    confidence: float       # 0..1 heuristic
    query_hashes: int
    clip_start_s: float     # where the clip begins inside the matched song
    runner_up_score: int


# ------------------------------------------------------------ audio -> hashes
def load_audio(data: bytes, suffix: str = ".wav", max_seconds: Optional[float] = None):
    """Decode bytes to a mono 22.05 kHz float array (librosa/soundfile)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    except Exception as exc:
        raise AudioDecodeError(
            "Could not decode this audio file. Use a valid MP3/WAV "
            "(MP3 decoding may require ffmpeg on older libsndfile builds)."
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return y


def _validate(y: np.ndarray) -> None:
    if y.size / SAMPLE_RATE < MIN_DURATION_S:
        raise AudioTooShortError(f"Audio is too short (need at least {MIN_DURATION_S:.0f}s).")
    if float(np.sqrt(np.mean(y ** 2))) < SILENCE_RMS:
        raise SilentAudioError("Audio is silent or almost silent.")


def compute_spectrogram(y: np.ndarray) -> np.ndarray:
    """STFT magnitude in dB, shape (freq_bins, frames)."""
    _, _, sxx = spectrogram(y, fs=SAMPLE_RATE, window="hann",
                            nperseg=NPERSEG, noverlap=NOVERLAP)
    return 10.0 * np.log10(sxx + 1e-10)


def find_peaks(s_db: np.ndarray):
    """Local maxima via maximum_filter; returns (freq_idx, time_idx) sorted by time."""
    local_max = maximum_filter(s_db, size=PEAK_NEIGHBORHOOD, mode="nearest")
    mask = (s_db == local_max) & (s_db > s_db.max() - PEAK_FLOOR_DB)
    f_idx, t_idx = np.nonzero(mask)
    order = np.lexsort((f_idx, t_idx))
    return f_idx[order], t_idx[order]


def generate_hashes(f_idx: np.ndarray, t_idx: np.ndarray) -> list[tuple[int, int]]:
    """Pair each anchor peak with up to FAN_OUT following peaks.

    hash = (f1 & 0xFF) | ((f2 & 0xFF) << 8) | ((delta_t & 0xFFFF) << 16)
    Frequencies are quantised to 8 bits (bin >> 2, ~43 Hz); delta_t is in frames.
    """
    f = [min(int(x) >> FREQ_SHIFT, 255) for x in f_idx]
    t = [int(x) for x in t_idx]
    n = len(t)
    out: set[tuple[int, int]] = set()
    for i in range(n):
        paired = 0
        for j in range(i + 1, n):
            dt = t[j] - t[i]
            if dt > MAX_DT:
                break
            if dt < MIN_DT:
                continue
            h = (f[i] & 0xFF) | ((f[j] & 0xFF) << 8) | ((dt & 0xFFFF) << 16)
            out.add((h, int(round(t[i] * HOP_MS))))
            paired += 1
            if paired >= FAN_OUT:
                break
    return sorted(out, key=lambda x: x[1])


def fingerprint_audio(y: np.ndarray) -> list[tuple[int, int]]:
    _validate(y)
    f_idx, t_idx = find_peaks(compute_spectrogram(y))
    return generate_hashes(f_idx, t_idx)


# ------------------------------------------------------------- registration
def register_song(title: str, artist: str, data: bytes, suffix: str = ".mp3",
                  progress: Optional[Callable[[float, str], None]] = None) -> tuple[int, int]:
    """Fingerprint and store a song. Returns (song_id, n_fingerprints)."""
    report = progress or (lambda p, m: None)
    file_hash = hashlib.sha256(data).hexdigest()

    with cursor() as cur:
        cur.execute("SELECT id FROM songs WHERE file_hash = %s", (file_hash,))
        if cur.fetchone():
            raise DuplicateSongError("This exact file is already in the library.")

    report(0.05, "Decoding audio...")
    y = load_audio(data, suffix)
    report(0.25, "Computing spectrogram, peaks and hashes...")
    fps = fingerprint_audio(y)
    if not fps:
        raise NoFingerprintsError("No usable fingerprints could be extracted from this file.")

    conn = get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO songs (title, artist, file_hash) VALUES (%s, %s, %s)",
                        (title.strip(), artist.strip(), file_hash))
        except IntegrityError as exc:
            raise DuplicateSongError("This exact file is already in the library.") from exc
        song_id = cur.lastrowid
        rows = [(h, song_id, off) for h, off in fps]
        for i in range(0, len(rows), INSERT_BATCH):
            cur.executemany(
                "INSERT INTO fingerprints (hash_val, song_id, offset_ms) VALUES (%s, %s, %s)",
                rows[i:i + INSERT_BATCH])
            done = min(i + INSERT_BATCH, len(rows))
            report(0.4 + 0.6 * done / len(rows), f"Inserted {done:,}/{len(rows):,} fingerprints")
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()  # returns connection to the pool
    report(1.0, "Done")
    return song_id, len(fps)


# ----------------------------------------------------------- identification
def identify(data: bytes, suffix: str = ".wav") -> MatchResult:
    """Match a clip against the library by offset-aligned hash voting."""
    y = load_audio(data, suffix, max_seconds=MAX_QUERY_S)
    sample = fingerprint_audio(y)
    if len(sample) < MIN_QUERY_HASHES:
        raise NoFingerprintsError("Not enough distinctive audio features. Try a louder, longer clip.")

    by_hash: dict[int, list[int]] = defaultdict(list)
    for h, off in sample:
        by_hash[h].append(off)

    votes: Counter = Counter()  # (song_id, delta_in_frames) -> collisions
    hashes = list(by_hash)
    with cursor() as cur:
        for i in range(0, len(hashes), QUERY_BATCH):
            chunk = hashes[i:i + QUERY_BATCH]
            ph = ",".join(["%s"] * len(chunk))
            cur.execute(f"SELECT hash_val, song_id, offset_ms FROM fingerprints "
                        f"WHERE hash_val IN ({ph})", chunk)
            for h, song_id, db_off in cur.fetchall():
                for s_off in by_hash[h]:
                    # delta_offset = db_offset - sample_offset (binned to frames to absorb rounding)
                    votes[(song_id, int(round((db_off - s_off) / HOP_MS)))] += 1

        if not votes:
            raise NoMatchError(0)

        best_per_song: dict[int, tuple[int, int]] = {}
        for (sid, delta), cnt in votes.items():
            if sid not in best_per_song or cnt > best_per_song[sid][0]:
                best_per_song[sid] = (cnt, delta)
        ranked = sorted(best_per_song.items(), key=lambda kv: kv[1][0], reverse=True)
        top_sid, (top_cnt, top_delta) = ranked[0]
        runner_up = ranked[1][1][0] if len(ranked) > 1 else 0

        if top_cnt < MIN_MATCHES:
            raise NoMatchError(top_cnt)

        cur.execute("SELECT title, artist FROM songs WHERE id = %s", (top_sid,))
        row = cur.fetchone()
    if row is None:
        raise NoMatchError(top_cnt)

    confidence = min(1.0, top_cnt / (MIN_MATCHES * 4))
    if runner_up and runner_up * 2 > top_cnt:        # ambiguous: penalise
        confidence *= 0.5
    return MatchResult(top_sid, row[0], row[1], top_cnt, confidence, len(sample),
                       max(0.0, top_delta * HOP_MS / 1000.0), runner_up)


# --------------------------------------------------------------- library ops
def list_songs() -> list[dict]:
    with cursor(dictionary=True) as cur:
        cur.execute("SELECT s.id, s.title, s.artist, COUNT(f.song_id) AS fingerprints "
                    "FROM songs s LEFT JOIN fingerprints f ON f.song_id = s.id "
                    "GROUP BY s.id ORDER BY s.id DESC")
        return cur.fetchall()


def delete_song(song_id: int) -> None:
    with cursor(commit=True) as cur:
        cur.execute("DELETE FROM songs WHERE id = %s", (song_id,))  # fingerprints cascade

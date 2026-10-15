from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import engine
from db import DatabaseError, check_connection

st.set_page_config(page_title="Audio Fingerprinting", page_icon="🎵", layout="centered")
st.title("🎵 Audio Fingerprinting")

try:
    check_connection()
except DatabaseError as exc:
    st.error(str(exc))
    st.stop()

tab_lib, tab_id = st.tabs(["Song Library", "Identify"])

# ------------------------------------------------------------- Song Library
with tab_lib:
    st.subheader("Add a song")
    up = st.file_uploader("Upload MP3 or WAV", type=["mp3", "wav"], key="lib_upload")
    name_key = up.name if up else "none"
    c1, c2 = st.columns(2)
    title = c1.text_input("Title", value=Path(up.name).stem if up else "", key=f"title_{name_key}")
    artist = c2.text_input("Artist", key=f"artist_{name_key}")

    if st.button("Fingerprint & add", type="primary", disabled=up is None):
        if not title.strip() or not artist.strip():
            st.warning("Please enter both title and artist.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                _, n = engine.register_song(
                    title, artist, up.getvalue(), Path(up.name).suffix,
                    progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
                st.success(f"Added “{title}” by {artist} ({n:,} fingerprints).")
            except engine.DuplicateSongError as exc:
                bar.empty(); st.info(str(exc))
            except engine.FingerprintError as exc:
                bar.empty(); st.error(str(exc))
            except DatabaseError as exc:
                bar.empty(); st.error(str(exc))
            except Exception as exc:  # unexpected
                bar.empty(); st.error(f"Unexpected error: {exc}")

    st.divider()
    st.subheader("Library")
    try:
        songs = engine.list_songs()
    except DatabaseError as exc:
        songs = []
        st.error(str(exc))
    if songs:
        st.dataframe(songs, hide_index=True, use_container_width=True)
        labels = {f"{s['title']} — {s['artist']} (#{s['id']})": s["id"] for s in songs}
        d1, d2 = st.columns([3, 1])
        pick = d1.selectbox("Remove a song", list(labels), index=None, placeholder="Select...")
        if d2.button("Delete", disabled=pick is None):
            engine.delete_song(labels[pick])
            st.rerun()
    else:
        st.caption("No songs yet — add one above.")

# ----------------------------------------------------------------- Identify
with tab_id:
    st.subheader("Identify a clip")
    mode = st.radio("Source", ["Microphone", "Upload file"], horizontal=True)
    audio = None
    if mode == "Microphone":
        st.caption("Record 5–10 seconds, close to the sound source.")
        audio = st.audio_input("Record a clip")
    else:
        audio = st.file_uploader("Upload an audio clip", type=["mp3", "wav", "m4a", "ogg", "flac"],
                                 key="id_upload")

    if st.button("Identify", type="primary", disabled=audio is None):
        t0 = time.perf_counter()
        try:
            with st.spinner("Listening..."):
                suffix = Path(getattr(audio, "name", "") or "clip.wav").suffix or ".wav"
                res = engine.identify(audio.getvalue(), suffix)
            elapsed = time.perf_counter() - t0
            st.success("Match found!")
            st.markdown(f"### {res.title}\n**{res.artist}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{res.confidence * 100:.0f}%")
            m2.metric("Aligned hashes", res.score)
            m3.metric("Time taken", f"{elapsed:.2f}s")
            st.progress(res.confidence)
            mins, secs = divmod(int(res.clip_start_s), 60)
            st.caption(f"Clip starts at ≈{mins}:{secs:02d} in the song · "
                       f"{res.query_hashes:,} query hashes · runner-up score {res.runner_up_score}")
        except engine.NoMatchError as exc:
            st.warning("No confident match. The song may not be in the library, or the clip is too "
                       f"noisy (best alignment: {exc.best_count} hashes, need {engine.MIN_MATCHES}). "
                       "Try a longer, cleaner clip with less background noise.")
        except engine.SilentAudioError:
            st.warning("The clip is silent. Check your microphone and try again.")
        except engine.AudioTooShortError as exc:
            st.warning(str(exc))
        except engine.FingerprintError as exc:
            st.error(str(exc))
        except DatabaseError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

import streamlit as st
import os
import yt_dlp
import whisper
import pandas as pd
import numpy as np
import subprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from moviepy import VideoFileClip
from moviepy import concatenate_videoclips

st.set_page_config(page_title="YouTube Highlight Extractor", layout="wide")

st.title("🎥 YouTube Highlight Extraction")

youtube_url = st.text_input("Enter YouTube URL")

if st.button("Generate Highlights"):

    os.makedirs("downloads", exist_ok=True)
    os.makedirs("audio", exist_ok=True)

    with st.spinner("Downloading video..."):

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "merge_output_format": "mp4",
            "outtmpl": "downloads/video.%(ext)s",
            "noplaylist": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

    video_path = None

    for file in os.listdir("downloads"):
        if file.endswith(".mp4"):
            video_path = os.path.join("downloads", file)
            break

    st.success("Video Downloaded")

    # Extract Audio
    audio_path = "audio/audio.wav"

    subprocess.run([
        "ffmpeg",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
        "-y"
    ])

    st.success("Audio Extracted")

    # Whisper
    with st.spinner("Transcribing..."):
        model = whisper.load_model("base")

        result = model.transcribe(
            audio_path,
            language="en"
        )

    segments = result["segments"]

    rows = []

    for seg in segments:
        rows.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    df = pd.DataFrame(rows)

    df["clean_text"] = df["text"].astype(str)

    df["word_count"] = df["clean_text"].apply(
        lambda x: len(x.split())
    )

    df = df[df["word_count"] >= 3]

    # merged transcript

    merged = []
    i = 0

    while i < len(df):

        current = df.iloc[i].copy()

        while (
            i + 1 < len(df)
            and len(current["clean_text"].split()) < 8
        ):
            nxt = df.iloc[i + 1]

            current["clean_text"] += " " + nxt["clean_text"]
            current["end"] = nxt["end"]

            i += 1

        merged.append(current)
        i += 1

    merged_df = pd.DataFrame(merged)

    conversations = []

    for i in range(len(merged_df)):

        prev_text = ""
        next_text = ""

        if i > 0:
            prev_text = merged_df.iloc[i - 1]["clean_text"]

        if i < len(merged_df) - 1:
            next_text = merged_df.iloc[i + 1]["clean_text"]

        current = merged_df.iloc[i]["clean_text"]

        conversations.append(
            f"""
            Previous: {prev_text}

            Current: {current}

            Next: {next_text}
            """
        )

    conversation_df = pd.DataFrame({
        "start": merged_df["start"],
        "end": merged_df["end"],
        "conversation": conversations
    })

    analyzer = SentimentIntensityAnalyzer()

    conversation_df["sentiment_score"] = (
        conversation_df["conversation"]
        .apply(
            lambda x:
            abs(analyzer.polarity_scores(x)["compound"])
        )
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30
    )

    tfidf = vectorizer.fit_transform(
        conversation_df["conversation"]
    )

    conversation_df["tfidf_score"] = (
        np.asarray(tfidf.sum(axis=1)).flatten()
    )

    conversation_df["highlight_score"] = (
        conversation_df["tfidf_score"] * 0.6
        + conversation_df["sentiment_score"] * 0.4
    ) * 100

    ranked_df = conversation_df.sort_values(
        "highlight_score",
        ascending=False
    )

    top_highlights = ranked_df.head(6)

    st.subheader("Top Highlights")

    st.dataframe(
        top_highlights[
            [
                "start",
                "end",
                "highlight_score"
            ]
        ]
    )

    # Clip Extraction

    with st.spinner("Creating highlight video..."):

        video = VideoFileClip(video_path)

        clips = []

        for _, row in top_highlights.iterrows():

            start_time = max(
                0,
                row["start"] - 5
            )

            end_time = min(
                video.duration,
                row["start"] + 5
            )

            clips.append(
                video.subclip(
                    start_time,
                    end_time
                )
            )

        final_video = concatenate_videoclips(
            clips,
            method="compose"
        )

        output_file = "FINAL_HIGHLIGHTS.mp4"

        final_video.write_videofile(
            output_file,
            codec="libx264",
            audio_codec="aac"
        )

    st.success("Highlight Video Created")

    st.video(output_file)

    with open(output_file, "rb") as f:
        st.download_button(
            "Download Highlight Video",
            f,
            file_name="FINAL_HIGHLIGHTS.mp4"
        )

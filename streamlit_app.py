import tempfile
from pathlib import Path

import streamlit as st

from src.highlights.config import load_config
from src.highlights.infer import run_inference


st.set_page_config(page_title="CLIP Highlight Detection", layout="wide")
st.title("CLIP Highlight Detection")

config_path = st.sidebar.text_input("Config", "configs/clip_tcn_mrhisum.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint", "outputs/checkpoints/best.pt")
make_preview = st.sidebar.checkbox("Render highlight preview", value=True)
uploaded = st.file_uploader("Upload mp4", type=["mp4", "mov", "m4v"])

if uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        video_path = tmp_dir / uploaded.name
        video_path.write_bytes(uploaded.read())
        st.video(str(video_path))
        if st.button("Run inference", type="primary"):
            with st.spinner("Extracting CLIP features and scoring highlights..."):
                cfg = load_config(config_path, repo_root=Path.cwd())
                result = run_inference(
                    video_path=video_path,
                    checkpoint_path=checkpoint_path,
                    out_dir="outputs/streamlit_demo",
                    config=cfg,
                    make_preview=make_preview,
                )
            scores = result["scores"]
            intervals = result["highlights"]["intervals"]
            left, right = st.columns([2, 1])
            with left:
                st.image(result["timeline_path"], caption="Highlight score timeline")
                if result["preview_path"]:
                    st.video(result["preview_path"])
            with right:
                st.metric("Duration, sec", f"{scores['duration_sec']:.1f}")
                st.metric("Processing, sec", f"{scores['processing_sec']:.1f}")
                if scores["real_time_factor"] is not None:
                    st.metric("RTF", f"{scores['real_time_factor']:.3f}")
                st.dataframe(intervals, use_container_width=True)
else:
    st.info("Upload a short video and choose a trained checkpoint.")

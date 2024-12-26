import streamlit as st
import os
from PIL import Image
import numpy as np
from typing import List
from dataclasses import dataclass
from datetime import datetime, date
import torch
import clip
from mergedcpu import MobileImageSearch
from update import ImageSearchUpdater
import shutil
from pathlib import Path
import time

# Set page configuration
st.set_page_config(
    page_title="Smart Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Complete UI styling

st.markdown("""
<style>
    /* Global styles */
    [data-testid="stAppViewContainer"] {
        background: #2c2f33;
    }

    /* Header styles */
    .search-header {
        text-align: center;
        padding: 2rem 0;
        background: #3b4148;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        position: sticky;
        top: 0;
        z-index: 100;
    }

    /* Logo styles */
    .logo {
        font-size: 2.5rem;
        font-weight: 700;
        color: #03A9F4;
        margin-bottom: 1rem;
        font-family: 'Google Sans', sans-serif;
    }

    /* Search box styles */
    .search-container {
        max-width: 700px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 24px;
        border: 2px solid #03A9F4;
        box-shadow: 0 4px 12px rgba(3, 169, 244, 0.1);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:hover {
        box-shadow: 0 4px 16px rgba(3, 169, 244, 0.15);
    }
    .stTextInput > div > div > input:focus {
        border-color: #03A9F4;
        box-shadow: 0 4px 20px rgba(3, 169, 244, 0.2);
    }

    /* Image grid styles */
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 1rem;
        padding: 2rem;
    }

    .image-card {
        background: #3b4148;
        border-radius: 15px;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }

    /* Button styles */
    .stButton > button {
        border-radius: 30px;
        padding: 15px 25px;
        background: linear-gradient(45deg, #03A9F4, #4CAF50);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(3, 169, 244, 0.2);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #03A9F4, #4CAF50);
        color: white;
    }

    /* Search history styling */
    .history-item {
        padding: 8px 15px;
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        margin: 5px 0;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .history-item:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }

    /* Upload section styles */
    .upload-section {
        background: #3b4148;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 2rem 0;
        text-align: center;
    }

    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 80px;
        height: 80px;
        margin: 2rem auto;
    }
    .loading-spinner:after {
        content: " ";
        display: block;
        width: 64px;
        height: 64px;
        border-radius: 50%;
        border: 6px solid #6c5ce7;
        border-color: #6c5ce7 transparent #6c5ce7 transparent;
        animation: loading-spinner 1.2s linear infinite;
    }
    @keyframes loading-spinner {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #6c5ce7;
    }

    /* Upload status */
    .upload-status {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 8px;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class ImageSearchApp:
    def __init__(self):
        self.initialize_session_state()
        self.search_engine = self.initialize_search_engine()
        self.storage_path = "/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage"
        self.updater = ImageSearchUpdater(
            images_folder='/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage',  # Replace with your images folder path
            embedding_cache_file='embedding_cache1.npy',
            batch_size=16
        )

    @staticmethod
    def initialize_session_state():
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'last_search_results' not in st.session_state:
            st.session_state.last_search_results = []
        if 'similarity_threshold' not in st.session_state:
            st.session_state.similarity_threshold = 0.25
        if 'processing_upload' not in st.session_state:
            st.session_state.processing_upload = False
        if 'upload_status' not in st.session_state:
            st.session_state.upload_status = None

    @staticmethod
    def initialize_search_engine():
        return MobileImageSearch(
            images_folder="/Users/ayaankhan/Desktop/Testing/pythoncodes/hackathon/results/Storage",
            embedding_cache_file="embedding_cache1.npy",
            similarity_threshold=st.session_state.similarity_threshold,
            batch_size=16
        )

    def render_header(self):
        st.markdown("""
            <div class="search-header">
                <div class="logo">🔍 Visual Search Engine</div>
                <p style='text-align: center; font-size: 20px; color: #666; margin-bottom: 30px;'>
                    Discover your images through the power of AI-driven natural language search
                </p>
            </div>
        """, unsafe_allow_html=True)

    def save_uploaded_files(self, uploaded_files):
        saved_paths = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type.startswith('image/'):
                file_path = os.path.join(self.storage_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(file_path)
        return saved_paths

    def render_upload_section(self):


        col1, _= st.columns(2)

        with col1:
            uploaded_files = st.file_uploader(
                "Upload Individual Images",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                key="single_upload"
            )



        if uploaded_files:
            if st.button("Process Uploads", type="primary"):
                st.session_state.processing_upload = True
                saved_paths = []

                with st.spinner("Processing uploads..."):
                    if uploaded_files:
                        saved_paths.extend(self.save_uploaded_files(uploaded_files))


                if saved_paths:
                    progress_placeholder = st.empty()
                    progress_placeholder.markdown("""
                        <div style='text-align: center;'>
                            <div class='loading-spinner'></div>
                            <h3>Creating embeddings for new images...</h3>
                            <p>This may take a few seconds depending on the number of images.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    try:
                        self.updater.update_index_with_new_images()
                        self.search_engine = self.initialize_search_engine()

                        progress_placeholder.success(f"✅ Successfully processed {len(saved_paths)} new images!")
                        st.session_state.upload_status = "success"
                        time.sleep(2)
                        progress_placeholder.empty()

                    except Exception as e:
                        progress_placeholder.error(f"Error processing images: {str(e)}")
                        st.session_state.upload_status = "error"

                    st.session_state.processing_upload = False

    def render_search_box(self):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            query = st.text_input(
                "",
                placeholder="Try 'sunset on beach' or 'person smiling'...",
                key="search_input"
            )

            search_col, space_col = st.columns([2, 3])
            with search_col:
                search_clicked = st.button("🔍 Search", use_container_width=True)

        return query if search_clicked or query else None

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("""
            <h2 style='color: white; margin-bottom: 20px;'>🎮 Control Panel</h2>
            """, unsafe_allow_html=True)

            new_threshold = st.slider(
                "✨ Similarity Threshold",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.similarity_threshold,
                step=0.05,
                help="Adjust to control the precision of search results"
            )

            if new_threshold != st.session_state.similarity_threshold:
                st.session_state.similarity_threshold = new_threshold
                self.search_engine = self.initialize_search_engine()

            st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

            if st.session_state.search_history:
                st.markdown("""
                <h3 style='color: white; margin-bottom: 15px;'>🕒 Recent Searches</h3>
                """, unsafe_allow_html=True)
                for query in reversed(st.session_state.search_history[-5:]):
                    st.markdown(f"""
                    <div class='history-item'>🔍 {query}</div>
                    """, unsafe_allow_html=True)

            st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)

            st.markdown("""
            <h3 style='color: white; margin-bottom: 15px;'>📊 Database Stats</h3>
            """, unsafe_allow_html=True)

            stats = self.search_engine.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Images", f"{stats['total_images']:,}")
            with col2:
                st.metric("Indexed", f"{stats['indexed_images']:,}")

    def render_image_grid(self, image_paths: List[str]):
        if not image_paths:
            st.warning(
                "🔍 No images found matching your search criteria. Try adjusting your search terms or similarity threshold!")
            return

        st.markdown(f"""
        <div style='color: #4a4a4a; font-size: 20px; margin: 25px 0; padding: 15px 25px; 
                    border-radius: 12px; background: linear-gradient(135deg, #f6f8fb, #f0f2f6);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>
            ✨ Found {len(image_paths)} matching images
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, image_path in enumerate(image_paths):
            try:
                with cols[idx % 3]:
                    img = Image.open(image_path)
                    aspect_ratio = img.size[1] / img.size[0]
                    target_width = 400
                    target_height = int(target_width * aspect_ratio)
                    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                    st.image(
                        img,
                        caption=os.path.basename(image_path),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Unable to load image: {os.path.basename(image_path)}")

    def run(self):
        self.render_header()
        self.render_sidebar()
        self.render_upload_section()

        if not st.session_state.processing_upload:
            query = self.render_search_box()

            if query:
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)

                with st.spinner('🔮 Exploring your image collection...'):
                    results = self.search_engine.search_images(query)
                    st.session_state.last_search_results = results

                self.render_image_grid(results)

            elif st.session_state.last_search_results:
                self.render_image_grid(st.session_state.last_search_results)


if __name__ == "__main__":
    app = ImageSearchApp()
    app.run()
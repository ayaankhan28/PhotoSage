# Photosage

# PhotoSage: AI-Powered Image Search Engine

<img width="1440" alt="Screenshot 2024-12-21 at 4 48 13 PM" src="https://github.com/user-attachments/assets/8dd79df0-8d0d-4920-876b-bb1893e9f3f7" />
<img width="1437" alt="Screenshot 2024-12-21 at 4 49 38 PM" src="https://github.com/user-attachments/assets/fd8f9463-0785-4f00-bbc8-ae03fd17be45" />

An intelligent image search engine that enables natural language queries to find relevant images in your collection. Built with Python, CLIP, and FAISS, PhotoSage transforms the way you interact with your photo library.

## 🚀 Inspiration

Modern smartphones contain thousands of photos, making it increasingly difficult to find specific images. Traditional methods rely on manually scrolling and searching images, which is time-consuming and often ineffective. PhotoSage solves this by allowing users to find photos using natural language descriptions, making photo organization and retrieval intuitive and efficient.

## 🧠 What it does

PhotoSage uses vectore embedding to understand both images and natural language queries, enabling users to:

- Search photos using natural language descriptions
- Find images based on content, context, and abstract concepts
- Get results in (<3 seconds)
- Process and organize large photo collections efficiently

## ⚙️ How we built it

### Tech Stack

- **Frontend**: Streamlit for the web interface(For demo only)
    - Final Product will be in smartphone as App.
- **Core AI**: CLIP (Contrastive Language-Image Pre-training) by OpenAI
- **Search Algorythm**: FAISS (Facebook AI Similarity Search)
- **Processing**: Python, PyTorch, NumPy
- **Image Processing**: Pillow, torchvision

### Key Components

- `MobileImageSearch`: Core search engine implementation
- `ImageSearchUI`: Streamlit-based user interface
- Embedding cache system for performance optimization
- FAISS indexing for efficient similarity search

## 💡 Challenges we ran into

1. **Performance Optimization**
    - Solved slow search times by implementing embedding caching
    - Optimized memory usage through batch processing
    - Balanced accuracy vs. speed in similarity matching
    - Used FAISS over simple cosine similarity
        - 
        
        | Feature | Cosine Similarity | FAISS |
        | --- | --- | --- |
        | Dataset Size | Small to medium datasets | Large-scale datasets |
        | Search Speed | Slow for large datasets | Fast with ANN optimization |
        | Accuracy | Exact | Adjustable (exact/approximate) |
        | Memory Usage | High for large datasets | Efficient with quantization |
        | Hardware Support | CPU only (typically) | CPU & GPU support |
2. **Scale Management**
    - Implemented efficient indexing for large photo collections
    - Created a robust update system for incremental processing

## 🏆 Accomplishments that we're proud of

- Achieved 2-3-second search times on 2000 photo collections
- High Accuracy beating the **Google Photos** in accuracy in 70% percent of test cases.
- Created an intuitive, user-friendly interface (Will be same for mobile app)
- Implemented efficient CPU-only processing for broad compatibility
- Developed a robust caching system for optimal performance

## 🧪 What we learned

- Vector search optimization techniques
- Vector similarity search implementation
- Large-scale image processing strategies

## 🔮 What's next for PhotoSage

- Mobile app development (Releasing for people soon)
- 3D models search for 3D artists
- Search for videos to return the timestamp for a scene happend in a video.
- More optimization for Near Instant result.
- Will release for some close people to test the market requirements and Iterate over
- for now it search in a sing folder. In later version it will get you the data from all over the device.

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
clip-by-openai>=0.2.0
faiss-cpu>=1.7.4
streamlit>=1.27.0
Pillow>=9.5.0
numpy>=1.24.0

```

## 🔧 Installation

1. Clone the repository:

```bash
git clone git@github.com:Metadome-emergingtechhackathon/hackathon-photosage.git
cd photosage

```

1. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\\venv\\Scripts\\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

1. Install dependencies:

```bash
pip install -r requirements.txt

```

## 🛠️ Configuration

1. Update image folder path in configuration files:

```python
images_folder = "/path/to/your/Storage"  # Update this path where all images are stored.

```

1. Configure search parameters (optional):

```python
search_engine = MobileImageSearch(
    images_folder="your/images/path",
    embedding_cache_file="embedding_cache.npy",
    similarity_threshold=0.25,
    batch_size=16
)

```

## 🚀 Usage

### Initial Setup

1. Generate embeddings:

```bash
python update.py

```

1. Launch the interface:

```bash
streamlit run main.py

```

1. Access the web interface at `http://localhost:8501`

### Search Tips

1. Use descriptive queries:
    - "person wearing red shirt on beach"
    - "sunset over mountains"
    - "group photo at birthday party"
2. Adjust similarity threshold:
    - Higher (>0.3): More precise, fewer results
    - Lower (<0.2): More results, less precise
    - Default: 0.25

## 📊 System Requirements

Minimum:

- CPU: Multi-core processor
- RAM: 8GB
- Storage: Varies with collection size
- Python 3.9+

Recommended:

- CPU: 4+ cores
- RAM: 16GB+
- SSD Storage
- GPU (optional)

## 📁 Project Structure

```
photosage/
│
├── main.py                 # Streamlit interface
├── update.py              # Embedding generation and update
├── requirements.txt       # Dependencies
├── embedding_cache.npy    # Cache file
└── Storage/               # Image storage

```

## 🔄 Update Process

To add new images:

1. Add images to the configured folder or you can directly upload images from ui.
2. Run `python update.py`
3. Restart the Streamlit interface

## 🛠️ Troubleshooting

1. No search results:
    - Lower similarity threshold
    - Use more general search terms
    - Verify image indexing
2. Slow performance:
    - Reduce batch size
    - Check system memory
    - Verify storage speed
3. Common errors:
    - "No module found": Reinstall requirements
    - "Cannot access folder": Check permissions
    - "Out of memory": Adjust batch size

## 👥 Team

- Ayaan Khan
    - Design
    - Algorithm design and implementation
    - Execution

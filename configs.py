import os, sys
from pathlib import Path
from collections import OrderedDict

APP_VERSION = "1.0.0"
STOPWORDS = [" ", ",", "，", "。", "的", "‘", "’", "“", "”", "、","_","-"]
SCRIPT_DIR = Path(__file__).resolve().parent

DEVICE='CPU'  # placeholder

def get_English_only_folder() -> Path:
    """
    Solves the issue that faiss can't handle paths containing non-ASCII characters
    Returns:

    """
    if sys.platform.startswith("win"):
        # better to store the files in the installation folder it's usable
        # base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        base = os.path.dirname(os.path.abspath(__file__))
        if base.isascii():  # all characters are ASCII
            return Path(base)
        public_env = os.environ.get("PUBLIC")
        if public_env:
            return Path(public_env)
        # Fallback
        return Path(r"C:\Users\Public")
    else:
        return Path("/tmp")


PROGRAM_DATA_PATH = get_English_only_folder() / "local_search"
AUTOSTART_ARG = "--autostart"  # for quietly starting the app when autostart is enabled
INDEX_DIR = str(
    PROGRAM_DATA_PATH / "index"
)  # Directory to save/load search index
CONFIG_FILE = str(PROGRAM_DATA_PATH / "config.json")
NO_CONTENT_PARSING_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff",
                                 ".mp3",".wav",".flac",".aac",".m4a",".wma",
                                 ".mp4",".mkv",".mov",".avi",".wmv",".webm"]
CONTENT_PARSING_EXTENSIONS = [".txt", ".docx", ".pptx", ".md", ".xlsx",
                        ".xls", ".csv", ".html", ".htm", ".odt", ".pdf", ".xml",]
SUPPORTED_EXTENSIONS = CONTENT_PARSING_EXTENSIONS + NO_CONTENT_PARSING_EXTENSIONS
# The default model is multilingual-e5-small, with some onnx optimizations
EMBEDDING_TOKENIZER_FILE = str(SCRIPT_DIR / "resources/embedding_model/tokenizer.json")
EMBEDDING_MODEL_ONNX_FILE = str(SCRIPT_DIR / "resources/embedding_model/model.onnx") # the model_gpu.onnx is the model_O4.onnx of multilingual-e5-small
EMBEDDING_MODEL_TOKEN_LENGTH = 512
EMBEDDING_DIM = 384
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
SEARCH_ENTRIES_TOPK=70
SEARCH_MODE_MAPPING = OrderedDict(
    {
        "Smart Search": "hybrid",
        "Keyword Search": "bm25_document",
        "Semantic Only Search": "embedding"
    }
)
QUERY_PREFIX="query: " # for multilingual-e5 models. set this to empty string if using other models
CHUNK_PREFIX="passage: " # for multilingual-e5 models. set this to empty string if using other models
LOG_PATH = str(PROGRAM_DATA_PATH / "local_search_app.log")
EMBEDDING_ENCODING_BATCH_SIZE=64
FILE_CHANGE_CHECK_INTERVAL = 60 # seconds
def is_running_in_pyinstaller():
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")
<p align="center">
<img src="resources/icon.png" width="350" />
</p>

[English](./README.md) | [中文文档](./README-zh.md)

# LocalSearch

一个隐私优先的 PySide6 桌面应用程序，用于搜索本地文件和文件夹中的内容。它结合了**词法（关键词）搜索 (BM25)** 和 **向量嵌入相似度 (Embedding similarity)**。兼容 Windows、Linux 和 macOS。

UI 设计灵感来源于 [gety.ai](https://gety.ai/)。

---

# 为什么选择 LocalSearch

你是否曾尝试在本地电脑的**数百个文档中搜索某些内容**，但**记不清确切的词汇**？例如，你想搜索“太阳能”，但实际文件使用的是“光伏阵列”这个术语。

本应用让你能够同时利用**语义搜索（Semantic Search）和传统的关键词搜索 (BM25)** 来检索文件内容（以及文件名、路径）。

### **核心功能**：

* **平台兼容性**：支持 Windows、Linux、macOS。
* **无需格式转换**：无需更改现有文档格式（无需迁移到专有的知识库格式）。
* **混合搜索**：结合 Embedding 向量与词法搜索。
* **多格式支持**：pptx, docx, md, txt, xlsx, csv, pdf, html, odt 等。
* **结果预览**：在预览面板中直接查看搜索结果。
* **筛选功能**：可按日期和文件类型过滤搜索结果。
* **自定义范围**：自由选择需要包含在搜索中的文件夹。
* **结果高亮**：搜索关键词自动高亮显示。
* **多语言支持**：Embedding 模型支持包括中文、英文在内的约 100 种语言（[查看详情](https://huggingface.co/intfloat/multilingual-e5-small)）。
* **GPU 加速**：支持 GPU 计算以加快文本向量化速度。
* **安全与隐私**：完全离线运行。
* **自动监控**：自动监测文件内容变更。
* **性能优化**：针对运行效率进行了深度优化。

---

# 屏幕截图

---

# 与其他同类工具的对比

一些工具（如 Windows 上的 Everything）默认只搜索文件名，缺乏对文件内容进行语义搜索的能力。使用 Everything 进行内容搜索速度较慢，因为它不建立索引。

目前有很多具备混合搜索功能的新工具，如 Cherry Studio, AnythingLLM, Maxkb, FastGPT, Obsidian, Logseq 等。

然而，**没有一款工具能同时具备以下特性**：

* 完全离线运行，支持多种文件格式。
* 无需部署庞大的 Docker 容器，拥有原生 GUI。
* 支持多种语言（例如，中文等语言需要额外的分词处理）。
* 无需用户手动将文档迁移到专有的知识库中。

这就是我开发 LocalSearch 的原因。

---

# 安装与使用

只要能安装 PySide6 和其他必要依赖，`LocalSearch` 就可以在 Windows、Linux 和 macOS（未测试）上使用。

如果你是 Windows 用户，可以**直接从 Release 页面下载**。发布版本基于 DirectML，可以利用 GPU，但性能可能略逊于 CUDA。否则，请参考以下步骤：

首先，下载仓库（下载 ZIP 或使用 Git）：

```bash
git clone https://github.com/neural-koala/LocalSearch.git
cd LocalSearch

```

如果 `resources/embedding_model` 目录下的 ONNX 模型文件未正常下载，你可能需要执行 `git lfs pull`。

## 环境要求

强烈建议创建虚拟环境，因为本仓库需要卸载 `orjson`（由于 `bm25s` 包的 bug），这可能会破坏你现有的环境依赖。

```bash
conda create -n LocalSearch python=3.13
conda activate LocalSearch

# CPU 运行
pip install -r ./requirements.txt

# CUDA GPU 运行
# 确保已安装 GPU 版本的 torch，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu126 (或对应的 cuda 版本)
pip install -r ./requirements_gpu.txt
pip uninstall orjson

```

你也可以选择 `requirements_windows_DirectML.txt`，它支持 Windows 上的多种 GPU。如果你在 Linux 或 macOS 上，请根据 [ONNX Runtime 执行提供程序](https://onnxruntime.ai/docs/execution-providers/) 修改 requirements 中的 `onnxruntime` 版本。

## 使用 CPU 运行

直接运行：

```bash
python main.py

```

## 使用 GPU 运行

LocalSearch 使用 `onnxruntime-gpu` 或 `onnxruntime-directml` 来计算文本 Embedding。

运行应用：

```bash
python main.py

```

代码会自动检测你的硬件是否支持 GPU 加速。

---

# 高级配置

## 更换 Embedding 模型

默认情况下，程序使用 `multilingual-e5-small` 模型。你可以将自己的 `model.onnx` 和分词器放入 `resources/embedding_model` 进行替换。

然后，进入 `configs.py` 并相应地修改 `EMBEDDING_MODEL_TOKEN_LENGTH` 和 `EMBEDDING_DIM` 变量。

默认情况下，如果电脑有 GPU，程序会优先选择 `model_gpu.onnx` 文件，如果不存在则回退到 `model.onnx`。

如果你的 GPU 性能强劲且索引文件不多，可以考虑切换到更重的模型，如 `multilingual-e5-base`, `BGE-M3`, `EmbeddingGemma-300M`。

---

# 局限性

* **语言处理**：目前主要针对中英文文档，中文分词使用了 `rjieba`。Embedding 模型虽然支持 100 多种语言，且以空格分隔的语言（如英文）工作正常，但对于日文、韩文等不使用空格分隔的语言，你可能需要将 `rjieba` 修改为相应的分词器。目前似乎还没有一个能完美处理所有语言（中日韩泰等）分词的 Python 库。
* **OCR**：目前不对图片和 PDF 进行 OCR（仅提取 PDF 中的嵌入文本），这是出于资源占用、速度和应用体积的考虑。但你可以利用 `unstructured` 库轻松修改代码以支持此功能。
* **重排序**：为了保证 CPU 性能和更小的体积，目前采用基于启发式的重排序（Reranking），而非使用专门的重排序模型（如 flashrank）。


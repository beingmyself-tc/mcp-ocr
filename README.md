# MCP HunyuanOCR Server

A Model Context Protocol (MCP) server that provides local OCR capabilities using the [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) model (1B parameters).

## Features

- **Local Processing**: Runs entirely on your machine using PyTorch.
- **Apple Silicon Support**: Automatically uses MPS (Metal Performance Shaders) for acceleration on Mac.
- **MCP Compatible**: Works with Claude Desktop, VS Code (via MCP extension), and other MCP clients.

## Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`
- `poppler` (required for PDF support)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt-get install poppler-utils`
  - Windows: Download and add to PATH

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/beingmyself-tc/mcp-ocr.git
   cd mcp-ocr
   ```

2. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # OR using pip
   pip install .
   ```

## Usage

### Running the Server

You can run the server directly:

```bash
# If installed with uv
uv run mcp-ocr

# If installed with pip
python -m mcp_ocr.server
```

### Configuring with Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mcp-ocr": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/mcp-ocr",
        "run",
        "mcp-ocr"
      ]
    }
  }
}
```

### Configuring with VS Code

To use this MCP server with VS Code, add the following to your global MCP settings file (typically `~/Library/Application Support/Code/User/mcp.json` on macOS):

```json
{
  "servers": {
    "mcp-ocr": {
      "type": "stdio",
      "command": "/absolute/path/to/mcp-ocr/.venv/bin/python",
      "args": [
        "-m",
        "mcp_ocr.server"
      ]
    }
  }
}
```

Make sure to replace `/absolute/path/to/mcp-ocr` with the actual path to your cloned repository.

## Tools

### `ocr_image`
Performs OCR on a single image file.
- **Input**: `image_path` (string) - Absolute path to the image file (e.g., .jpg, .png).
- **Output**: Extracted text.

### `ocr_pdf`
Performs OCR on a PDF file.
- **Input**: `pdf_path` (string) - Absolute path to the PDF file.
- **Output**: Extracted text (concatenated pages).

## License

MIT

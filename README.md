# MCP HunyuanOCR Server

A Model Context Protocol (MCP) server that provides local OCR capabilities using the [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) model (1B parameters).

## Features

- **Local Processing**: Runs entirely on your machine using PyTorch.
- **Apple Silicon Support**: Automatically uses MPS (Metal Performance Shaders) for acceleration on Mac.
- **MCP Compatible**: Works with Claude Desktop, VS Code (via MCP extension), and other MCP clients.

## Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`

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
    "hunyuan-ocr": {
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

### Configuring with VS Code (MCP Extension)

If you are using an MCP extension in VS Code, configure it similarly to point to the `mcp-ocr` executable or run via `uv`.

## Tools

### `ocr_image`
Performs OCR on a local image file.
- **Input**: `image_path` (string) - Absolute path to the image.
- **Output**: Extracted text.

## License

MIT

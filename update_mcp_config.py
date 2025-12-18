import json
import os

config_path = os.path.expanduser("~/Library/Application Support/Code/User/mcp.json")

with open(config_path, 'r') as f:
    config = json.load(f)

config['servers']['mcp-ocr'] = {
    "type": "stdio",
    "command": "/Users/seb/code/mcp-ocr/.venv/bin/python",
    "args": [
        "-m",
        "mcp_ocr.server"
    ]
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print("Successfully updated mcp.json")

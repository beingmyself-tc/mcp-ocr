import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from pdf2image import convert_from_path
from mcp.server.fastmcp import FastMCP

# Initialize MCP Server
mcp = FastMCP("HunyuanOCR")

# Global model variables
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print("Loading HunyuanOCR model...")
        model_path = "tencent/HunyuanOCR"
        
        # Use MPS (Metal) if available, otherwise CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("Model loaded successfully.")

@mcp.tool()
def ocr_image(image_path: str) -> str:
    """
    Perform OCR on an image file or PDF using HunyuanOCR.
    
    Args:
        image_path: Absolute path to the image or PDF file.
    """
    global model, processor
    
    if not os.path.exists(image_path):
        return f"Error: File not found at {image_path}"
        
    if model is None:
        load_model()
        
    try:
        images = []
        if image_path.lower().endswith('.pdf'):
            try:
                # Convert PDF to list of PIL Images
                images = convert_from_path(image_path)
            except Exception as e:
                return f"Error converting PDF: {str(e)}. Make sure poppler is installed (brew install poppler)."
        else:
            # Handle single image
            images = [Image.open(image_path).convert("RGB")]
            
        results = []
        for i, image in enumerate(images):
            # Prepare inputs
            # HunyuanOCR typically uses a prompt like "OCR" or specific instructions
            # Based on repo: "OCR" is the standard prompt for full text extraction
            prompt = "OCR" 
            
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=False # Deterministic for OCR
                )
                
            # Decode
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            if len(images) > 1:
                results.append(f"--- Page {i+1} ---\n{text}")
            else:
                results.append(text)
                
        return "\n\n".join(results)
        
    except Exception as e:
        return f"Error processing file: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()

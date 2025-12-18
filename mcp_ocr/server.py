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

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.pdf'}
MAX_PDF_PAGES = 50  # Limit to prevent memory issues

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model():
    global model, processor
    if model is None:
        print("Loading HunyuanOCR model...")
        model_path = "tencent/HunyuanOCR"
        
        device = get_device()
        print(f"Using device: {device}")
        
        # Use float16 for GPU, float32 for CPU
        dtype = torch.float32 if device == "cpu" else torch.float16

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=dtype,
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
    
    # Validate file extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"Error: Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        
    if model is None:
        load_model()
        
    try:
        images = []
        is_pdf = ext == '.pdf'
        
        if is_pdf:
            try:
                # Convert PDF to list of PIL Images
                pdf_images = convert_from_path(image_path)
                if len(pdf_images) > MAX_PDF_PAGES:
                    return f"Error: PDF has {len(pdf_images)} pages, exceeding limit of {MAX_PDF_PAGES}. Please split the PDF."
                images = [img.convert("RGB") for img in pdf_images]
            except Exception as e:
                return f"Error converting PDF: {str(e)}. Make sure poppler is installed (brew install poppler)."
        else:
            # Handle single image
            images = [Image.open(image_path).convert("RGB")]
            
        results = []
        for i, image in enumerate(images):
            # Prepare inputs
            prompt = "OCR"
            
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=1024,
                    do_sample=False
                )
                
            # Decode
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            if len(images) > 1:
                results.append(f"--- Page {i+1} ---\n{text}")
            else:
                results.append(text)
            
            # Free memory for large PDFs
            if is_pdf:
                image.close()
                
        return "\n\n".join(results)
        
    except Exception as e:
        return f"Error processing file: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()

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
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
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

def _run_inference(images: list[Image.Image]) -> str:
    """Internal helper to run OCR on a list of images."""
    global model, processor
    if model is None:
        load_model()
        
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
            
    return "\n\n".join(results)

@mcp.tool()
def ocr_image(image_path: str) -> str:
    """
    Perform OCR on a single image file.
    
    Args:
        image_path: Absolute path to the image file (e.g., .jpg, .png).
    """
    if not os.path.exists(image_path):
        return f"Error: File not found at {image_path}"
    
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.pdf':
         return "Error: This tool is for images only. Please use 'ocr_pdf' for PDF files."
    
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return f"Error: Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        
    try:
        image = Image.open(image_path).convert("RGB")
        return _run_inference([image])
    except Exception as e:
        return f"Error processing image: {str(e)}"

@mcp.tool()
def ocr_pdf(pdf_path: str) -> str:
    """
    Perform OCR on a PDF file.
    
    Args:
        pdf_path: Absolute path to the PDF file.
    """
    if not os.path.exists(pdf_path):
        return f"Error: File not found at {pdf_path}"
        
    if not pdf_path.lower().endswith('.pdf'):
        return "Error: File is not a PDF. Please use 'ocr_image' for image files."
        
    try:
        # Convert PDF to list of PIL Images
        try:
            pdf_images = convert_from_path(pdf_path)
        except Exception as e:
            return f"Error converting PDF: {str(e)}. Make sure poppler is installed (brew install poppler)."

        if len(pdf_images) > MAX_PDF_PAGES:
            return f"Error: PDF has {len(pdf_images)} pages, exceeding limit of {MAX_PDF_PAGES}. Please split the PDF."
            
        images = [img.convert("RGB") for img in pdf_images]
        result = _run_inference(images)
        
        # Cleanup
        for img in images:
            img.close()
            
        return result
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()

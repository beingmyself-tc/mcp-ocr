import os
from PIL import Image
from pdf2image import convert_from_path
from mcp.server.fastmcp import FastMCP
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Initialize MCP Server
mcp = FastMCP("Qwen2-VL-OCR-MLX")

# Global model variables
model = None
processor = None
config = None
MODEL_PATH = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
MAX_PDF_PAGES = 50  # Limit to prevent memory issues

# OCR prompts
PROMPTS = {
    "ocr": "Extract all text from this image exactly as it appears.",
    "format": "Extract all information from this document image and represent it in markdown format. Tables should be expressed in HTML format, formulas should use LaTeX format. Parse according to reading order.",
    "formula": "Extract all formulas from this image and represent them in LaTeX format.",
    "table": "Extract all tables from this image and represent them in HTML format.",
    "translation": "Extract all text from this image and translate it to English. Keep tables in HTML format and formulas in LaTeX format."
}

def load_model():
    global model, processor, config
    if model is None:
        print(f"Loading {MODEL_PATH} with MLX...")
        model, processor = load(MODEL_PATH, trust_remote_code=True)
        config = load_config(MODEL_PATH, trust_remote_code=True)
        print("Model loaded successfully.")

def _run_inference(images: list[Image.Image], mode: str = "ocr") -> str:
    """Internal helper to run OCR on a list of images."""
    global model, processor, config
    if model is None:
        load_model()
        
    prompt_text = PROMPTS.get(mode, PROMPTS["ocr"])
    
    results = []
    for i, image in enumerate(images):
        # Resize image if it's too large to save memory/compute
        max_size = 1280
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Apply chat template
        formatted_prompt = apply_chat_template(
            processor, config, prompt_text, num_images=1
        )
        
        # Generate
        output = generate(
            model, 
            processor, 
            formatted_prompt, 
            [image], 
            verbose=False,
            max_tokens=4096,
            temp=0.0
        )
        
        if len(images) > 1:
            results.append(f"--- Page {i+1} ---\n{output}")
        else:
            results.append(output)
            
    return "\n\n".join(results)

@mcp.tool()
def ocr_image(image_path: str, mode: str = "format") -> str:
    """
    Perform OCR on an image file using Qwen2-VL (MLX optimized).
    
    Args:
        image_path: Absolute path to the image file.
        mode: Processing mode. Options:
            - "ocr": Extract text exactly as it appears.
            - "format": Extract text/tables/formulas in Markdown/HTML/LaTeX (default).
            - "formula": Extract formulas in LaTeX.
            - "table": Extract tables in HTML.
            - "translation": Extract and translate to English.
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
        return _run_inference([image], mode=mode)
    except Exception as e:
        return f"Error processing image: {str(e)}"

@mcp.tool()
def ocr_pdf(pdf_path: str, mode: str = "format") -> str:
    """
    Perform OCR on a PDF file using Qwen2-VL (MLX optimized).
    
    Args:
        pdf_path: Absolute path to the PDF file.
        mode: Processing mode. Options:
            - "ocr": Extract text exactly as it appears.
            - "format": Extract text/tables/formulas in Markdown/HTML/LaTeX (default).
            - "formula": Extract formulas in LaTeX.
            - "table": Extract tables in HTML.
            - "translation": Extract and translate to English.
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
        result = _run_inference(images, mode=mode)
        
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

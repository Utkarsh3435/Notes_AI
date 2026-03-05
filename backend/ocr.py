"""
ocr.py — Extract text from handwritten PDF pages using Gemini Vision API.

Each PDF page is converted to an image and sent to Gemini 2.5 Flash for OCR.
"""

import os
import io
from pdf2image import convert_from_bytes
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.genai as genai


# Load API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables")


# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Recommended fast model for OCR
MODEL_NAME = "gemini-2.5-flash"


OCR_PROMPT = """
You are an expert OCR system for handwritten notes.

Rules:
- Transcribe the text EXACTLY as written.
- Preserve headings, lists, and structure.
- Do NOT summarize.
- Do NOT add explanations.
- If a word is unreadable write [illegible].

Return ONLY the transcription text.
"""


def page_to_image_bytes(pil_image: Image.Image, max_width: int = 1600) -> bytes:
    """
    Resize page image if needed and convert to JPEG bytes.
    """

    if pil_image.width > max_width:
        ratio = max_width / pil_image.width
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height), Image.LANCZOS)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    return buf.read()


def ocr_page(image_bytes: bytes, page_num: int) -> str:
    """
    Send one page image to Gemini and return the extracted text.
    """

    try:
        image = Image.open(io.BytesIO(image_bytes))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                OCR_PROMPT,
                image
            ]
        )

        text = response.text.strip()

        print(f"[OCR] Page {page_num} extracted ({len(text)} chars)")

        return text

    except Exception as e:
        print(f"[OCR] ERROR on page {page_num}: {str(e)}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes, pdf_id: str = "", filename: str = "") -> list[dict]:
    """
    Convert PDF pages to images and run OCR using Gemini.
    """

    print(f"[OCR] Converting PDF '{filename}' to images…")

    images = convert_from_bytes(pdf_bytes, dpi=200)

    print(f"[OCR] {len(images)} pages found — starting transcription…")

    pages = []

    # Run OCR in parallel but limited to avoid API limits
    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = {}

        for i, img in enumerate(images, start=1):

            print(f"[OCR] → Page {i}")

            img_bytes = page_to_image_bytes(img)

            futures[executor.submit(ocr_page, img_bytes, i)] = i

        for future in as_completed(futures):

            page_num = futures[future]

            text = future.result()

            pages.append({
                "page_num": page_num,
                "text": text
            })

    pages.sort(key=lambda x: x["page_num"])

    print(f"[OCR] Extracted {len(pages)} pages")

    return pages
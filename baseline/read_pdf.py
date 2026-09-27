import pymupdf
import pytesseract
from PIL import Image
import io

def ocr_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""

    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text += pytesseract.image_to_string(img) + "\n"

    return text

paper_text = ocr_pdf("paper.pdf")
print(paper_text[:3000])
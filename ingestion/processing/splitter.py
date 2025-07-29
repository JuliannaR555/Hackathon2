import fitz  # PyMuPDF
import docx
from bs4 import BeautifulSoup
from odf.opendocument import load as load_odt
from odf import text, teletype

def split_by_page(file_path):
    ext = file_path.lower().split('.')[-1]

    if ext == "pdf":
        doc = fitz.open(file_path)
        return [page.get_text() for page in doc]

    elif ext == "docx":
        doc = docx.Document(file_path)
        return [para.text for para in doc.paragraphs if para.text.strip() != ""]

    elif ext == "odt":
        odt_doc = load_odt(file_path)
        allparas = odt_doc.getElementsByType(text.P)
        return [teletype.extractText(p) for p in allparas if teletype.extractText(p).strip() != ""]

    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content.split("\n\n")

    elif ext in ["html", "htm"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return [soup.get_text()]

    else:
        return []

def score_longueur(text):
    words = len(text.split())
    return min(words / 100, 1.0)

def couleur_score(score):
    if score >= 0.8:
        return "#d4edda"  # vert clair
    elif score >= 0.5:
        return "#fff3cd"  # jaune clair
    else:
        return "#f8d7da"  # rouge clair



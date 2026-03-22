import re
from pypdf import PdfReader


# -----------------------------
# TEXT PROCESSING
def split_sentences(text):
    sentences = re.split(r"(?<=[.!?]) +", text)
    return sentences


def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks


# -----------------------------
# PDF & TXT PARSING
def extract_text_from_pdf(file):
    """
    Extract text from uploaded PDF file
    """
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text


def extract_text_from_txt(file):
    """
    Extract text from uploaded TXT file
    """
    return file.read().decode("utf-8")


def parse_document(file):
    """
    Detect file type and extract text
    """
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)

    elif file.type == "text/plain":
        return extract_text_from_txt(file)

    else:
        raise ValueError("Unsupported file type")


# -----------------------------
if __name__ == "__main__":
    sample_text = "This is a sample document. It contains multiple sentences!"
    print("Split Sentences:")
    for sentence in split_sentences(sample_text):
        print(sentence)

    print("\nChunked Text:")
    for chunk in chunk_text(sample_text, chunk_size=5, overlap=2):
        print(chunk)

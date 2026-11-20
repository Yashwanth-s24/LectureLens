
import re
from pypdf import PdfReader


UNWANTED_PHRASES = [
    "end of sample notes",
    "use this document to test",
    "lecturelens testing",
    "mcq generation",
    "lecturelens",
    "instead writing separate",
    "customers similar purchasing",
    "group customers similar",
    "generated summary",
    "important keywords",
    "important concepts",
    "practice questions",
    "show correct answer",
    "ai summary"
]


def clean_pdf_text(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        lower_line = line.lower()

        # Remove unwanted phrases.
        if any(
            phrase in lower_line
            for phrase in UNWANTED_PHRASES
        ):
            continue

        # Remove standalone page numbers.
        if re.fullmatch(
            r"(page\s*)?\d+(\s*of\s*\d+)?",
            line,
            re.IGNORECASE
        ):
            continue

        # Remove common document headings.
        if re.fullmatch(
            r"(chapter|unit|section)\s*\d+.*",
            line,
            re.IGNORECASE
        ):
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


def extract_text_from_pdf(pdf_file):

    try:

        pdf_file.seek(0)

        reader = PdfReader(pdf_file)

        extracted_pages = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                extracted_pages.append(page_text)

        if not extracted_pages:
            raise ValueError(
                "No readable text found in this PDF."
            )

        complete_text = "\n".join(extracted_pages)

        cleaned_text = clean_pdf_text(
            complete_text
        )

        if not cleaned_text:
            raise ValueError(
                "No meaningful lecture content found."
            )

        return cleaned_text

    except Exception as error:

        raise ValueError(
            f"Unable to read PDF: {error}"
        )

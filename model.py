import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------
# MODEL CONFIGURATION
# --------------------------------

MODEL_NAME = "google/flan-t5-small"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading LectureLens model on: {DEVICE}")


# --------------------------------
# LOAD PRETRAINED MODEL
# --------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model = model.to(DEVICE)
model.eval()

print("LectureLens model loaded successfully!")


# --------------------------------
# COMMON TEXT GENERATION FUNCTION
# --------------------------------

def generate_response(prompt, max_tokens=200):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response.strip()


# --------------------------------
# 1. TEXT SUMMARIZATION
# --------------------------------

def summarize_text(text):

    prompt = (
        "summarize: "
        "Write a clear summary of the following text in 2 to 3 sentences. "
        "Include the main topic, important concepts, and key facts. "
        "Do not give general advice or introduce information "
        "that is not in the text.\n\n"
        f"{text}"
    )

    return generate_response(prompt, max_tokens=120)


# --------------------------------
# 2. QUESTION ANSWERING
# --------------------------------

def answer_question(text, question):

    prompt = (
        "Answer the question using only the information "
        "provided in the lecture notes. "
        "If the answer is not present, say that the notes "
        "do not contain enough information.\n\n"
        f"Lecture notes:\n{text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return generate_response(prompt, max_tokens=180)


# --------------------------------
# 3. MCQ GENERATION
# --------------------------------

def generate_mcqs(text):

    prompt = (
        "Create 3 multiple choice questions based on these notes. "
        "For each question, provide four options labelled A, B, C and D. "
        "Give the correct answer after each question. "
        "Do not create questions about information absent from the notes.\n\n"
        f"Lecture notes:\n{text}\n\n"
        "Questions:"
    )

    return generate_response(prompt, max_tokens=350)


# --------------------------------
# 4. KEYWORD EXTRACTION
# --------------------------------

def extract_keywords(text):

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10
    )

    try:

        matrix = vectorizer.fit_transform([text])

        scores = matrix.toarray()[0]
        words = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(words, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [word for word, score in ranked]

    except ValueError:

        return []


# --------------------------------
# 5. INPUT VALIDATION
# --------------------------------

def validate_text(text):

    if not text or not text.strip():

        raise ValueError(
            "Please provide lecture notes."
        )

    if len(text.strip()) < 50:

        raise ValueError(
            "Please provide at least 50 characters of lecture notes."
        )

    return True
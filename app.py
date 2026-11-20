
import streamlit as st
import re
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_reader import extract_text_from_pdf


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="LectureLens",
    page_icon="📚",
    layout="wide"
)

st.title("📚 LectureLens")
st.subheader("AI-Powered Lecture Notes Assistant")

st.write(
    "Upload lecture notes to summarize content, extract "
    "important concepts, ask questions and generate MCQs."
)


# =====================================================
# CONCEPT DATABASE
# =====================================================

CONCEPTS = {
    "artificial intelligence": [
        "computer science",
        "human intelligence",
        "decision making",
        "problem solving"
    ],

    "machine learning": [
        "artificial intelligence",
        "learning patterns",
        "training data",
        "predictions"
    ],

    "supervised learning": [
        "labelled data",
        "classification",
        "regression",
        "known target"
    ],

    "unsupervised learning": [
        "unlabelled data",
        "clustering",
        "hidden patterns",
        "customer grouping"
    ],

    "reinforcement learning": [
        "agent",
        "environment",
        "rewards",
        "actions"
    ],

    "deep learning": [
        "neural networks",
        "multiple layers",
        "representation learning"
    ],

    "classification": [
        "supervised learning",
        "categorical output",
        "prediction"
    ],

    "regression": [
        "supervised learning",
        "continuous output",
        "prediction"
    ],

    "neural networks": [
        "deep learning",
        "neurons",
        "layers"
    ],

    "natural language processing": [
        "language",
        "text",
        "human communication"
    ],

    "computer vision": [
        "images",
        "image recognition",
        "visual information"
    ],

    "training data": [
        "machine learning",
        "model training",
        "examples"
    ],

    "labelled data": [
        "supervised learning",
        "known answers",
        "target values"
    ],

    "unlabelled data": [
        "unsupervised learning",
        "hidden patterns",
        "clustering"
    ]
}


# =====================================================
# TEXT CLEANING
# =====================================================

UNWANTED_PHRASES = [
    "end of sample notes",
    "use this document to test",
    "lecturelens testing",
    "mcq generation",
    "generated summary",
    "important keywords",
    "important concepts",
    "practice questions",
    "show correct answer",
    "ai summary",
    "instead writing separate",
    "customers similar purchasing",
    "group customers similar"
]


def clean_lecture_text(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        lower_line = line.lower()

        if any(
            phrase in lower_line
            for phrase in UNWANTED_PHRASES
        ):
            continue

        if re.fullmatch(
            r"(page\s*)?\d+(\s*of\s*\d+)?",
            line,
            re.IGNORECASE
        ):
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines).strip()


# =====================================================
# SENTENCE PROCESSING
# =====================================================

def split_sentences(text):

    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.split()) >= 5
    ]


# =====================================================
# 1. EXTRACTIVE SUMMARIZATION
# =====================================================

def summarize_notes(text):

    sentences = split_sentences(text)

    if not sentences:
        return "Not enough meaningful text to summarize."

    if len(sentences) <= 5:
        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(sentences)

        scores = matrix.sum(axis=1).A1

        scores = scores / [
            len(sentence.split()) ** 0.5
            for sentence in sentences
        ]

        for index in range(min(3, len(scores))):
            scores[index] *= 1.15

        number_to_select = min(6, len(sentences))

        selected_indices = scores.argsort()[
            ::-1
        ][:number_to_select]

        selected_indices = sorted(selected_indices)

        summary = [
            sentences[index]
            for index in selected_indices
        ]

        return " ".join(summary)

    except ValueError:

        return "Unable to generate summary."


# =====================================================
# 2. KEYWORD EXTRACTION
# =====================================================

def get_keywords(text):

    text_lower = text.lower()

    found_concepts = []

    for concept in CONCEPTS:

        pattern = r"\b" + re.escape(concept) + r"\b"

        if re.search(pattern, text_lower):
            found_concepts.append(concept)

    try:

        sentences = split_sentences(text)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(2, 3),
            max_features=150
        )

        matrix = vectorizer.fit_transform(sentences)

        terms = vectorizer.get_feature_names_out()

        scores = matrix.sum(axis=0).A1

        ranked_terms = sorted(
            zip(terms, scores),
            key=lambda item: item[1],
            reverse=True
        )

        unwanted_words = {
            "example", "examples", "important",
            "introduction", "lecturelens",
            "generated", "summary", "question",
            "questions", "answer", "answers",
            "using", "instead", "writing",
            "separate", "customers", "purchasing",
            "group", "maximize", "total", "reward"
        }

        for term, score in ranked_terms:

            term = term.lower().strip()

            if term in found_concepts:
                continue

            if len(term.split()) < 2:
                continue

            if any(
                word in unwanted_words
                for word in term.split()
            ):
                continue

            if len(found_concepts) >= 12:
                break

            found_concepts.append(term)

    except ValueError:
        pass

    return found_concepts[:12]


# =====================================================
# 3. QUESTION ANSWERING
# =====================================================

def answer_from_notes(text, question):

    if not question.strip():
        return "Please enter a question."

    sentences = split_sentences(text)

    if not sentences:
        return "No readable information found."

    question_lower = question.lower()

    # Identify the main concept in the question.
    known_concepts = sorted(
        CONCEPTS.keys(),
        key=len,
        reverse=True
    )

    target_concept = None

    for concept in known_concepts:

        pattern = r"\b" + re.escape(concept) + r"\b"

        if re.search(pattern, question_lower):
            target_concept = concept
            break

    if not target_concept:

        return (
            "Please ask a question about a concept "
            "present in your lecture notes."
        )

    # Find sentences that explicitly mention the concept.
    matching_sentences = []

    concept_pattern = (
        r"\b" + re.escape(target_concept) + r"\b"
    )

    for sentence in sentences:

        if re.search(
            concept_pattern,
            sentence.lower()
        ):
            matching_sentences.append(sentence)

    if not matching_sentences:

        return (
            f"The notes do not contain information about "
            f"{target_concept.title()}."
        )

    try:

        # Calculate TF-IDF similarity only among relevant sentences.
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            matching_sentences + [question]
        )

        similarities = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        definition_patterns = [
            r"\bis\b",
            r"\bare\b",
            r"\brefers to\b",
            r"\bdefined as\b",
            r"\buses?\b",
            r"\binvolves?\b",
            r"\bconsists of\b",
            r"\bfocuses on\b",
            r"\bbranch of\b",
            r"\bfield of\b"
        ]

        ranked_sentences = []

        for index, sentence in enumerate(
            matching_sentences
        ):

            sentence_lower = sentence.lower()

            score = float(similarities[index])

            # Reward sentences explaining the concept.
            if any(
                re.search(pattern, sentence_lower)
                for pattern in definition_patterns
            ):
                score += 0.5

            # Penalize chapter headings.
            if re.match(
                r"^(introduction to|chapter|unit|section)\b",
                sentence_lower
            ):
                score -= 1.0

            # Penalize sentences that look like page headings.
            if re.search(
                r"\b\d+\s*$",
                sentence_lower
            ):
                score -= 0.3

            # Prefer meaningful explanatory sentences.
            if len(sentence.split()) >= 8:
                score += 0.1

            ranked_sentences.append(
                (score, sentence)
            )

        ranked_sentences.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_sentence = ranked_sentences[0]

        if best_score < 0.05:

            return (
                f"No clear definition of "
                f"{target_concept.title()} was found in the notes."
            )

        return best_sentence

    except ValueError:

        # If TF-IDF cannot be calculated, return the first
        # sentence that explicitly defines the requested concept.
        for sentence in matching_sentences:

            if re.search(
                r"\b(is|are|refers to|defined as)\b",
                sentence.lower()
            ):
                return sentence

        return matching_sentences[0]


# =====================================================
# 4. CONCEPT-BASED MCQ GENERATION
# =====================================================

def generate_questions(text):

    text_lower = text.lower()

    questions = []

    templates = {

        "artificial intelligence": {
            "question": (
                "Artificial Intelligence is a field of "
                "computer science that focuses on creating "
                "systems capable of performing tasks that "
                "normally require ______."
            ),
            "answer": "human intelligence"
        },

        "machine learning": {
            "question": (
                "Which branch of Artificial Intelligence "
                "enables computers to learn patterns from data?"
            ),
            "answer": "Machine Learning"
        },

        "supervised learning": {
            "question": (
                "Which type of machine learning uses labelled data?"
            ),
            "answer": "Supervised Learning"
        },

        "unsupervised learning": {
            "question": (
                "Which type of machine learning uses unlabelled data?"
            ),
            "answer": "Unsupervised Learning"
        },

        "reinforcement learning": {
            "question": (
                "Which learning approach involves an agent "
                "interacting with an environment and receiving rewards?"
            ),
            "answer": "Reinforcement Learning"
        },

        "classification": {
            "question": (
                "Which supervised learning task predicts "
                "categorical outputs?"
            ),
            "answer": "Classification"
        },

        "regression": {
            "question": (
                "Which supervised learning task predicts "
                "continuous numerical values?"
            ),
            "answer": "Regression"
        },

        "deep learning": {
            "question": (
                "Which machine learning approach uses "
                "neural networks with multiple layers?"
            ),
            "answer": "Deep Learning"
        },

        "neural networks": {
            "question": (
                "Which computational models are inspired by "
                "the structure of the human brain?"
            ),
            "answer": "Neural Networks"
        },

        "natural language processing": {
            "question": (
                "Which AI field focuses on understanding "
                "and processing human language?"
            ),
            "answer": "Natural Language Processing"
        },

        "computer vision": {
            "question": (
                "Which AI field enables computers to "
                "interpret images and visual information?"
            ),
            "answer": "Computer Vision"
        }
    }

    available = [
        concept
        for concept in templates
        if re.search(
            r"\b" + re.escape(concept) + r"\b",
            text_lower
        )
    ]

    random.shuffle(available)

    for concept in available[:5]:

        question_data = templates[concept]

        answer = question_data["answer"]

        distractors = [
            other
            for other in templates
            if other != concept
        ]

        options = random.sample(
            distractors,
            min(3, len(distractors))
        )

        options = [
            option.title()
            for option in options
        ]

        options.append(answer)

        random.shuffle(options)

        questions.append({
            "question": question_data["question"],
            "options": options,
            "answer": answer
        })

    return questions


# =====================================================
# PDF UPLOAD AND MANUAL TEXT
# =====================================================

st.divider()

uploaded_file = st.file_uploader(
    "Upload your lecture notes (PDF)",
    type=["pdf"]
)

manual_text = st.text_area(
    "Or paste your lecture notes here",
    height=150,
    placeholder="Paste your lecture notes..."
)

lecture_text = ""

if uploaded_file:

    try:

        lecture_text = extract_text_from_pdf(
            uploaded_file
        )

        st.success(
            "PDF uploaded and text extracted successfully!"
        )

    except Exception as error:

        st.error(str(error))


if manual_text.strip():

    lecture_text = clean_lecture_text(manual_text)


if lecture_text:

    st.success(
        f"Lecture notes ready! Characters extracted: "
        f"{len(lecture_text)}"
    )

    with st.expander("Preview extracted lecture notes"):

        st.write(lecture_text[:3000])

else:

    st.info(
        "Upload a PDF or paste lecture notes to begin."
    )


# =====================================================
# APPLICATION TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Summary",
    "🔑 Keywords",
    "❓ Ask Questions",
    "📋 Generate MCQs",
    "🤖 AI Summary"
])


# =====================================================
# TAB 1: SUMMARY
# =====================================================

with tab1:

    st.header("Lecture Summary")

    if st.button("Generate Summary"):

        if lecture_text:

            with st.spinner("Generating summary..."):

                summary = summarize_notes(lecture_text)

            st.subheader("Generated Summary")

            st.write(summary)

        else:

            st.warning(
                "Please upload a PDF or paste lecture notes first."
            )


# =====================================================
# TAB 2: KEYWORDS
# =====================================================

with tab2:

    st.header("Important Keywords")

    if st.button("Extract Keywords"):

        if lecture_text:

            keywords = get_keywords(lecture_text)

            if keywords:

                st.subheader("Important Concepts")

                for keyword in keywords:

                    st.write("•", keyword.title())

            else:

                st.warning("No meaningful keywords found.")

        else:

            st.warning(
                "Please provide lecture notes first."
            )


# =====================================================
# TAB 3: QUESTION ANSWERING
# =====================================================

with tab3:

    st.header("Ask Questions From Your Notes")

    question = st.text_input(
        "Enter your question",
        placeholder="What is supervised learning?"
    )

    if st.button("Find Answer"):

        if lecture_text and question.strip():

            with st.spinner("Searching lecture notes..."):

                answer = answer_from_notes(
                    lecture_text,
                    question
                )

            st.subheader("Answer")

            st.info(answer)

        else:

            st.warning(
                "Please provide lecture notes and enter a question."
            )


# =====================================================
# TAB 4: MCQ GENERATION
# =====================================================

with tab4:

    st.header("Generate Multiple Choice Questions")

    if st.button("Generate MCQs"):

        if lecture_text:

            questions = generate_questions(lecture_text)

            if questions:

                st.subheader("Practice Questions")

                for index, item in enumerate(
                    questions,
                    start=1
                ):

                    st.write(
                        f"**Q{index}. {item['question']}**"
                    )

                    for option_index, option in enumerate(
                        item["options"]
                    ):

                        st.write(
                            f"{chr(65 + option_index)}. {option}"
                        )

                    with st.expander("Show correct answer"):

                        st.success(
                            f"Correct Answer: {item['answer']}"
                        )

            else:

                st.warning(
                    "No supported concepts were found "
                    "for MCQ generation. Try another lecture PDF."
                )

        else:

            st.warning(
                "Please provide lecture notes first."
            )


# =====================================================
# TAB 5: FLAN-T5 AI SUMMARY
# =====================================================

with tab5:

    st.header("FLAN-T5 AI Summarization")

    st.write(
        "This experimental feature uses the pretrained "
        "Google FLAN-T5 Small language model."
    )

    if st.button("Generate AI Summary"):

        if lecture_text:

            try:

                with st.spinner(
                    "Loading AI model and generating summary..."
                ):

                    import model

                    ai_summary = model.summarize_text(
                        lecture_text
                    )

                st.subheader("AI-Generated Summary")

                st.write(ai_summary)

                st.caption(
                    "Experimental output. Verify important "
                    "details against the original lecture notes."
                )

            except Exception as error:

                st.error(
                    f"AI model error: {error}"
                )

        else:

            st.warning(
                "Please provide lecture notes first."
            )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "LectureLens | AI-Powered Lecture Notes Assistant | "
    "Python | NLP | Streamlit | FLAN-T5"
)

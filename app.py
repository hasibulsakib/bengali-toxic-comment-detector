
import joblib
import gradio as gr
from scipy.sparse import hstack


# ==============================
# Load saved model and files
# ==============================

model = joblib.load("bengali_toxic_model.pkl")
word_vectorizer = joblib.load("word_tfidf_vectorizer.pkl")
char_vectorizer = joblib.load("char_tfidf_vectorizer.pkl")
threshold = joblib.load("best_threshold.pkl")


# Label names
labels = [
    "vulgar",
    "hate",
    "religious",
    "threat",
    "troll",
    "Insult"
]


# ==============================
# Prediction Function
# ==============================

def predict_toxicity(text):

    if not text or not text.strip():
        return (
            "⚠️ Please enter a comment.",
            "",
            ""
        )

    # Word TF-IDF
    word_features = word_vectorizer.transform([text])

    # Character TF-IDF
    char_features = char_vectorizer.transform([text])

    # Combine features
    features = hstack([
        word_features,
        char_features
    ])

    # Prediction probabilities
    probabilities = model.predict_proba(features)[0]

    # Apply threshold
    predictions = (
        probabilities >= threshold
    ).astype(int)

    # Detect labels
    detected = []

    for label, prediction in zip(
        labels,
        predictions
    ):
        if prediction == 1:
            detected.append(label)

    # Overall result
    if detected:

        result = "⚠️ Toxic Comment"

        detected_text = ", ".join(
            detected
        )

    else:

        result = "✅ Non-toxic Comment"

        detected_text = (
            "No toxic labels detected."
        )

    # Probability output
    probability_text = ""

    for label, probability in zip(
        labels,
        probabilities
    ):

        probability_text += (
            f"{label:<10}: "
            f"{probability:.2f}\n"
        )

    return (
        result,
        detected_text,
        probability_text
    )


# ==============================
# Gradio Interface
# ==============================

app = gr.Interface(

    fn=predict_toxicity,

    inputs=gr.Textbox(
        lines=5,
        placeholder="বাংলা কমেন্ট এখানে লিখুন...",
        label="Bengali Comment"
    ),

    outputs=[
        gr.Textbox(
            label="Result"
        ),

        gr.Textbox(
            label="Detected Labels"
        ),

        gr.Textbox(
            label="Label Probabilities"
        )
    ],

    title="🇧🇩 Bengali Toxic Comment Detector",

    description=(
        "A Machine Learning based system "
        "for detecting Bengali toxic comments."
    )
)


# ==============================
# Launch App
# ==============================

import os

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)

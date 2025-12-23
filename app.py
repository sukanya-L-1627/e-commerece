from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("model/risk_model.pkl")
encoders = joblib.load("model/encoders.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# -----------------------------
# Safe encoder
# -----------------------------
def safe_encode(value, encoder):
    return encoder.transform([value])[0] if value in encoder.classes_ else -1

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    form = request.form

    raw_input = {
        "availability": form["availability"],
        "brand_name": form["brand_name"],
        "breadcrumbs": form["category"],
        "seller_name": form["seller_name"],
        "list_price": float(form["list_price"]),
        "price_value": float(form["selling_price"]),
        "rating_count": int(form["rating_count"]),
        "rating_stars": float(form["product_rating"]),
        "avg_review_rating": float(form["avg_review_rating"]),
        "rating_variance": float(form["rating_variance"]),
        "review_volume": int(form["review_volume"]),
        "negative_review_ratio": float(form["negative_review_ratio"]),
        "avg_sentiment": float(form["avg_sentiment"]),
        "avg_review_length": float(form["avg_review_length"]),
    }

    # Encode input
    encoded = {}
    for col in feature_columns:
        if col in encoders:
            encoded[col] = safe_encode(raw_input[col], encoders[col])
        else:
            encoded[col] = raw_input[col]

    X = pd.DataFrame([encoded])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    selected_risk = risk_labels[pred]
    confidence = round(proba[pred] * 100, 1)

    # -----------------------------
    # Human Explanation (simple)
    # -----------------------------
    if selected_risk == "High Risk":
        explanation = [
            "Many customers are unhappy with this product.",
            "Negative feedback may reduce buyer trust.",
            "Selling this product now may impact brand reputation."
        ]
        recommended_action = [
            "Fix common customer complaints.",
            "Improve product quality or description.",
            "Delay promotions until reviews improve."
        ]

    elif selected_risk == "Medium Risk":
        explanation = [
            "Customer feedback is mixed.",
            "Some buyers are satisfied, while others face issues.",
            "The product needs attention but is not critical."
        ]
        recommended_action = [
            "Monitor new reviews closely.",
            "Address repeated customer issues.",
            "Improve customer support and communication."
        ]

    else:
        explanation = [
            "Most customers are satisfied.",
            "Reviews are consistent and positive.",
            "The product is performing well."
        ]
        recommended_action = [
            "Proceed with promotions.",
            "Maintain current quality standards.",
            "Use this product for marketing campaigns."
        ]

    return render_template(
        "result.html",
        risk=selected_risk,
        confidence=confidence,
        explanation=explanation,
        recommended_action=recommended_action
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

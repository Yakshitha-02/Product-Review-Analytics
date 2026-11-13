
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Product Review Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path.home() / "ProductReviewAnalytics"
MODEL_PATH = BASE_DIR / "models" / "final_sentiment_model.pkl"
RESULTS_DIR = BASE_DIR / "results"

# ==========================================
# THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263449;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #17263d, #1e3150);
    border: 1px solid #304765;
    padding: 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #9caec5;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
    font-weight: 700;
}

h1 {
    color: #f8fafc;
    font-weight: 800;
}

h2, h3 {
    color: #7dd3fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 22px;
}

div.stButton > button:hover {
    border: 1px solid #7dd3fc;
    color: white;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #29384e;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# DATA
# ==========================================

sentiment_data = pd.DataFrame({
    "Sentiment": ["Positive", "Negative", "Neutral"],
    "Reviews": [6032198, 1118187, 470215]
})

model_data = pd.DataFrame({
    "Model": ["Naive Bayes", "Logistic Regression", "Random Forest"],
    "Accuracy": [68.15, 68.13, 62.93],
    "Macro F1": [68.23, 68.02, 63.06]
})

# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["vectorizer"]

try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error(f"Model loading failed: {error}")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("# 📊 Review Analytics")
    st.caption("BIG DATA • HADOOP • MACHINE LEARNING")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Hadoop Analysis",
            "Model Evaluation",
            "Live Prediction"
        ]
    )

    st.divider()

    st.markdown("### PROJECT DETAILS")
    st.caption("Dataset: Amazon Reviews 2023")
    st.caption("Processing: Apache Hadoop")
    st.caption("Classification: Logistic Regression")

    if model_loaded:
        st.success("ML Model Loaded")
    else:
        st.error("ML Model Unavailable")

    st.divider()
    st.caption("Big Data Analytics Project • 2026")

# ==========================================
# PLOTLY THEME
# ==========================================

plot_layout = {
    "paper_bgcolor": "#111c2e",
    "plot_bgcolor": "#111c2e",
    "font": {"color": "#e5e7eb"},
    "margin": {"l": 30, "r": 20, "t": 50, "b": 30}
}

# ==========================================
# OVERVIEW
# ==========================================

if page == "Overview":

    st.title("Amazon Product Review Analytics")
    st.markdown(
        "### Large-Scale Sentiment Classification "
        "using Hadoop and Machine Learning"
    )

    st.caption(
        "An analytical overview of large-scale product reviews "
        "and customer sentiment."
    )

    st.divider()

    st.subheader("📌 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Reviews Processed", "7.62 Million")
    c2.metric("Dataset Size", "1.45 GB")
    c3.metric("Product Groups", "767,881")
    c4.metric("ML Training Sample", "30,000")

    st.divider()

    st.subheader("📈 Sentiment Distribution")

    left, right = st.columns([1.2, 1])

    with left:
        fig = px.pie(
            sentiment_data,
            names="Sentiment",
            values="Reviews",
            hole=0.62,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Neutral": "#f59e0b"
            },
            title="Distribution of 7.62 Million Reviews"
        )

        fig.update_layout(
            **plot_layout,
            legend_title="Sentiment",
            height=390
        )

        fig.update_traces(
            textinfo="percent",
            textfont_size=14
        )

        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("#### Sentiment Statistics")

        for _, row in sentiment_data.iterrows():
            percentage = row["Reviews"] / 7620600 * 100

            st.markdown(f"**{row['Sentiment']}**")
            st.progress(percentage / 100)
            st.caption(
                f"{row['Reviews']:,} reviews • {percentage:.2f}%"
            )

        st.info(
            "Labels are derived from ratings: "
            "1–2 stars = Negative, 3 stars = Neutral, "
            "4–5 stars = Positive."
        )

    st.divider()

    st.subheader("⚙️ Project Technology Stack")

    a, b, c, d = st.columns(4)

    a.info("🐘 Apache Hadoop\n\nDistributed data processing")
    b.info("⚡ Apache Spark\n\nLarge-scale data preparation")
    c.info("🐍 Python\n\nMachine learning")
    d.info("📊 Streamlit\n\nInteractive dashboard")

# ==========================================
# HADOOP ANALYSIS
# ==========================================

elif page == "Hadoop Analysis":

    st.title("Hadoop & MapReduce Analysis")
    st.caption("Large-scale product review aggregation")

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("Map Input Records", "7,620,600")
    c2.metric("Reduce Input Groups", "767,881")
    c3.metric("Reduce Output Records", "767,881")

    st.divider()

    st.subheader("🔄 MapReduce Architecture")

    stages = [
        ("01", "Input", "Review records stored in HDFS"),
        ("02", "Mapper", "Extracts product ID and rating"),
        ("03", "Shuffle & Sort", "Groups ratings by product ID"),
        ("04", "Reducer", "Calculates review count and average rating"),
        ("05", "Output", "Stores product-wise aggregation in HDFS")
    ]

    for number, title, description in stages:
        with st.container(border=True):
            col1, col2 = st.columns([1, 8])
            col1.markdown(f"## {number}")
            col2.markdown(f"**{title}**")
            col2.caption(description)

    st.success("Hadoop MapReduce job completed successfully.")

    st.divider()

    st.subheader("📊 Processing Details")

    details = pd.DataFrame({
        "Parameter": [
            "Map Tasks",
            "Reduce Tasks",
            "Map Input Records",
            "Map Output Records",
            "Reduce Input Records",
            "Reduce Output Records",
            "Reduce Groups"
        ],
        "Value": [
            "4",
            "1",
            "7,620,600",
            "7,620,600",
            "7,620,600",
            "767,881",
            "767,881"
        ]
    })

    st.dataframe(details, width="stretch", hide_index=True)

    st.caption(
        "The Hadoop job aggregates review counts and average ratings "
        "for each product."
    )

# ==========================================
# MODEL EVALUATION
# ==========================================

elif page == "Model Evaluation":

    st.title("Machine Learning Model Evaluation")
    st.caption(
        "Comparison, evaluation and final model performance"
    )

    st.divider()

    st.subheader("🏆 Final Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Accuracy", "74.27%")
    c2.metric("Macro Precision", "74.43%")
    c3.metric("Macro Recall", "74.27%")
    c4.metric("Macro F1 Score", "74.32%")

    st.divider()

    st.subheader("📊 Initial Model Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Accuracy",
        x=model_data["Model"],
        y=model_data["Accuracy"],
        marker_color="#38bdf8"
    ))

    fig.add_trace(go.Bar(
        name="Macro F1",
        x=model_data["Model"],
        y=model_data["Macro F1"],
        marker_color="#a78bfa"
    ))

    fig.update_layout(
        **plot_layout,
        barmode="group",
        title="Baseline Model Performance",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        height=430
    )

    st.plotly_chart(fig, width="stretch")

    st.dataframe(
        model_data,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "These are the initial baseline results. "
        "Further TF-IDF experiments improved performance."
    )

    st.divider()

    st.subheader("🎯 Final Confusion Matrix")

    matrix_path = RESULTS_DIR / "final_confusion_matrix.png"

    if matrix_path.exists():
        st.image(
            str(matrix_path),
            caption="Final model confusion matrix",
            width="stretch"
        )
    else:
        st.warning("Confusion matrix image not found.")

    st.divider()

    st.subheader("Final Model Configuration")

    st.markdown("""
    - **Algorithm:** Logistic Regression
    - **Feature extraction:** Combined Word + Character TF-IDF
    - **Regularization parameter:** C = 0.25
    - **Training records:** 25,500
    - **Testing records:** 4,500
    - **Classes:** Negative, Neutral and Positive
    """)

# ==========================================
# LIVE PREDICTION
# ==========================================

elif page == "Live Prediction":

    st.title("Live Sentiment Prediction")
    st.caption("Analyze the sentiment of an Amazon product review")

    st.divider()

    st.subheader("✍️ Enter Product Review")

    review = st.text_area(
        "Review text",
        placeholder=(
            "Example: This product is amazing! "
            "The quality is excellent..."
        ),
        height=160
    )

    if st.button("🔍 Analyze Sentiment", width="stretch"):

        if not review.strip():
            st.warning("Please enter a product review.")

        elif not model_loaded:
            st.error("The trained model is unavailable.")

        else:
            features = vectorizer.transform([review])

            prediction = model.predict(features)[0]

            probabilities = model.predict_proba(features)[0]
            classes = model.classes_

            st.divider()

            st.subheader("Prediction Result")

            if prediction == "Positive":
                st.success("😊 Positive Review")
            elif prediction == "Negative":
                st.error("😞 Negative Review")
            else:
                st.warning("😐 Neutral Review")

            probability_data = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })

            fig = px.bar(
                probability_data,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#10b981",
                    "Negative": "#ef4444",
                    "Neutral": "#f59e0b"
                },
                title="Model-Estimated Class Probabilities"
            )

            fig.update_layout(
                **plot_layout,
                yaxis_range=[0, 1],
                height=380
            )

            st.plotly_chart(fig, width="stretch")

            st.caption(
                "Probabilities are model estimates, "
                "not necessarily calibrated confidence scores."
            )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center style='color:#8293aa'>"
    "Large-Scale Product Review Analytics | "
    "Apache Hadoop + Machine Learning | 2026"
    "</center>",
    unsafe_allow_html=True
)

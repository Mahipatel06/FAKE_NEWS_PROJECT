import pickle
import pandas as pd
from matplotlib import pyplot as plt
from vectorizer import(
    plot_label_distribution,
    plot_article_length_histogram,
    plot_label_pie_chart
)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorize = pickle.load(f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(model, f)

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown(

    """
    <style>
    .stApp {
        background-color: #f2f2f2;
        }
        </style>
        """,
        unsafe_allow_html=True
)    

st.title("🧠 Fake & Real News Detector")

fake_file = st.file_uploader("📂 Upload Fake News CSV", type=["csv"])
true_file = st.file_uploader("📂 Upload Real News CSV", type=["csv"])

if fake_file and true_file:
    fake_df = pd.read_csv(fake_file)
    true_df = pd.read_csv(true_file)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    fake_count = len(fake_df)
    real_count = len(true_df)

    st.subheader("📊 Fake vs Real News Count")
    st.pyplot(plot_label_distribution(fake_count, real_count))

    st.subheader("📎 Fake vs Real Percentage (Pie Chart)")
    st.pyplot(plot_label_pie_chart(fake_count, real_count))

    st.subheader("📉 Article Length Distribution")
    st.pyplot(plot_article_length_histogram(df))


st.header("🔍 Check News Article")
text = st.text_area("Paste your news article here:")


if st.button("Predict"):
    if text:
        vect_text = vectorize.transform([text])
        pred = model.predict(vect_text)[0]
        st.subheader("🧾 Result:")
        st.success("✅ Real News") if pred == 1 else st.error("❌ Fake News")
    else:
        st.warning("Please enter a news article.")  

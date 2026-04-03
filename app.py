import streamlit as st
from src.predict import analyze_complaint

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title= "AI Complaint Intelligence System",
    page_icon= "🤖",
    layout= "centered",
)


# -----------------------------
# Title
# -----------------------------

st.title("🤖 AI Complaint Intelligence System")
st.markdown("Analyze Customer Complaints using AI")

# -----------------------------
# Input Box
# -----------------------------

user_input= st.text_area(
    "Enter Customer Complaint",
    height= 150,
    placeholder= "e.g. I didn't receive my refund and this is the worst service ever"
)

# ------------------------------
# Button
# -----------------------------

if st.button("Analyze Complaint"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint")
    else:
        result= analyze_complaint(user_input)

        st.success("Analysis Complete ✅")

        # -----------------------------
        # Output Display
        # -----------------------------
        st.subheader("📊 Analysis Result")

        st.write(f"**Category:** {result['category']}")
        st.write(f"**Sentiment:** {result['sentiment']}")
        st.write(f"**Urgency:** {result['urgency']}")
        st.write(f"**Risk:** {result['risk']}")

        st.subheader("💬 Suggested Reply")
        st.write(f"**Suggested Reply:** {result['reply']}")

from cProfile import label
import html
from sklearn.metrics import r2_score
import streamlit as st
from src.features import extract_features, find_suspicious_url_patterns
from src.predictor import predict_phishing

# ---------------------------------------------------------
# Page setup and simple styling
# ---------------------------------------------------------
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
    .result-card {
        padding: 18px 22px;
        border-radius: 10px;
        border-left: 8px solid #888;
        background-color: rgba(128, 128, 128, 0.12);
        margin-bottom: 10px;
    }
    .result-card h2 { margin: 0; font-size: 1.8rem; }
    .result-card p { margin: 4px 0 0 0; opacity: 0.8; }
    .indicator {
        padding: 10px 14px;
        border-radius: 8px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-bottom: 8px;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Functions (each one does a single job)
# ---------------------------------------------------------
def find_suspicious_indicators(email_text, features):
    """Builds the list of suspicious indicators using the numbers from src/features.py.
    Each indicator is a dictionary with a 'title' and a 'detail'."""
    findings = []

    if features["num_urgency_words"] > 0:
        findings.append({
            "title": "Urgent or pressuring language",
            "detail": str(features["num_urgency_words"]) + " urgency word(s) found",
        })

    if features["num_suspicious_keywords"] > 0:
        findings.append({
            "title": "Suspicious keywords (password, verify, bank, ...)",
            "detail": str(features["num_suspicious_keywords"]) + " keyword(s) found",
        })

    if features["num_urls"] > 0:
        findings.append({
            "title": "Email contains links",
            "detail": str(features["num_urls"]) + " link(s), longest is "
                      + str(features["max_url_length"]) + " characters",
        })

    # one indicator for every suspicious link pattern
    url_reasons = find_suspicious_url_patterns(email_text)
    for reason in url_reasons:
        findings.append({"title": "Suspicious link pattern", "detail": reason})

    if features["num_exclamation_marks"] >= 3:
        findings.append({
            "title": "Excessive exclamation marks",
            "detail": str(features["num_exclamation_marks"]) + " exclamation marks",
        })

    if features["email_length"] > 0:
        capital_share = features["num_capital_letters"] / features["email_length"]
        if capital_share > 0.3:
            findings.append({
                "title": "Excessive capital letters",
                "detail": str(round(capital_share * 100)) + "% of the text is in capitals",
            })

    return findings


def predict_email(email_text, features, indicators):
    label, r2_score = predict_phishing(email_text)

    return label, r2_score


def get_risk_level(risk_score):
    """Turns the number into Low / Medium / High with a color."""
    if risk_score >= 70:
        return "High", "#d93025"
    elif risk_score >= 30:
        return "Medium", "#f29900"
    else:
        return "Low", "#188038"


# ---------------------------------------------------------
# Page content
# ---------------------------------------------------------
st.title("🛡️ AI-Based Email Phishing Detection and Analysis Platform")
st.write(
    "Paste the text of an email below. The platform analyzes it, predicts whether it is "
    "**phishing or legitimate**, gives a **risk score**, and highlights the "
    "**suspicious indicators** it found."
)

email_text = st.text_area(
    "Email text",
    height=300,
    placeholder="Paste the full email (subject and body) here...",
)

analyze_clicked = st.button("Analyze Email", type="primary")

if analyze_clicked:

    if email_text.strip() == "":
        st.warning("Please paste an email before clicking Analyze Email.")
    else:
        features = extract_features(email_text)
        indicators = find_suspicious_indicators(email_text, features)
        label, risk_score = predict_email(email_text, features, indicators)
        risk_level, risk_color = get_risk_level(risk_score)

        st.divider()

        # ----- Prediction section -----
        st.subheader("Prediction")
        if label == "Phishing":
            card_color = "#d93025"
            card_note = "This email shows signs of a phishing attempt."
        else:
            card_color = "#188038"
            card_note = "No strong phishing signs were detected."

        st.markdown(
            '<div class="result-card" style="border-left-color:' + card_color + '">'
            '<h2 style="color:' + card_color + '">' + label + '</h2>'
            '<p>' + card_note + '</p></div>',
            unsafe_allow_html=True,
        )

        # ----- Risk score section -----
        st.subheader("Risk Score")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk score", str(risk_score) + " / 100")
        with col2:
            st.markdown(
                '<div style="padding-top:6px">Risk level</div>'
                '<div style="font-size:1.8rem; font-weight:600; color:' + risk_color + '">'
                + risk_level + '</div>',
                unsafe_allow_html=True,
            )
        st.progress(risk_score / 100)

        # ----- Suspicious indicators section -----
        st.subheader("Suspicious Indicators")
        if len(indicators) == 0:
            st.success("No suspicious indicators found.")
        else:
            st.write(str(len(indicators)) + " indicator(s) found:")
            for item in indicators:
                # html.escape makes sure text from the email can never break the page
                st.markdown(
                    '<div class="indicator"><b>' + html.escape(item["title"]) + '</b><br>'
                    + html.escape(item["detail"]) + '</div>',
                    unsafe_allow_html=True,
                )

        # ----- Extracted features (useful for the team and for demos) -----
        with st.expander("View extracted features"):
            st.json(features)

st.caption("College project - AI-Based Email Phishing Detection and Analysis Platform")
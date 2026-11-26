"""
features.py - feature extraction for the AI-Based Email Phishing Detection
and Analysis Platform.

Every function takes the email text (a string) and returns a number or a list.
They are kept simple (loops and if-statements) so the whole team can read them.

IMPORTANT: always pass the ORIGINAL email text, not the cleaned/normalized text.
The cleaning step removes URLs, capital letters and punctuation, which are exactly
the things these functions need to look at.

How to use it in another file (for example the Streamlit app, with app.py in
the project root folder):

    from src.features import extract_features, find_suspicious_url_patterns

    features = extract_features(email_text)             # dictionary of numbers
    reasons = find_suspicious_url_patterns(email_text)  # list of text reasons
"""

from urllib.parse import urlparse
import pandas as pd


# ---------------------------------------------------------
# Word lists (edit these to add or remove words)
# ---------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "verify", "password", "login", "account", "suspended", "locked", "bank",
    "credit card", "ssn", "otp", "confirm", "security", "click here", "winner",
    "prize", "refund", "payment", "invoice", "gift card", "unauthorized",
    "restricted", "update your",
]

URGENCY_WORDS = [
    "urgent", "immediately", "asap", "act now", "final notice", "within 24 hours",
    "expires", "last chance", "limited time", "hurry", "today only", "right away",
    "time sensitive", "immediate action",
]

URL_SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"]

SUSPICIOUS_ENDINGS = [".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".click", ".work", ".zip"]

SENSITIVE_URL_WORDS = ["login", "verify", "secure", "update", "account", "confirm", "password", "banking"]


# ---------------------------------------------------------
# URL features
# ---------------------------------------------------------
def extract_urls(email_text):
    """Finds all links in the email and returns them as a list.

    How it works: split the email into words, and keep every word that starts
    with http://, https:// or www. (after removing punctuation like . , ) around it).
    Example: "Click https://abc.com/login now." -> ["https://abc.com/login"]
    """
    email_text = str(email_text)
    urls = []

    words = email_text.split()
    for word in words:
        cleaned_word = word.strip('.,;:!?()[]<>"\'')
        lower_word = cleaned_word.lower()
        if lower_word.startswith("http://") or lower_word.startswith("https://") or lower_word.startswith("www."):
            urls.append(cleaned_word)

    return urls


def count_urls(email_text):
    """Returns how many links the email contains (phishing emails often push a link)."""
    return len(extract_urls(email_text))


def get_average_url_length(email_text):
    """Returns the average length (in characters) of the links. 0 if there are no links."""
    urls = extract_urls(email_text)
    if len(urls) == 0:
        return 0

    total_length = 0
    for url in urls:
        total_length = total_length + len(url)
    return round(total_length / len(urls), 1)


def get_longest_url_length(email_text):
    """Returns the length of the longest link. 0 if there are no links.
    Very long links are a common trick to hide the real address."""
    urls = extract_urls(email_text)

    longest = 0
    for url in urls:
        if len(url) > longest:
            longest = len(url)
    return longest


def find_suspicious_url_patterns(email_text):
    """Checks every link for known suspicious patterns.

    Returns a LIST of short reasons (text), e.g.
        ["Link is not secure (http, not https): http://abc.com"]
    An empty list means nothing suspicious was found.
    The list is handy for showing 'Suspicious Indicators' in the Streamlit app.
    """
    urls = extract_urls(email_text)
    reasons = []

    for url in urls:
        lower_url = url.lower()

        # get the domain part (host) of the link, e.g. "www.abc.com"
        if "://" in lower_url:
            link_for_parsing = lower_url
        else:
            link_for_parsing = "http://" + lower_url
        host = urlparse(link_for_parsing).netloc
        host = host.split("@")[-1]   # ignore anything before an @ sign
        host = host.split(":")[0]    # ignore the port number

        # 1. not secure
        if lower_url.startswith("http://"):
            reasons.append("Link is not secure (http, not https): " + url)

        # 2. IP address instead of a domain name (like http://192.168.1.5/login)
        if host.count(".") == 3 and host.replace(".", "").isdigit():
            reasons.append("Link uses an IP address instead of a domain name: " + url)

        # 3. @ symbol inside the link (can hide the real destination)
        if "@" in lower_url:
            reasons.append("Link contains an @ symbol: " + url)

        # 4. too many dots in the domain (e.g. login.secure.bank.com.fake.xyz)
        if host.count(".") >= 4:
            reasons.append("Link has too many subdomains: " + url)

        # 5. many hyphens in the domain (e.g. secure-bank-login-update.com)
        if host.count("-") >= 2:
            reasons.append("Link domain has many hyphens: " + url)

        # 6. URL shortener (hides the real destination)
        for shortener in URL_SHORTENERS:
            if host == shortener or host.endswith("." + shortener):
                reasons.append("Link uses a URL shortener (" + shortener + "): " + url)

        # 7. domain ending often used in scams
        for ending in SUSPICIOUS_ENDINGS:
            if host.endswith(ending):
                reasons.append("Link ends with a suspicious domain (" + ending + "): " + url)

        # 8. very long link
        if len(url) > 75:
            reasons.append("Link is very long: " + url)

        # 9. sensitive words inside the link (login, verify, ...)
        found_words = []
        for word in SENSITIVE_URL_WORDS:
            if word in lower_url:
                found_words.append(word)
        if len(found_words) > 0:
            reasons.append("Link contains sensitive words (" + ", ".join(found_words) + "): " + url)

    return reasons


def count_suspicious_url_patterns(email_text):
    """Returns HOW MANY suspicious link patterns were found (the length of the list above)."""
    return len(find_suspicious_url_patterns(email_text))


# ---------------------------------------------------------
# Keyword features
# ---------------------------------------------------------
def count_suspicious_keywords(email_text):
    """Counts how many times words from SUSPICIOUS_KEYWORDS appear in the email
    (verify, password, bank, ...). Not case-sensitive."""
    lower_text = str(email_text).lower()

    total = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        total = total + lower_text.count(keyword)
    return total


def count_urgency_words(email_text):
    """Counts how many times words from URGENCY_WORDS appear in the email
    (urgent, immediately, act now, ...). Phishing tries to make you act in a hurry."""
    lower_text = str(email_text).lower()

    total = 0
    for word in URGENCY_WORDS:
        total = total + lower_text.count(word)
    return total


# ---------------------------------------------------------
# Character and length features
# ---------------------------------------------------------
def count_special_characters(email_text):
    """Counts characters that are not a letter, a number or a space (! $ % @ # ...)."""
    email_text = str(email_text)

    count = 0
    for character in email_text:
        if not character.isalnum() and not character.isspace():
            count = count + 1
    return count


def count_capital_letters(email_text):
    """Counts capital letters (A-Z). SHOUTING IN CAPITALS is a common phishing style."""
    email_text = str(email_text)

    count = 0
    for character in email_text:
        if character.isupper():
            count = count + 1
    return count


def count_exclamation_marks(email_text):
    """Counts the ! characters (extra feature: phishing emails use them to add pressure)."""
    return str(email_text).count("!")


def get_email_length(email_text):
    """Returns the email length in characters."""
    return len(str(email_text))


def get_word_count(email_text):
    """Returns the number of words in the email (extra feature)."""
    return len(str(email_text).split())


# ---------------------------------------------------------
# Put everything together
# ---------------------------------------------------------
def extract_features(email_text):
    """Runs all the feature functions on ONE email and returns a dictionary.

    This is the main function other team members should call.
    Example result:
        {"num_urls": 1, "avg_url_length": 32.0, "max_url_length": 32, ...}
    """
    # a missing email (None or NaN) is treated as an empty email
    if pd.isna(email_text):
        email_text = ""
    email_text = str(email_text)

    features = {
        "num_urls": count_urls(email_text),
        "avg_url_length": get_average_url_length(email_text),
        "max_url_length": get_longest_url_length(email_text),
        "num_suspicious_url_patterns": count_suspicious_url_patterns(email_text),
        "num_suspicious_keywords": count_suspicious_keywords(email_text),
        "num_urgency_words": count_urgency_words(email_text),
        "num_special_chars": count_special_characters(email_text),
        "num_capital_letters": count_capital_letters(email_text),
        "num_exclamation_marks": count_exclamation_marks(email_text),
        "email_length": get_email_length(email_text),
        "word_count": get_word_count(email_text),
    }
    return features


def build_feature_table(email_list):
    """Runs extract_features on MANY emails and returns a pandas DataFrame
    (one row per email, one column per feature). Use this for model training.

    Example:
        feature_table = build_feature_table(data['text'])   # data = the RAW dataset
    """
    rows = []
    for email in email_list:
        rows.append(extract_features(email))
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Quick demo: run  python src/features.py  to see it working
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_email = (
        "Dear customer, URGENT! Your account has been suspended. "
        "Verify your password immediately at http://secure-login-update.example.xyz/verify "
        "or your account will be closed!!!"
    )

    print("FEATURES:")
    sample_features = extract_features(sample_email)
    for name, value in sample_features.items():
        print(" ", name, ":", value)

    print()
    print("SUSPICIOUS URL PATTERNS:")
    for reason in find_suspicious_url_patterns(sample_email):
        print("  -", reason)
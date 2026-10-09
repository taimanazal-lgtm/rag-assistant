"""Text cleaning functions for the ArXiv corpus."""
import html
import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # makes language detection deterministic

# Only real HTML tags, so math like "x < y" is not touched
HTML_TAG = re.compile(
    r"</?(?:p|br|b|i|em|strong|a|span|div|sub|sup|ul|ol|li|h[1-6])\b[^>]*>",
    re.IGNORECASE,
)


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities like &amp;."""
    return html.unescape(HTML_TAG.sub(" ", text))


def normalize_unicode(text: str) -> str:
    """NFKC normalization + remove invisible characters."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\u200b-\u200d\ufeff]", "", text)


def clean_latex(text: str) -> str:
    """Remove LaTeX markup but keep the readable content."""
    # \emph{word} -> word
    text = re.sub(
        r"\\(?:emph|textbf|textit|texttt|text|mathrm|mathbf|mathcal)\{([^{}]*)\}",
        r"\1",
        text,
    )
    # $x^2$ -> x^2 (keep the content, drop the $ signs)
    text = re.sub(r"(?<!\\)\$([^$]*)\$", r"\1", text)
    # \% -> %   \& -> &   \_ -> _
    text = re.sub(r"\\([%&_#${}])", r"\1", text)
    # remaining commands like \times
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return re.sub(r"[{}]", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse tabs/newlines/multiple spaces into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """Return True if the text is detected as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """Full cleaning pipeline for one text."""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_latex(text)
    return normalize_whitespace(text)
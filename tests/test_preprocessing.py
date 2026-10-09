from src.preprocessing import (
    clean_latex,
    clean_text,
    is_english,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
)


def test_strip_html_removes_tags_and_entities():
    result = normalize_whitespace(strip_html("<p>Hello <b>world</b> &amp; all</p>"))
    assert result == "Hello world & all"


def test_strip_html_keeps_math_comparisons():
    text = "if x < y and y > z"
    assert strip_html(text) == text


def test_normalize_unicode_ligature_and_invisible_chars():
    assert normalize_unicode("\ufb01ne") == "fine"       # ligature -> "fi"
    assert normalize_unicode("a\u200bb") == "ab"          # zero-width space


def test_clean_latex_escaped_percent():
    assert clean_latex(r"50\% time horizon") == "50% time horizon"


def test_clean_latex_emph():
    assert clean_latex(r"\emph{converts} time") == "converts time"


def test_clean_latex_math_mode():
    assert normalize_whitespace(clean_latex(r"a $10 \times$ jump")) == "a 10 jump"


def test_normalize_whitespace():
    assert normalize_whitespace("a \n\t b   c ") == "a b c"


def test_is_english_true():
    assert is_english("This paper proposes a new method for training neural networks.")


def test_is_english_false_for_other_languages():
    assert not is_english("Ceci est une phrase écrite en français pour tester la détection.")
    assert not is_english("هذه جملة مكتوبة باللغة العربية لاختبار اكتشاف اللغة")


def test_is_english_empty_string():
    assert is_english("") is False


def test_clean_text_full_pipeline():
    raw = r"METR's 50\% horizon, with \emph{converts}  and $10 \times$ jump"
    assert clean_text(raw) == "METR's 50% horizon, with converts and 10 jump"


def test_clean_text_is_idempotent():
    raw = r"<b>Hi</b>  50\% \emph{x}"
    once = clean_text(raw)
    assert clean_text(once) == once
    
def test_clean_latex_escaped_braces():
    assert clean_latex(r"values in \{p,q\}") == "values in p,q"

def test_clean_text_latex_thin_space():
    assert clean_text(r"0.40\,m") == "0.40 m"


def test_clean_text_backslash_space():
    assert clean_text(r"vs.\ 42") == "vs. 42"


def test_clean_text_inline_math_delimiters():
    assert clean_text(r"we prove \(O(T)\) regret") == "we prove O(T) regret"


def test_clean_text_keeps_currency():
    assert clean_text(r"cost \$5,694 total") == "cost $5,694 total"
    
def test_clean_text_line_break():
    assert clean_text(r"Operators:\\ Determines") == "Operators: Determines"


def test_clean_text_optional_hyphen():
    assert clean_text(r"al\-ter\-na\-ting") == "alternating"


def test_clean_text_norm_bars():
    assert clean_text(r"\|x\|^2") == "|x|^2"


def test_clean_text_stray_backslash():
    assert clean_text(r"at most 1\.23") == "at most 1.23"
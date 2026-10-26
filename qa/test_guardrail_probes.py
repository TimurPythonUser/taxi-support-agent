"""Deterministic guardrails against realistic customer wording. IDs match qa/PLAN.md."""

import pytest

from taxi_agent.agent import guardrails

# --- price output guard ------------------------------------------------------------------------


@pytest.mark.major
@pytest.mark.parametrize(
    "reply",
    [
        "The Business ride will cost 1,150 UAH (estimate, valid for 10 minutes).",
        "Business: UAH 1,150, valid for 10 minutes.",
    ],
)
def test_p01_thousands_separator_is_read_as_one_amount(reply):
    """P-01: "1,150 UAH" is parsed as 1 → a real quote is treated as invented and replaced by the fallback."""
    assert guardrails.unverified_amounts(reply, {1150}) == set()


@pytest.mark.minor
@pytest.mark.parametrize("reply", ["Поездка 25.10 245 грн, Крещатик → вокзал", "Заказ 731842\n245 грн"])
def test_p02_neighbouring_numbers_are_not_glued_into_one_amount(reply):
    """P-02: `\\s` in the amount regex glues a date or an order number to the price (10245, 1842245)."""
    assert guardrails.mentioned_amounts(reply) == {245}


@pytest.mark.major
@pytest.mark.parametrize(
    "reply",
    [
        "Поездка обойдётся примерно в двести гривен.",
        "It will be about 300 hryvnias.",
    ],
)
def test_p03_invented_price_in_words_or_english_currency_is_caught(reply):
    """P-03: amounts in words and the English currency name slip past the guard."""
    assert guardrails.unverified_amounts(reply, set())


@pytest.mark.minor
def test_p04_invented_price_without_currency_is_caught():
    """P-04: a bare number in a price sentence is not seen at all."""
    assert guardrails.unverified_amounts("Поездка будет стоить около 300, точнее скажу после расчёта.", set())


# --- safety keywords ---------------------------------------------------------------------------


@pytest.mark.major
@pytest.mark.parametrize(
    "text",
    [
        "Забыл в такси складной нож в чехле",
        "Сын забыл в машине игрушечный пистолет",
        "Я вчера был немного пьяный и забыл в такси куртку",
        "Водитель 15 минут стоял с включённой аварийкой и не подъезжал",
        "Ехал из травмпункта, забыл в машине пакет",
        "I left my knife sharpener in the car",
    ],
)
def test_s01_lost_items_and_plain_complaints_are_not_safety_alarms(text):
    """S-01: a false alarm freezes the dialog with an URGENT handoff instead of a normal lost-item ticket."""
    assert not guardrails.is_safety_incident(text)


@pytest.mark.major
@pytest.mark.parametrize(
    "text",
    [
        "Водій був п’яний",  # U+2019: the apostrophe phone keyboards actually type
        "Водитель пристаёт ко мне",
        "Водитель лапает меня",
        "Водитель закрыл двери и не выпускает меня из машины",
        "Водитель напился и гонит",
        "Водитель сбил пешехода",
        "We crashed on the way to the airport",
        "The driver touched me",
    ],
)
def test_s02_obvious_incidents_are_detected_without_the_model(text):
    """S-02: present tense, slang, typographic apostrophe and plain English miss the deterministic check."""
    assert guardrails.is_safety_incident(text)


# --- human request -----------------------------------------------------------------------------


@pytest.mark.major
@pytest.mark.parametrize(
    "text",
    [
        "Не хочу оператора, давайте разберёмся сами",
        "Не надо соединять меня с оператором",
        "Не переключайте на оператора, я просто спросить",
    ],
)
def test_h01_negated_request_is_not_a_handoff(text):
    """H-01: negation is ignored → the session is frozen and handed off against the customer's will."""
    assert not guardrails.wants_human(text)


@pytest.mark.minor
@pytest.mark.parametrize(
    "text",
    [
        "Дайте контакты этого человека, он забрал мою сумку",
        "Хочу оставить отзыв о менеджере",
    ],
)
def test_h02_mentions_of_people_are_not_handoff_requests(text):
    """H-02: "дайте … человека" / "хочу … менеджере" within three words trigger a handoff."""
    assert not guardrails.wants_human(text)


@pytest.mark.minor
@pytest.mark.parametrize(
    "text",
    [
        "З’єднайте з оператором",  # U+2019, the ASCII variant is covered by tests/
        "Дай оператора",
        "Хочу с человеком поговорить",
        "human please",
        "I need a human",
    ],
)
def test_h03_plain_requests_for_a_human_are_detected(text):
    """H-03: common phrasings fall through to the model (works only if the model obeys the prompt)."""
    assert guardrails.wants_human(text)


# --- language ----------------------------------------------------------------------------------


@pytest.mark.major
@pytest.mark.parametrize(
    ("text", "previous", "expected"),
    [
        ("Забыла iPhone 15 Pro Max", "ru", "ru"),
        ("Наушники AirPods Pro", "ru", "ru"),
        ("Добре, чекаю", "uk", "uk"),
        ("Так, правильно", "uk", "uk"),
    ],
)
def test_l01_language_does_not_flip_on_brand_names_or_short_ukrainian(text, previous, expected):
    """L-01: brand names switch a Russian dialog to English; short Ukrainian replies switch it to Russian.
    The state block then tells the model the wrong language and fixed replies come in it."""
    assert guardrails.detect_language(text, previous) == expected


@pytest.mark.minor
def test_l02_russian_will_be_kind_is_not_ukrainian():
    """L-02: the marker word "будь" turns a Russian message into Ukrainian."""
    assert guardrails.detect_language("Будь добр, подскажи цену до вокзала", "ru") == "ru"

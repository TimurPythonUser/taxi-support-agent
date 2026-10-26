import pytest

from taxi_agent.agent import guardrails


@pytest.mark.parametrize(
    "text",
    [
        "Соедините меня с оператором",
        "позовите живого человека",
        "Хочу поговорить с оператором",
        "дайте оператора",
        "оператор",
        "З'єднайте з оператором, будь ласка",
        "I want to talk to a human",
        "can I speak with a real person?",
    ],
)
def test_human_request_detected(text):
    assert guardrails.wants_human(text)


@pytest.mark.parametrize(
    "text",
    [
        "Хочу пожаловаться на водителя",
        "оператор вчера обещал перезвонить, но не перезвонил",
        "Сколько стоит поездка?",
        "Я забыл телефон в машине",
    ],
)
def test_no_false_human_request(text):
    assert not guardrails.wants_human(text)


@pytest.mark.parametrize(
    "text",
    [
        "Водитель был пьяный",
        "Мы попали в ДТП",
        "он мне угрожал",
        "водій погрожував мені",
        "the driver was drunk",
        "водитель приставал ко мне",
    ],
)
def test_safety_detected(text):
    assert guardrails.is_safety_incident(text)


@pytest.mark.parametrize("text", ["Водитель грубил и курил", "забыл зонт", "сколько до вокзала"])
def test_no_false_safety(text):
    assert not guardrails.is_safety_incident(text)


@pytest.mark.parametrize(
    ("text", "previous", "expected"),
    [
        ("Здравствуйте, я забыл зонт", "en", "ru"),
        ("Добрий день, я забув парасольку", "ru", "uk"),
        ("Hi, I left my bag in the car", "ru", "en"),
        ("да", "uk", "uk"),
        ("ok", "ru", "ru"),
    ],
)
def test_language_detection(text, previous, expected):
    assert guardrails.detect_language(text, previous) == expected


def test_amounts_not_returned_by_tools_are_flagged():
    tool_msgs = [{"role": "tool", "content": '{"quote": {"price": 330, "currency": "UAH"}}'}]
    grounded = guardrails.grounded_numbers(tool_msgs)
    assert guardrails.unverified_amounts("Поездка обойдётся в 330 грн", grounded) == set()
    assert guardrails.unverified_amounts("Примерно 350 грн или ₴400", grounded) == {350, 400}
    assert guardrails.unverified_amounts("Поездка займёт 40 минут", grounded) == set()


def test_amount_with_thousands_separator():
    assert guardrails.mentioned_amounts("около 1 200 грн") == {1200}


def test_order_id_grounding_uses_only_customer_messages():
    messages = [
        {"role": "user", "content": "мой заказ № 731-842"},
        {"role": "assistant", "content": "Заказ 999999?"},
    ]
    assert guardrails.is_grounded_order_id("731842", messages)
    assert not guardrails.is_grounded_order_id("999999", messages)

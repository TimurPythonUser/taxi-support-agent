"""Deterministic checks that must not depend on the model's mood."""

import json
import re
from typing import Any

_FLAGS = re.IGNORECASE | re.UNICODE

# Customer explicitly asks for a human. "человек" alone is too broad ("хочу, чтобы этого человека уволили"),
# so it counts only after verbs that mean "bring/connect" or in "talk to a person".
_STAFF = r"(оператор\w*|менеджер\w*|специалист\w*|фахівц\w*)"
_HUMAN_REQUEST = [
    re.compile(p, _FLAGS)
    for p in (
        r"\b(жив\w*|реальн\w*|настоящ\w*|справжн\w*)\s+(человек\w*|людин\w*|оператор\w*)",
        r"\b(позов\w*|соедин\w*|переключ\w*|переве\w*|подключ\w*|дайте|з'?єдна\w*|покли\w*)\b\W+"
        r"(?:\w+\W+){0,3}?(оператор\w*|менеджер\w*|специалист\w*|фахівц\w*|человек\w*|людин\w*|сотрудник\w*)",
        r"\b(хочу|нужен|нужна|можно|потрібен|потрібна)\b\W+(?:\w+\W+){0,3}?" + _STAFF,
        r"\b(поговор\w*|говорить|пообща\w*|поспілкува\w*)\s+(?:с|з|со)\s+(?:живым\s+|живою\s+)?"
        r"(человеком|людиною|оператором|менеджером)",
        r"\b(talk|speak|chat|connect)\w*\s+(?:me\s+)?(?:to|with)\s+(?:a\s+|an\s+|the\s+)?"
        r"(human|person|operator|agent|manager|someone)",
        r"\b(real|live)\s+(person|human|agent|operator)",
        r"^\W*(оператор\w*|operator|human|людину|человека)\W*$",
    )
]

# Safety incidents: accident, violence, threats, harassment, drunk driving (ru / uk / en stems).
# Kept narrow on purpose: a false alarm freezes the dialog with an urgent handoff.
_SAFETY = re.compile(
    r"\b(дтп|авари\w*|аварі\w*|столкновени\w*|врезал\w*|врізав\w*|угрож\w*|погрож\w*|ударил\w*|"
    r"вдарив\w*|избил\w*|побив\w*|напал\w*|напав\w*|насили\w*|насильств\w*|домога\w*|приставал\w*|"
    r"лапал\w*|изнасил\w*|нож|ножом|пистолет\w*|оруж\w*|зброя\w*|пьян\w*|п'ян\w*|пʼян\w*|нетрезв\w*|"
    r"под кайфом|наркот\w*|травм\w*|кровь|крови|кровью|accident|car crash|crashed into|threat\w*|"
    r"assault\w*|harass\w*|drunk|weapon|knife|gun|attacked|injur\w*|hit me|grop\w*)\b",
    _FLAGS,
)

# Money amounts in a reply: "245 грн", "₴245", "UAH 245", "1 200 гривен".
_AMOUNT_AFTER = re.compile(
    r"(\d[\d\s ]{0,6}\d|\d)(?:[.,]\d+)?\s*(грн|гривн\w*|гривен\w*|uah|₴|руб\w*|₽|usd|\$|долл\w*|eur|€|евро)",
    _FLAGS,
)
_AMOUNT_BEFORE = re.compile(r"(₴|\$|€|uah|usd|eur)\s*(\d[\d\s ]{0,6}\d|\d)", _FLAGS)


def wants_human(text: str) -> bool:
    return any(p.search(text) for p in _HUMAN_REQUEST)


def is_safety_incident(text: str) -> bool:
    return bool(_SAFETY.search(text))


_UK_WORDS = {"добрий", "дякую", "будь", "ласка", "забув", "забула", "що", "щось", "вже", "дуже", "зараз", "або"}


def detect_language(text: str, previous: str) -> str:
    """ru / uk / en by script and marker words; short replies like "да" or "ok" keep the previous language."""
    lowered = text.lower()
    letters = [ch for ch in lowered if ch.isalpha()]
    if len(letters) < 6:
        return previous
    if re.search("[іїєґ]", lowered):
        return "uk"
    cyrillic = sum(1 for ch in letters if "а" <= ch <= "я" or ch == "ё")
    if cyrillic / len(letters) > 0.5:
        if not re.search("[ыэъё]", lowered) and _UK_WORDS & set(re.findall(r"\w+", lowered)):
            return "uk"
        return "ru"
    latin = sum(1 for ch in letters if "a" <= ch <= "z")
    if latin / len(letters) > 0.5:
        return "en"
    return previous


def _to_int(raw: str) -> int:
    return int(re.sub(r"\D", "", raw))


def mentioned_amounts(text: str) -> set[int]:
    amounts = {_to_int(m.group(1)) for m in _AMOUNT_AFTER.finditer(text)}
    amounts |= {_to_int(m.group(2)) for m in _AMOUNT_BEFORE.finditer(text)}
    return amounts


def grounded_numbers(messages: list[dict[str, Any]]) -> set[int]:
    """Money amounts that came from tool results: `price` fields and amounts with a currency in texts."""
    amounts: set[int] = set()

    def walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif key == "price" and isinstance(node, int | float):
            amounts.add(int(node))
        elif isinstance(node, str):
            amounts.update(mentioned_amounts(node))

    for msg in messages:
        if msg.get("role") == "tool":
            try:
                walk(json.loads(msg.get("content") or "{}"))
            except ValueError:
                continue
    return amounts


def unverified_amounts(reply: str, grounded: set[int]) -> set[int]:
    """Money amounts in the reply that no tool ever returned — i.e. invented by the model."""
    return mentioned_amounts(reply) - grounded


_DIGIT_GROUP = re.compile(r"\d+(?:[  -]\d+)*")


def number_candidates(text: str) -> set[str]:
    """Whole numbers the customer typed: "№ 731-842" → {"731", "842", "731842"}; never a slice of a phone."""
    out: set[str] = set()
    for group in _DIGIT_GROUP.findall(text):
        runs = re.findall(r"\d+", group)
        out.update(runs)
        out.add("".join(runs))
    return out


def is_grounded_order_id(order_id: str, messages: list[dict[str, Any]]) -> bool:
    return any(order_id in number_candidates(str(m.get("content", ""))) for m in messages if m.get("role") == "user")


def tool_payload(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


FIXED_MESSAGES: dict[str, dict[str, str]] = {
    "handed_off": {
        "ru": "Ваше обращение уже передано оператору — он продолжит разговор в этом чате. "
        "Если ситуация срочная и есть угроза жизни или здоровью, звоните 112.",
        "uk": "Ваше звернення вже передано оператору — він продовжить розмову в цьому чаті. "
        "Якщо ситуація термінова і є загроза життю чи здоров'ю, телефонуйте 112.",
        "en": "Your request has already been passed to an operator, who will continue in this chat. "
        "If there is an immediate danger to life or health, call 112.",
    },
    "human_requested": {
        "ru": "Хорошо, передаю диалог оператору. Он подключится в этом чате; всё, что вы уже рассказали, "
        "он увидит. Операторы отвечают с 08:00 до 23:00.",
        "uk": "Добре, передаю діалог оператору. Він підключиться в цьому чаті й побачить усе, що ви вже "
        "розповіли. Оператори відповідають з 08:00 до 23:00.",
        "en": "Sure, I'm passing this conversation to an operator. They will join this chat and see "
        "everything you've told me. Operators are available from 08:00 to 23:00.",
    },
    "safety_forced": {
        "ru": "Я срочно передал ваше обращение дежурному оператору. Если есть угроза жизни или здоровью "
        "прямо сейчас, пожалуйста, звоните 112.",
        "uk": "Я терміново передав ваше звернення черговому оператору. Якщо є загроза життю чи здоров'ю "
        "просто зараз, будь ласка, телефонуйте 112.",
        "en": "I've urgently passed your request to the operator on duty. If there is an immediate danger "
        "to life or health, please call 112.",
    },
    "stalled": {
        "ru": "Похоже, мне не удаётся помочь самостоятельно — передаю диалог оператору, он подключится в этом чате.",
        "uk": "Схоже, мені не вдається допомогти самостійно — передаю діалог оператору, він підключиться в цьому чаті.",
        "en": "It looks like I can't sort this out on my own, so I'm passing the conversation to an operator "
        "who will join this chat.",
    },
    "technical_error": {
        "ru": "Извините, у меня технический сбой. Попробуйте повторить сообщение через минуту.",
        "uk": "Вибачте, у мене технічний збій. Спробуйте повторити повідомлення за хвилину.",
        "en": "Sorry, I hit a technical problem. Please try sending your message again in a minute.",
    },
    "technical_handoff": {
        "ru": "Извините, у меня повторяется технический сбой — передаю ваше обращение оператору, "
        "он подключится в этом чате.",
        "uk": "Вибачте, технічний збій повторюється — передаю ваше звернення оператору, він підключиться в цьому чаті.",
        "en": "Sorry, the technical problem keeps happening, so I'm passing your request to an operator "
        "who will join this chat.",
    },
    "session_limit": {
        "ru": "Это демо-диалог, и лимит сообщений в нём исчерпан. Начните, пожалуйста, новый диалог.",
        "uk": "Це демо-діалог, і ліміт повідомлень у ньому вичерпано. Почніть, будь ласка, новий діалог.",
        "en": "This is a demo conversation and its message limit has been reached. Please start a new one.",
    },
    "daily_limit": {
        "ru": "Демо-стенд исчерпал дневной лимит запросов к модели. Попробуйте завтра.",
        "uk": "Демо-стенд вичерпав денний ліміт запитів до моделі. Спробуйте завтра.",
        "en": "The demo has reached its daily model request limit. Please try again tomorrow.",
    },
    "price_fallback": {
        "ru": "Извините, я не могу назвать точную стоимость без расчёта. Подскажите, пожалуйста, "
        "адрес отправления и назначения — я сразу посчитаю.",
        "uk": "Вибачте, я не можу назвати точну вартість без розрахунку. Підкажіть, будь ласка, "
        "адресу відправлення та призначення — я одразу порахую.",
        "en": "Sorry, I can't name an exact price without calculating it. Please tell me the pickup and "
        "drop-off addresses and I'll check right away.",
    },
}


def fixed(key: str, language: str) -> str:
    variants = FIXED_MESSAGES[key]
    return variants.get(language, variants["ru"])

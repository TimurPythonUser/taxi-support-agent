"""Deterministic checks that must not depend on the model's mood."""

import json
import re
from typing import Any

_FLAGS = re.IGNORECASE | re.UNICODE
_APOS = "['’ʼ`]"  # phone keyboards type U+2019 / U+02BC, not ASCII

# --- customer asks for a human ------------------------------------------------------------------
# "человек" alone is too broad ("дайте контакты этого человека"), so it counts only after verbs that mean
# "bring/connect" or in "talk to a person"; "хочу/дайте" need an explicit staff word.
_STAFF = r"(?:оператор(?:а|у|ом|ів|ам)?|менеджер(?:а|у|ом)?|специалист(?:а|у|ом)?|фахівц(?:я|ю|ем)|фахівець)\b"
_PERSON = r"(?:человек(?:а|у|ом)?|людин(?:а|у|ою|и)?|сотрудник(?:а|у|ом)?)\b"
_HUMAN_REQUEST = [
    re.compile(p, _FLAGS)
    for p in (
        r"\b(?:жив\w*|реальн\w*|настоящ\w*|справжн\w*)\s+(?:человек\w*|людин\w*|оператор\w*)",
        r"\b(?:позов\w*|соедин\w*|з" + _APOS + r"?єдна\w*|переключ\w*|переве(?:д|ст)\w*|подключ\w*|покли\w*)\b\W+"
        r"(?:\w+\W+){0,3}?(?:" + _STAFF + "|" + _PERSON + ")",
        r"\b(?:хочу|нужен|нужна|нужно|можно|дайте|дай|давайте|потрібен|потрібна|дайте)\b\W+(?:\w+\W+){0,2}?" + _STAFF,
        r"\b(?:поговор\w*|говорить|пообща\w*|поспілкува\w*|связаться|зв" + _APOS + r"?язатися)\s+(?:с|з|со)\s+"
        r"(?:живым\s+|живою\s+|живим\s+)?(?:человеком|людиною|оператором|менеджером)",
        r"\b(?:с|з|со)\s+(?:живым\s+|живою\s+|живим\s+)?(?:человеком|людиною|оператором)\s+"
        r"(?:поговор\w*|пообща\w*|поспілкува\w*|связаться)",
        r"\b(?:talk|speak|chat|connect)\w*\s+(?:me\s+)?(?:to|with)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:human|person|operator|agent|manager|someone)",
        r"\b(?:real|live)\s+(?:person|human|agent|operator)",
        r"\b(?:need|want|get me)\s+(?:a\s+|an\s+|the\s+)?(?:human|person|operator|agent)\b",
        r"\b(?:human|operator|person)\W*\s*please\b",
        r"^\W*(?:оператор\w*|operator|human|людину|человека)\W*$",
    )
]
# "Не хочу оператора", "не надо соединять", "I don't need a human" — the request is negated.
_NEGATED = re.compile(
    r"(?:\b(?:не|ні|ни|don't|dont|do not|not|no need to)\s+$"
    r"|\b(?:не надо|не нужно|не хочу|не стоит|не треба|не потрібно|без)\s+(?:(?:меня|мне|нас|вас|мене|нам)\s+)?$)",
    _FLAGS,
)
_WHY_NOT = re.compile(r"\b(?:почему|зачем|чего|чому|навіщо|why)\s+(?:вы\s+|ви\s+)?не\s+$", _FLAGS)


def wants_human(text: str) -> bool:
    for pattern in _HUMAN_REQUEST:
        for match in pattern.finditer(text):
            before = text[max(0, match.start() - 30) : match.start()]
            if _WHY_NOT.search(before) or not _NEGATED.search(before):
                return True
    return False


# --- safety incidents ---------------------------------------------------------------------------
# Narrow on purpose: a false alarm freezes the dialog with an urgent handoff. "Пьяный" counts only about
# the driver, a knife or a gun only when someone threatens with it.
_DRIVER = r"(?:водител\w*|водій|водія|водієм|таксист\w*|шофер\w*|driver)"
_INTOXICATED = (
    r"(?:пьян\w*|п"
    + _APOS
    + r"ян\w*|нетрезв\w*|напил\w*|напив\w*|выпивш\w*|под кайфом|под наркот\w*|drunk|intoxicated)(?!\w)"
)
_SAFETY = [
    re.compile(p, _FLAGS)
    for p in (
        r"\b(?:дтп|авари(?:я|и|ю|ей)|аварі(?:я|ї|ю|єю)|столкновени\w*|зіткненн\w*|врезал\w*|врізав\w*|"
        r"сбил(?!ся|ась|ось|ись)\w*|збив(?!ся|лася|лося|лися)\w*|врізал\w*|(?<!by )accident|car crash|crashed into|"
        r"we crashed|car crashed|collision)\b",
        r"\b(?:угрож\w*|погрож\w*|ударил\w*|вдарив\w*|избил\w*|побив\w*|напал\w*|напав\w*|насили\w*|"
        r"насильств\w*|изнасил\w*|ножом|threat\w*|assault\w*|attacked|hit me|beat me|punched)\b",
        r"\b(?:наставил|направил|достал|вытащил|показал|навів|дістав)\w*\s+(?:[\w-]+\s+){0,3}?"
        r"(?:пистолет\w*|оружи\w*|нож\w*|ствол|пістолет\w*|зброю)",
        r"\b(?:pulled|drew|pointed)\s+(?:a\s+)?(?:gun|knife|weapon)",
        r"\b(?:домога\w*|пристал\w*|приставал\w*|приставав\w*|приста[её]т\w*|лапа(?:ет|л|ла|ли|ть)\w*|чіпля\w*|harass\w*|grop\w*|"
        r"touched me|grabbed me)\b",
        r"\b(?:не выпуска\w*|не выпустил\w*|не отпуска\w*|не відпуска\w*|не випуска\w*|запер\w*|"
        r"заблокировал\w*\s+двер\w*|won't let me out|locked me in)",
        r"\b(?:травм(?:а|ы|у|ой|е|и)|травмирова\w*|кровь(?!\s+из\s+носа)|крови|кровью|кров|injur\w*|bleeding)\b",
        r"\b" + _DRIVER + r"\W+(?:\w+\W+){0,4}?" + _INTOXICATED,
        r"\b" + _INTOXICATED + r"\s+" + _DRIVER,
        r"\bdrunk driv\w*",
        r"\bdriver\s+(?:was|is|seemed|seems|looked|looks)\s+(?:\w+\s+)?high\b",
    )
]


def is_safety_incident(text: str) -> bool:
    return any(p.search(text) for p in _SAFETY)


# --- language -----------------------------------------------------------------------------------
_UK_WORDS = {
    "добрий", "добре", "дякую", "ласка", "забув", "забула", "що", "щось", "щоб", "вже", "дуже", "зараз",
    "або", "чекаю", "вчора", "замовлення", "також", "треба",
}  # fmt: skip
_RU_WORDS = {
    "что", "это", "нет", "да", "спасибо", "здравствуйте", "пожалуйста", "сколько", "вчера", "заказ", "привет",
    "хорошо", "только", "уже", "очень", "сейчас",
}  # fmt: skip


_BRANDS = {"iphone", "ipad", "airpods", "macbook", "samsung", "xiaomi", "galaxy", "apple", "pro", "max", "mini", "plus"}


def detect_language(text: str, previous: str) -> str:
    """ru / uk / en. Brand names (iPhone, AirPods) do not make a message English, and short replies without
    language markers ("Так, правильно", "ok") keep the previous language."""
    words = re.findall(r"[^\W\d_]+", text)
    if sum(len(w) for w in words) < 6:
        return previous
    cyrillic = [w for w in words if re.search("[а-яёіїєґ]", w.lower())]
    # capitalised words are names and brands ("iPhone", "Васильківська"); lowercase brands are listed
    plain_latin = [w for w in words if re.fullmatch("[a-z]{2,}", w) and w not in _BRANDS]
    if cyrillic and len(cyrillic) >= len(plain_latin):
        common = [w.lower() for w in cyrillic if not w[0].isupper()] or [w.lower() for w in cyrillic]
        letters = "".join(common)
        ru = len(re.findall("[ыэъё]", letters)) + sum(w in _RU_WORDS for w in (x.lower() for x in cyrillic))
        uk = len(re.findall("[іїєґ]", letters)) + sum(w in _UK_WORDS for w in (x.lower() for x in cyrillic))
        if uk > ru:
            return "uk"
        if ru > uk:
            return "ru"
        return previous if previous in {"ru", "uk"} else "ru"
    if plain_latin:
        return "en"
    return previous


# --- money amounts ------------------------------------------------------------------------------
# A number is not glued to a neighbouring date or order number ("25.10 245 грн" → 245) and keeps its
# thousands separator ("1,150 UAH" → 1150, "1 200 грн" → 1200).
_NUM = r"(?<![\d.,:])(\d{1,3}(?:[   ,]\d{3})+(?!\d)|\d+)(?:[.,]\d{1,2}(?!\d))?"
_CURRENCY = r"(?:грн|гривн\w*|гривен\w*|гривень|hryvni\w*|uah|₴|руб\w*|₽|usd|\$|долл\w*|dollars?|eur\b|euros?|€|евро)"
_AMOUNT_AFTER = re.compile(_NUM + r"\s*" + _CURRENCY, _FLAGS)
_AMOUNT_BEFORE = re.compile(r"(?:₴|\$|€|\buah|\busd|\beur)\s*" + _NUM, _FLAGS)
# "будет стоить около 300" — a price sentence without a currency
_UNITS = (
    r"(?:мин\w*|хв\w*|км|km|час\w*|годин\w*|сек\w*|%|раз\w*|человек\w*|пассажир\w*|min\w*|hour\w*|day\w*|дн\w*|день)"
)
_AMOUNT_BARE = re.compile(
    r"\b(?:стои\w*|стоимость\w*|обойд\w*|цена|цену|составит|составляет|коштує|коштуватиме|вартість|cost\w*|"
    r"price|fare)(?:\s+(?:поездки|проезда|такси|поїздки|проїзду|of|the|ride|trip))*\s*[:—–-]?\s*"
    r"(?:(?:будет|буде|составит|около|примерно|приблизительно|где-то|порядка|близько|приблизно|"
    r"в|у|is|will|be|about|around|approximately|roughly)\s+)*" + _NUM + r"(?![:.]\d)(?!\s*" + _UNITS + ")",
    _FLAGS,
)
_WORD_NUMBERS = {
    # ru
    "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "ста": 100,
    "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300, "трёхсот": 300, "четыреста": 400,
    "четырехсот": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900, "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000,
    # uk
    "дві": 2, "чотири": 4, "п'ять": 5, "шість": 6, "сім": 7, "вісім": 8, "дев'ять": 9, "двадцять": 20,
    "тридцять": 30, "п'ятдесят": 50, "шістдесят": 60, "сімдесят": 70, "вісімдесят": 80, "дев'яносто": 90,
    "двісті": 200, "чотириста": 400, "п'ятсот": 500, "шістсот": 600, "сімсот": 700, "вісімсот": 800,
    "дев'ятсот": 900, "тисяча": 1000, "тисячі": 1000, "тисяч": 1000, "одинадцять": 11, "дванадцять": 12,
    "п'ятнадцять": 15, "двадцяти": 20,
    "одиннадцать": 11, "двенадцать": 12, "пятнадцать": 15, "двадцати": 20, "тридцати": 30, "сорока": 40,
    "пятидесяти": 50, "шестисот": 600, "семисот": 700, "восьмисот": 800, "девятисот": 900,
    # en
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}  # fmt: skip
_WORDS_BEFORE_CURRENCY = re.compile(r"((?:[^\W\d_]+(?:['’ʼ][^\W\d_]+)?\s+){1,5})" + _CURRENCY, _FLAGS)


def _to_int(raw: str) -> int:
    return int(re.sub(r"\D", "", raw))


def _words_to_int(words: list[str]) -> int | None:
    tail: list[str] = []
    for word in reversed(words):
        if word.replace("’", "'").replace("ʼ", "'") not in _WORD_NUMBERS:
            break
        tail.append(word.replace("’", "'").replace("ʼ", "'"))
    if not tail:
        return None
    total = current = 0
    for word in reversed(tail):
        value = _WORD_NUMBERS[word]
        if value == 1000:
            total += max(current, 1) * 1000
            current = 0
        elif word == "hundred":
            current = max(current, 1) * 100
        else:
            current += value
    return total + current


def mentioned_amounts(text: str) -> set[int]:
    amounts = {_to_int(m.group(1)) for m in _AMOUNT_AFTER.finditer(text)}
    amounts |= {_to_int(m.group(1)) for m in _AMOUNT_BEFORE.finditer(text)}
    amounts |= {_to_int(m.group(1)) for m in _AMOUNT_BARE.finditer(text)}
    for m in _WORDS_BEFORE_CURRENCY.finditer(text):
        value = _words_to_int(m.group(1).lower().split())
        if value:
            amounts.add(value)
    return amounts


_COMPENSATION = re.compile(
    r"(?:компенс\w*|возмещ\w*|верн\w*|возврат\w*|відшкод\w*|поверн\w*|refund\w*|compensat\w*)\W+(?:\w+\W+){0,4}$",
    _FLAGS,
)


def customer_amounts(text: str) -> set[int]:
    """Amounts the customer stated as facts ("списали 480 грн"); demands ("компенсируйте 5000 грн") excluded."""
    amounts: set[int] = set()
    for pattern in (_AMOUNT_AFTER, _AMOUNT_BEFORE):
        for m in pattern.finditer(text):
            if not _COMPENSATION.search(text[max(0, m.start() - 60) : m.start()]):
                amounts.add(_to_int(m.group(1)))
    return amounts


def grounded_numbers(messages: list[dict[str, Any]], *, include_customer: bool = False) -> set[int]:
    """Money amounts the reply may contain: quoted prices, amounts in knowledge-base texts and ticket
    policies — and, outside price quotes, amounts the customer typed (e.g. "списали 480 грн").

    Echoes of the case (update_case) and past trip fares (find_trip) do not count: otherwise the model
    could "ground" a price by writing it into a case field or pass an old fare off as a new quote.
    """
    amounts: set[int] = set()
    for msg in messages:
        if msg.get("role") == "user" and include_customer:
            amounts |= customer_amounts(str(msg.get("content", "")))
        if msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg.get("content") or "{}")
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        quote = data.get("quote")
        if isinstance(quote, dict) and isinstance(quote.get("price"), int | float):
            amounts.add(int(quote["price"]))
        for item in data.get("results") or []:
            if isinstance(item, dict):
                amounts |= mentioned_amounts(str(item.get("text", "")))
        for text in data.get("policy") or []:
            amounts |= mentioned_amounts(str(text))
    return amounts


def unverified_amounts(reply: str, grounded: set[int]) -> set[int]:
    """Money amounts in the reply that no trusted source gave — i.e. invented by the model."""
    return mentioned_amounts(reply) - grounded


# --- numbers typed by the customer --------------------------------------------------------------
_DIGIT_GROUP = re.compile(r"\+?\d+(?:[  ()-]+\d+)*")
_PHONE = re.compile(r"0\d{9}|380\d{9}")


def number_candidates(text: str) -> set[str]:
    """Whole numbers the customer typed: "№ 731-842" → {"731", "842", "731842"}; never a slice of a phone."""
    out: set[str] = set()
    for group in _DIGIT_GROUP.findall(text):
        runs = re.findall(r"\d+", group)
        out.update(runs)
        out.add("".join(runs))
    return out


def _customer_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content", "")) for m in messages if m.get("role") == "user"]


def is_grounded_order_id(order_id: str, messages: list[dict[str, Any]]) -> bool:
    return any(order_id in number_candidates(text) for text in _customer_texts(messages))


def looks_like_phone(digits: str) -> bool:
    return bool(_PHONE.fullmatch(digits))


def is_grounded_phone(phone_digits: str, messages: list[dict[str, Any]]) -> bool:
    """The model may normalize "067 123 45 67" to "+380671234567": compare the last 9 digits."""
    if len(phone_digits) < 9:
        return False
    tail = phone_digits[-9:]
    return any(
        len(candidate) >= 9 and candidate[-9:] == tail
        for text in _customer_texts(messages)
        for candidate in number_candidates(text)
    )


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
    "amount_fallback": {
        "ru": "Извините, эту сумму я подтвердить не могу — её уточнит оператор при разборе обращения.",
        "uk": "Вибачте, цю суму я підтвердити не можу — її уточнить оператор під час розгляду звернення.",
        "en": "Sorry, I can't confirm that amount — the operator will check it when reviewing your request.",
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

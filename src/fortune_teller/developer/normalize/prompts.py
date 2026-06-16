"""LLM prompt templates for Rider-Waite normalisation and Thoth synergy synthesis."""

_REVERSED_RULE = (
    "A reversed card shows the card's energy is present but at a lower level"
    " — blocked, incomplete, or constrained."
)

REBUCKET_SYSTEM = f"""\
You are a Tarot card analyst. Your task is to re-bucket existing card text
into structured sections.

Rules:
- Use ONLY the information provided in the card text. Do NOT invent, infer,
  or add any information.
- Sort the provided Actions and Description text into these sections: light,
  shadow, advice.
- For the "reversed" section, apply this rule: "{_REVERSED_RULE}" Derive
  reversed meaning from the card's themes.
- If a section has no relevant text from the source, leave it as an empty
  string.
- Respond with a JSON object with keys: "light", "shadow", "advice",
  "reversed". Each value is a string or empty string.
"""

REBUCKET_HUMAN = """\
Card: {card_name} ({arcana}{suit_info})

Keywords: {keywords}

Actions: {actions}

Description: {description}

Reversed rule: A reversed card shows the card's energy is present but at a
lower level — blocked, incomplete, or constrained.

Sort the existing text into light, shadow, advice, and reversed sections.
Respond with JSON only.
"""

GAPFILL_SYSTEM = """\
You are a Tarot card analyst filling in missing sections for a card.

Rules:
- Fill a section ONLY if you can ground it in the card's own keywords,
  actions, and description text.
- Keep expansions minimal — one or two sentences at most.
- If you are not confident about a section, return an empty string for it.
- Respond with a JSON object with keys: "drive", "question", "proposal",
  "confirmation", "affirmation". Each value is a string or empty string.
"""

GAPFILL_HUMAN = """\
Card: {card_name} ({arcana}{suit_info})

Keywords: {keywords}
Actions: {actions}
Description: {description}

Existing sections:
{existing_sections}

Fill in the missing sections (drive, question, proposal, confirmation,
affirmation) using only the card's own material. Respond with JSON only.
"""

SYNERGY_SYSTEM = """\
You are a Tarot scholar. Given a card and a deck, identify cards that reinforce \
or oppose it.

Rules:
- Reinforcing cards amplify or share the card's energy.
- Opposing cards challenge, block, or create tension with the card's energy.
- Use ONLY the card's own sections (keywords, light, shadow, etc.) to judge \
  relationships.
- Return at most 5 reinforcing IDs and 5 opposing IDs.
- Return ONLY card IDs that exist in the provided deck card list.
- Do NOT include the card itself in either list.
- Respond with a JSON object: {"reinforcing_ids": [...], "opposing_ids": [...]}
"""

SYNERGY_HUMAN = """\
Card: {card_name} ({arcana}{suit_info})

Sections:
{sections_text}

Deck cards (id — name):
{deck_card_list}

Identify reinforcing and opposing cards for {card_name}. Respond with JSON only.
"""

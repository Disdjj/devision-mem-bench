"""Memory 分类体系。

Jev 和 DeepSeek 的判断器共用这里的定义，保证两边面对的是同一套题目。
Jev 不能生成文本，所以类型、tag、优先级都必须是封闭集合。
"""

from __future__ import annotations

MEMORY_TYPES: dict[str, str] = {
    "preference": "The user's likes, dislikes, tastes, or preferred ways of doing things (food, tools, style, products).",
    "profile": "Stable facts about who the user is: name, job, location, family members, pets, health conditions, skills.",
    "project": "An ongoing goal, project, or long-running effort the user is working on, and its key details or decisions.",
    "event": "A specific dated or time-bound happening: a past experience, an upcoming plan, appointment, trip, or deadline.",
    "instruction": "A standing rule about how the assistant should behave or respond to this user in the future.",
}

TAGS: dict[str, str] = {
    "work": "the user's job, career, colleagues, or workplace",
    "tech": "software, programming, devices, or technical tools",
    "health": "physical or mental health, allergies, medical conditions, fitness, or sleep",
    "food": "food, cooking, restaurants, diet, or drinks",
    "travel": "trips, transportation, destinations, or places to visit",
    "family": "family members, partners, children, friends, or pets",
    "finance": "money, budgets, spending, investments, or purchases",
    "learning": "studying, courses, languages, books, or acquiring new skills",
    "hobby": "hobbies, sports, games, music, art, or leisure activities",
    "schedule": "dates, times, deadlines, routines, or calendar commitments",
    "communication": "how the user wants the assistant to talk: language, tone, format, length",
}

PRIORITY_LEVELS: list[str] = [
    "Low: nice-to-know trivia that rarely changes how the assistant should respond.",
    "Medium: useful context that improves responses whenever a related topic comes up.",
    "High: critical information that must not be forgotten, such as safety or health constraints, "
    "hard requirements, or something the user explicitly asked to remember.",
]
PRIORITY_NAMES = ["low", "medium", "high"]

# ---- Jev 题目文案（英文：Jev 以英文训练为主，指令用英文，state 可以是任意语言）----

STORE_INSTRUCTION = (
    "Does `latest_user_message` contain information about the user that is worth saving to long-term memory, "
    "meaning it would still be useful in a future, separate conversation?"
)
STORE_CRITERIA = {
    "true": "Durable personal information: preferences, facts about the user's life, ongoing projects, "
    "upcoming plans or important past events, or standing instructions for the assistant.",
    "false": "Small talk, greetings, general knowledge questions, one-off requests, or a passing state "
    "that only matters right now.",
}
TYPE_INSTRUCTION = "Which kind of memory best describes the durable information in `latest_user_message`?"
TAG_INSTRUCTION = "Is the information in `latest_user_message` related to {desc}?"
PRIORITY_INSTRUCTION = "How important is it for the assistant to remember the information in `latest_user_message`?"

RECALL_INSTRUCTION = (
    "Could knowing stored long-term memories about this specific user (their preferences, profile, projects, "
    "past events, or standing instructions) make the response to `latest_user_message` more personalized, "
    "relevant, or safe?"
)
RECALL_CRITERIA = {
    "true": "Recommendations, advice, planning, personal tasks, continuing earlier work, or anything where the "
    "user's tastes, constraints, history, or rules for the assistant could change the best response.",
    "false": "Greetings, small talk, or purely factual and general questions where nothing about the user "
    "would change the answer.",
}
RECALL_TYPE_INSTRUCTION = (
    "Would memories of this kind help respond to `latest_user_message`? Kind: {desc}"
)
RECALL_TAG_INSTRUCTION = (
    "Would memories related to {desc} help respond to `latest_user_message`?"
)
RELEVANCE_INSTRUCTION = (
    "Is `memory` relevant and useful for responding to the user's `latest_user_message`?"
)

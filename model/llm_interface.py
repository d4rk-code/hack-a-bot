"""
LLM interface using OpenAI's modern client.
Requires environment variable OPENAI_API_KEY to be set.

Usage:
    from model.llm_interface import generate_training_program
    plan_text = generate_training_program(
        sport="badminton",
        goal="endurance",
        duration_weeks=4,
        fitness_level="beginner"
    )
"""

import os
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI

load_dotenv()

#  Use environment variable or direct key
API_KEY = os.getenv("OPENAI_API_KEY")

def _get_client():
    if not API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
    return OpenAI(api_key=API_KEY)

def generate_training_program(
    sport: str,
    goal: str,
    duration_weeks: int = 4,
    fitness_level: str = "intermediate",
    model_name: Optional[str] = "gpt-5-nano"
) -> str:
    """
    Generate a multi-week training plan for a given sport and goal using the LLM.
    Returns plain text with a structured plan.
    """
    client = _get_client()

    prompt = f"""
You are an expert sports coach and data-driven fitness specialist.
Create a {duration_weeks}-week structured training plan to improve {goal} in {sport}
for a {fitness_level} athlete.

Include clearly:
- Weekly schedule (week-by-week)
- Exercises/drills with durations
- Frequency (how many times per week)
- Rest/recovery guidance
- Short rationale for progression

Return the plan in readable bullet points and short paragraphs.
"""

    # 🚀 Make request to GPT-5-nano (temperature not supported, so omitted)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful AI sports coach."},
            {"role": "user", "content": prompt}
        ]
    )

    return resp.choices[0].message.content.strip()


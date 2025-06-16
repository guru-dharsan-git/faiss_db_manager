from google.genai import types
from google import genai
from dotenv import load_dotenv
import os
import json
import re

def classify_news_as_lead(news: str, rulebook_path: str = "rulebook.txt") -> dict:
    """
    Classify a news item as a lead and assign it to a category based on the rulebook.
    Returns a dict with keys:
      - is_lead (bool)
      - category (str or None)
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    with open(rulebook_path, "r") as f:
        rulebook = f.read()

    client = genai.Client(api_key=api_key)

    categories = [
        "Agriculture", "Manufacturing", "Construction", "Hospitals and Health Care",
        "Financial Services", "Real Estate", "Transportation", "Energy",
        "Technology, Information and Internet", "Retail", "Hospitality",
        "Education", "Media and Telecommunications", "Government"
    ]

    prompt = f"""

        You are given a rulebook describing what qualifies as a lead:\n{rulebook}\n\n"
        For the following news item, determine if it is a lead. If it is, assign it to one of the categories: {', '.join(categories)}."
        "\nRespond with valid JSON with keys 'is_lead' (true/false) and 'category' (string or null)."
        \nNews: {news}\n"
        
    """
    sytle = """
    write the output in json format only
    {
    "is_lead": false,
    "category": null
    }
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt+"\n"+sytle,
        config=types.GenerateContentConfig(
            system_instruction="You will classify a news item as a lead or not and, if a lead, assign the correct category in JSON.",
            temperature=0.0,
            max_output_tokens=100
        )
    )

    raw = response.text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()
        raw = re.sub(r'^json\s*', '', raw, flags=re.IGNORECASE)
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from LLM response: {response.text}")

    return result

if __name__ == "__main__":
    example_news = (
        "Updates: Ex-Chief Minister Vijay Rupani's DNA Matches, 32 Bodies Identified discription Ahmedabad Plane Crash Live Updates: The bodies of the 274 victims of the Ahmedabad plane crash, who have been identified, are set to be handed over to their families by the Gujarat government on Sunday, sources said."
    )
    classification = classify_news_as_lead(example_news)
    print(json.dumps(classification, indent=2))

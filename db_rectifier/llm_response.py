from google.genai import types
from google import genai
from dotenv import load_dotenv
import os

def llm_true_false(news1: str, news2: str) -> bool:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    user_prompt = (
        f"News1: {news1}\n\n"
        f"News2: {news2}\n\n"
        "Answer with exactly 'true' if they describe the same event/story, "
        "or 'false' if they do not (no extra text)."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You will be given two news items as input; respond with exactly "
                "'true' if they describe the same event/story, or 'false' if they do not."
            ),
            temperature=0.0,
            max_output_tokens=4,
        ),
    )

    return response.text.strip().lower() == "true"

if __name__ == "__main__":
    ans = llm_true_false(
        "Air India Mumbai-London Flight Returns After 3 Hours In Air: Flightradar24",
        "An Air India flight, en route to London, returned to Mumbai after three hours in air, "
        "according to news agency PTI citing Flightradar24 data."
    )
    print("Same story?", ans)

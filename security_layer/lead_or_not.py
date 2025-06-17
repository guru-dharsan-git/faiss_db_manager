from google.genai import types
from google import genai
from dotenv import load_dotenv
import os
import json
import re

def classify_news_as_lead(news: str, rulebook_path: str = "rulebook.txt") -> dict:
    """
    Classify a news item as a lead and assign it to a category and sub-category based on the rulebook.
    Returns a dict with keys:
      - is_lead (bool)
      - category (str or None)
      - sub_category (str or None)
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    with open(rulebook_path, "r") as f:
        rulebook = f.read()

    client = genai.Client(api_key=api_key)

    # Categories with their respective sub-categories
    categories_with_subcategories = {
        "Agriculture": [
            "Crop Production", "Livestock", "Dairy Farming", "Poultry", 
            "Fisheries", "Agricultural Technology", "Food Processing", 
            "Organic Farming", "Agricultural Equipment", "Seeds and Fertilizers"
        ],
        "Manufacturing": [
            "Automotive", "Textiles", "Electronics", "Pharmaceuticals", 
            "Chemicals", "Steel and Metals", "Machinery", "Consumer Goods", 
            "Aerospace", "Food and Beverages"
        ],
        "Construction": [
            "Residential Construction", "Commercial Construction", "Infrastructure", 
            "Road Construction", "Bridge Construction", "Real Estate Development", 
            "Construction Materials", "Architecture", "Engineering Services", "Renovation"
        ],
        "Hospitals and Health Care": [
            "Hospitals", "Clinics", "Medical Devices", "Pharmaceuticals", 
            "Mental Health", "Emergency Services", "Nursing", "Medical Research", 
            "Telemedicine", "Health Insurance"
        ],
        "Financial Services": [
            "Banking", "Insurance", "Investment", "Stock Market", "Mutual Funds", 
            "Credit Services", "Fintech", "Cryptocurrency", "Financial Planning", 
            "Accounting Services"
        ],
        "Real Estate": [
            "Residential Real Estate", "Commercial Real Estate", "Property Management", 
            "Real Estate Investment", "Property Development", "Real Estate Brokerage", 
            "Land Development", "Property Valuation", "Real Estate Finance", "Rental Services"
        ],
        "Transportation": [
            "Airlines", "Railways", "Shipping", "Trucking", "Public Transportation", 
            "Logistics", "Warehousing", "Port Operations", "Airport Services", 
            "Ride Sharing"
        ],
        "Energy": [
            "Oil and Gas", "Renewable Energy", "Solar Power", "Wind Power", 
            "Nuclear Energy", "Coal", "Electricity Generation", "Energy Storage", 
            "Hydroelectric", "Biofuels"
        ],
        "Technology, Information and Internet": [
            "Software Development", "Hardware", "Internet Services", "Cloud Computing", 
            "Artificial Intelligence", "Cybersecurity", "Data Analytics", "Mobile Apps", 
            "E-commerce", "Social Media"
        ],
        "Retail": [
            "Fashion Retail", "Grocery Stores", "Electronics Retail", "Online Retail", 
            "Department Stores", "Specialty Stores", "Automotive Retail", "Home Improvement", 
            "Sporting Goods", "Pharmacy"
        ],
        "Hospitality": [
            "Hotels", "Restaurants", "Tourism", "Travel Agencies", "Event Management", 
            "Catering", "Resorts", "Airlines Hospitality", "Cruise Lines", "Entertainment"
        ],
        "Education": [
            "K-12 Education", "Higher Education", "Online Education", "Vocational Training", 
            "Educational Technology", "Private Tutoring", "Educational Publishing", 
            "Research Institutions", "Educational Services", "Student Services"
        ],
        "Media and Telecommunications": [
            "Television", "Radio", "Print Media", "Digital Media", "Telecommunications", 
            "Internet Services", "Broadcasting", "Publishing", "Advertising", "Mobile Services"
        ],
        "Government": [
            "Federal Government", "State Government", "Local Government", "Public Policy", 
            "Defense", "Law Enforcement", "Judiciary", "Public Services", "Regulatory Bodies", 
            "International Relations"
        ]
    }

    # Create formatted string for prompt
    categories_text = []
    for category, subcategories in categories_with_subcategories.items():
        subcategories_str = ", ".join(subcategories)
        categories_text.append(f"{category}: [{subcategories_str}]")
    
    categories_formatted = "\n".join(categories_text)

    prompt = f"""
You are given a rulebook describing what qualifies as a lead:
{rulebook}

For the following news item, determine if it is a lead. If it is, assign it to one of the main categories and then to a specific sub-category within that category.

Categories and Sub-categories:
{categories_formatted}

Respond with valid JSON with keys 'is_lead' (true/false), 'category' (string or null), and 'sub_category' (string or null).

News: {news}
"""

    style = """
Write the output in JSON format only:
{
    "is_lead": false,
    "category": null,
    "sub_category": null
}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt + "\n" + style,
        config=types.GenerateContentConfig(
            system_instruction="You will classify a news item as a lead or not and, if a lead, assign the correct category and sub-category in JSON format.",
            temperature=0.0,
            max_output_tokens=150
        )
    )

    raw = response.text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()
        raw = re.sub(r'^json\s*', '', raw, flags=re.IGNORECASE)
    
    try:
        result = json.loads(raw)
        
        # Validate that sub_category belongs to the selected category
        if result.get("is_lead") and result.get("category") and result.get("sub_category"):
            category = result["category"]
            sub_category = result["sub_category"]
            
            if category in categories_with_subcategories:
                valid_subcategories = categories_with_subcategories[category]
                if sub_category not in valid_subcategories:
                    # If sub_category is invalid, set it to None
                    result["sub_category"] = None
        
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from LLM response: {response.text}")

    return result

if __name__ == "__main__":
    example_news = (
        "Updates: Ex-Chief Minister Vijay Rupani's DNA Matches, 32 Bodies Identified discription Ahmedabad Plane Crash Live Updates: The bodies of the 274 victims of the Ahmedabad plane crash, who have been identified, are set to be handed over to their families by the Gujarat government on Sunday, sources said."
    )
    classification = classify_news_as_lead(example_news)
    print(json.dumps(classification, indent=2))
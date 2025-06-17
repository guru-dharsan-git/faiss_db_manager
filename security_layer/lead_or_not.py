from google.genai import types
from google import genai
from dotenv import load_dotenv
import os
import json
import re

def classify_news_as_lead(news: str = "The government has launched a ₹5,000 crore program linking education with industry training, aiming to enhance employability through skill-based curricula. Financial institutions will co-fund this initiative, creating a strong bridge between education, industry needs, and funding support.", rulebook_path: str = "rulebook.txt") -> dict:
    """
    Classify a news item as a lead and assign it to a category and sub-category based on the rulebook.
    Returns a dict with keys:
      - is_lead (bool)
      - category (list or None)
      - sub_category (list or None)
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

If the news item qualifies for multiple categories, include all relevant categories and their corresponding sub-categories.

Respond with valid JSON with keys 'is_lead' (true/false), 'category' (array or null), and 'sub_category' (array or null).

News: {news}
"""

    style = """
Write the output in JSON format only:

For single category:
{
    "is_lead": true,
    "category": ["Construction"],
    "sub_category": ["Commercial Construction"]
}

For multiple categories:
{
    "is_lead": true,
    "category": ["Construction", "Education"],
    "sub_category": ["Commercial Construction", "Vocational Training"]
}

For no lead:
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
            system_instruction="You will classify a news item as a lead or not and, if a lead, assign the correct category and sub-category in JSON format. Always return arrays for both category and sub_category fields when is_lead is true, even for single categories.",
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
        
        # Validate that sub_categories belong to the selected categories
        if result.get("is_lead") and result.get("category") and result.get("sub_category"):
            categories = result["category"]
            sub_categories = result["sub_category"]
            
            # Ensure both are lists
            if not isinstance(categories, list):
                categories = [categories]
            if not isinstance(sub_categories, list):
                sub_categories = [sub_categories]
            
            # Validate each sub_category belongs to its corresponding category
            validated_categories = []
            validated_sub_categories = []
            
            for i, category in enumerate(categories):
                if category in categories_with_subcategories:
                    validated_categories.append(category)
                    
                    # Check if corresponding sub_category exists and is valid
                    if i < len(sub_categories):
                        sub_category = sub_categories[i]
                        valid_subcategories = categories_with_subcategories[category]
                        if sub_category in valid_subcategories:
                            validated_sub_categories.append(sub_category)
                        else:
                            # If sub_category is invalid, skip this pair
                            validated_categories.pop()
            
            # Update result with validated data
            result["category"] = validated_categories if validated_categories else None
            result["sub_category"] = validated_sub_categories if validated_sub_categories else None
        
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from LLM response: {response.text}")

    return result

if __name__ == "__main__":
    example_news = (
        "Updates: Ex-Chief Minister Vijay Rupani's DNA Matches, 32 Bodies Identified discription Ahmedabad Plane Crash Live Updates: The bodies of the 274 victims of the Ahmedabad plane crash, who have been identified, are set to be handed over to their families by the Gujarat government on Sunday, sources said."
    )
    classification = classify_news_as_lead()
    print(json.dumps(classification, indent=2))
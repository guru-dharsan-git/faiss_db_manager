from google.genai import types
from google import genai
from dotenv import load_dotenv
import os
import json
import re
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from response text, handling various formats and edge cases.
    """
    try:
        # Remove code block markers if present
        cleaned_text = re.sub(r'```json\s*|\s*```', '', response_text.strip())
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        
        # If no braces found, try to parse the entire cleaned text
        return json.loads(cleaned_text)
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Response text: {response_text}")
        return None

def validate_and_structure_entities(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and structure the extracted entities with relationships.
    """
    # Initialize structured data
    structured_data = {
        "entities": [],
        "summary": "",
        "standalone_locations": [],
        "standalone_contacts": [],
        "metadata": {
            "total_people": 0,
            "total_companies": 0,
            "total_contacts": 0,
            "total_locations": 0
        }
    }
    
    # Extract entities (people/companies with their relationships)
    entities = data.get("entities", [])
    if isinstance(entities, list):
        for entity in entities:
            if isinstance(entity, dict):
                # Validate entity structure
                validated_entity = {
                    "name": str(entity.get("name", "")).strip(),
                    "type": str(entity.get("type", "")).strip().lower(),
                    "role": str(entity.get("role", "")).strip(),
                    "company": str(entity.get("company", "")).strip(),
                    "contacts": [],
                    "locations": []
                }
                
                # Process contacts
                contacts = entity.get("contacts", [])
                if isinstance(contacts, list):
                    validated_entity["contacts"] = [str(c).strip() for c in contacts if c]
                elif isinstance(contacts, str) and contacts.strip():
                    validated_entity["contacts"] = [contacts.strip()]
                
                # Process locations
                locations = entity.get("locations", [])
                if isinstance(locations, list):
                    validated_entity["locations"] = [str(l).strip() for l in locations if l]
                elif isinstance(locations, str) and locations.strip():
                    validated_entity["locations"] = [locations.strip()]
                
                if validated_entity["name"]:  # Only add if name exists
                    structured_data["entities"].append(validated_entity)
    
    # Extract summary
    structured_data["summary"] = str(data.get("summary", "")).strip()
    
    # Extract standalone items (not linked to specific entities)
    standalone_locations = data.get("standalone_locations", [])
    if isinstance(standalone_locations, list):
        structured_data["standalone_locations"] = [str(l).strip() for l in standalone_locations if l]
    
    standalone_contacts = data.get("standalone_contacts", [])
    if isinstance(standalone_contacts, list):
        structured_data["standalone_contacts"] = [str(c).strip() for c in standalone_contacts if c]
    
    # Calculate metadata
    people_count = len([e for e in structured_data["entities"] if e["type"] == "person"])
    company_count = len([e for e in structured_data["entities"] if e["type"] == "company"])
    total_contacts = len(structured_data["standalone_contacts"]) + sum(len(e["contacts"]) for e in structured_data["entities"])
    total_locations = len(structured_data["standalone_locations"]) + sum(len(e["locations"]) for e in structured_data["entities"])
    
    structured_data["metadata"] = {
        "total_people": people_count,
        "total_companies": company_count,
        "total_contacts": total_contacts,
        "total_locations": total_locations
    }
    
    return structured_data

def get_all_info_structured(scraped_news: str) -> Dict[str, Any]:
    """
    Extract structured information with entity relationships from scraped news content.
    
    Args:
        scraped_news (str): The news content to analyze
        
    Returns:
        Dict[str, Any]: Extracted information with entity relationships
    """
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        
        # Input validation
        if not scraped_news or not scraped_news.strip():
            logger.warning("Empty or invalid news content provided")
            return {
                "entities": [],
                "summary": "",
                "standalone_locations": [],
                "standalone_contacts": [],
                "metadata": {"total_people": 0, "total_companies": 0, "total_contacts": 0, "total_locations": 0},
                "error": "No content provided"
            }
        
        # Enhanced prompt for entity relationship extraction
        prompt = f"""
        Analyze the following news content and extract structured information about people, companies, and their relationships.
        Create entities that link people to their companies, roles, and contact information.

        News Content:
        {scraped_news}

        Instructions:
        1. Identify people and companies mentioned in the text
        2. Link people to their companies and roles when mentioned
        3. Associate contact information (emails, phones, social media) with specific people/companies when possible
        4. Associate locations with specific people/companies when mentioned together
        5. Create a brief summary of the news content

        Return ONLY a valid JSON object with this exact structure:
        {{
            "entities": [
                {{
                    "name": "Person/Company Name",
                    "type": "person" or "company",
                    "role": "job title or role (if mentioned)",
                    "company": "associated company name (for people only)",
                    "contacts": ["email@example.com", "phone_number", "social_handle"],
                    "locations": ["associated locations"]
                }}
            ],
            "summary": "Brief description of what the news is about",
            "standalone_locations": ["locations not tied to specific entities"],
            "standalone_contacts": ["contacts not tied to specific entities"]
        }}

        Guidelines:
        - Only extract information explicitly mentioned in the text
        - For people: include their role and company if mentioned
        - For companies: include associated contacts and locations
        - Keep contacts and locations arrays empty if none are found
        - Use empty strings for role/company if not mentioned
        """
        
        # System instruction for better control
        system_instruction = """
        You are a precise entity relationship extraction assistant.
        Focus on creating clear relationships between people, companies, and their associated information.
        Extract only factual information explicitly mentioned in the text.
        Structure the response to show clear connections between entities and their attributes.
        Return only valid JSON with no additional text.
        """
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # Generate content with optimized settings
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                max_output_tokens=800,
                top_p=0.1,
                top_k=1
            )
        )
        
        # Extract response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        logger.info(f"Raw response: {response_text}")
        
        # Parse JSON from response
        extracted_data = extract_json_from_response(response_text)
        
        if extracted_data is None:
            logger.error("Failed to extract valid JSON from response")
            return {
                "entities": [],
                "summary": "",
                "standalone_locations": [],
                "standalone_contacts": [],
                "metadata": {"total_people": 0, "total_companies": 0, "total_contacts": 0, "total_locations": 0},
                "error": "Failed to parse response"
            }
        
        # Validate and structure the extracted data
        structured_data = validate_and_structure_entities(extracted_data)
        
        logger.info("Successfully extracted structured information")
        return structured_data
        
    except Exception as e:
        logger.error(f"Error in get_all_info_structured: {str(e)}")
        return {
            "entities": [],
            "summary": "",
            "standalone_locations": [],
            "standalone_contacts": [],
            "metadata": {"total_people": 0, "total_companies": 0, "total_contacts": 0, "total_locations": 0},
            "error": str(e)
        }

def print_structured_results(data: Dict[str, Any]) -> None:
    """
    Print the structured results in a human-readable format.
    """
    print("=" * 60)
    print("EXTRACTED INFORMATION SUMMARY")
    print("=" * 60)
    
    # Print metadata
    metadata = data.get("metadata", {})
    print(f"📊 STATISTICS:")
    print(f"   • People: {metadata.get('total_people', 0)}")
    print(f"   • Companies: {metadata.get('total_companies', 0)}")
    print(f"   • Contacts: {metadata.get('total_contacts', 0)}")
    print(f"   • Locations: {metadata.get('total_locations', 0)}")
    print()
    
    # Print summary
    summary = data.get("summary", "")
    if summary:
        print(f"📄 NEWS SUMMARY:")
        print(f"   {summary}")
        print()
    
    # Print entities with relationships
    entities = data.get("entities", [])
    if entities:
        print("👥 PEOPLE & COMPANIES:")
        print("-" * 40)
        
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity['name']}")
            print(f"   Type: {entity['type'].title()}")
            
            if entity.get('role'):
                print(f"   Role: {entity['role']}")
            
            if entity.get('company') and entity['type'] == 'person':
                print(f"   Company: {entity['company']}")
            
            if entity.get('contacts'):
                print(f"   Contacts: {', '.join(entity['contacts'])}")
            
            if entity.get('locations'):
                print(f"   Locations: {', '.join(entity['locations'])}")
            
            print()
    
    # Print standalone information
    standalone_contacts = data.get("standalone_contacts", [])
    if standalone_contacts:
        print("📞 ADDITIONAL CONTACTS:")
        for contact in standalone_contacts:
            print(f"   • {contact}")
        print()
    
    standalone_locations = data.get("standalone_locations", [])
    if standalone_locations:
        print("📍 ADDITIONAL LOCATIONS:")
        for location in standalone_locations:
            print(f"   • {location}")
        print()
    
    # Print errors if any
    if "error" in data:
        print(f"⚠️  ERROR: {data['error']}")

def export_to_formats(data: Dict[str, Any], filename_base: str = "extracted_info") -> None:
    """
    Export the structured data to different formats for easy use.
    """
    # Export to JSON
    with open(f"{filename_base}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Export to CSV-like format
    with open(f"{filename_base}.csv", "w", encoding="utf-8") as f:
        f.write("Name,Type,Role,Company,Contacts,Locations\n")
        for entity in data.get("entities", []):
            contacts = "; ".join(entity.get("contacts", []))
            locations = "; ".join(entity.get("locations", []))
            f.write(f'"{entity["name"]}","{entity["type"]}","{entity.get("role", "")}","{entity.get("company", "")}","{contacts}","{locations}"\n')
    
    # Export to readable text format
    with open(f"{filename_base}.txt", "w", encoding="utf-8") as f:
        f.write("EXTRACTED INFORMATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        if data.get("summary"):
            f.write(f"SUMMARY:\n{data['summary']}\n\n")
        
        f.write("ENTITIES:\n")
        f.write("-" * 20 + "\n")
        
        for entity in data.get("entities", []):
            f.write(f"Name: {entity['name']}\n")
            f.write(f"Type: {entity['type'].title()}\n")
            if entity.get('role'):
                f.write(f"Role: {entity['role']}\n")
            if entity.get('company'):
                f.write(f"Company: {entity['company']}\n")
            if entity.get('contacts'):
                f.write(f"Contacts: {', '.join(entity['contacts'])}\n")
            if entity.get('locations'):
                f.write(f"Locations: {', '.join(entity['locations'])}\n")
            f.write("\n")
    
    print(f"✅ Data exported to: {filename_base}.json, {filename_base}.csv, {filename_base}.txt")

def main():
    """
    Example usage of the enhanced information extraction function.
    """
    # Example news content with more complex relationships
    sample_news = """
    Tech giant Microsoft announced that CEO Satya Nadella will be speaking at the upcoming 
    AI Summit conference in Seattle next month. The company's new AI initiative will be 
    launched from their headquarters in Redmond, Washington. Sarah Chen, VP of AI Research 
    at Microsoft, will also present the technical details. For media inquiries, contact 
    press@microsoft.com or reach out to Microsoft's communication team at +1-425-882-8080. 
    The conference will also feature speakers from Google, including Dr. John Smith, 
    Senior AI Researcher, who can be reached at j.smith@google.com. The event is being 
    held at the Seattle Convention Center.
    """
    
    # Extract structured information
    result = get_all_info_structured(sample_news)
    
    # Print results in a readable format
    print_structured_results(result)
    
    # Export to multiple formats
    export_to_formats(result, "sample_extraction")
    
    # Also print raw JSON for debugging
    print("\n" + "="*60)
    print("RAW JSON OUTPUT:")
    print("="*60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
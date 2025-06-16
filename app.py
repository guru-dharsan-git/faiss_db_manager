import json
from db_rectifier import check_and_add_json
from security_layer import classify_news_as_lead
classification=classify_news_as_lead("This is a news about a company")
def process_and_add(json_file):
    with open(json_file, "r",encoding="utf-8") as f:
        data = json.load(f)
    for i in data:
        example=(i["title"]+", details :"+i["description"])

        classification = classify_news_as_lead(example)
        print(json.dumps(classification, indent=2))
process_and_add("cnn_company.json")
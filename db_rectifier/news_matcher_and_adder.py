from .similarity_matcher import NewsSimilarityDB
from .llm_response import llm_true_false
db=NewsSimilarityDB()
import json
db.load_database()

def check(text):
    check=db.check_similarity(text)
    if check==[]:
        db.add_news(text)
    for i in check:
        return float(i["score"])
def extract_similar_news_fromdb(news):
    a=db.check_similarity(news)
    for i in a:
        return(i["news"])   


def check_and_add_json(json_file):
    count=0
    l=[]
    with open(json_file, "r",encoding="utf-8") as f:
        data = json.load(f)
    for i in data:
        news=str(i["title"]+", details :"+i["description"])
        print(i["title"])
        news_score=check(news)
        print(news_score)
        if (float(news_score)<0.8):
            print("news irrelevant")
            db.add_news(news)
        elif (float(news_score)>0.8 and float(news_score)<0.9):
            print("calling llm")
            db_news=extract_similar_news_fromdb(news)
            res=llm_true_false(news,db_news)
            if(res):
                print("same news")
            else:
                print("llm said not true")
                db.add_news(news)
        if (float(news_score)>0.9):
            print("news similar")
            count+=1
            db_news=extract_similar_news_fromdb(news)
            print(news)
            print(db_news)

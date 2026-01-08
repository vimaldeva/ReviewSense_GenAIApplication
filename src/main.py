import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import json
import plotly.graph_objects as go
import nbformat
import plotly.io as pio
from langchain_openai import ChatOpenAI
import os
import getpass

# Loading the Input Dataset
input_path = "../data/Udemy_user_review.csv"
comments_df = pd.read_csv(input_path)

summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small",
    min_length=10,
    max_length=50
)

# Processing review summaries with a sentiment-analysis model
reviews = comments_df["Comments"].fillna("").astype(str).tolist()
review_summary = []

for review in reviews:
    summary = summarizer(f"summarize: {review}")[0]['summary_text']
    review_summary.append(summary)
    
comments_df["summary"] = [r for r in review_summary]
comments = comments_df["Comments"].fillna("").astype(str).tolist()

sentiment_pipeline = pipeline("sentiment-analysis")
results = sentiment_pipeline(comments, batch_size=32, truncation=True)

# Storing the enriched Data in a CSV file
comments_df["sentiment_label"] = [r["label"] for r in results]
comments_df.to_csv('../output/comments_enriched.csv', index=False)

# Getting the overal review summary for the given product.
reviews = (
    comments_df["summary"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# remove trailing periods so we don't end up with ".."
cleaned = [text.rstrip(".") for text in reviews if text]

concatenated_reviews = ". ".join(cleaned)

if concatenated_reviews and not concatenated_reviews.endswith("."):
    concatenated_reviews += "."
    
system_prompt = """
You are an expert at summarizing product reviews. You will receive all comments for a product as a single concatenated text. Carefully read the reviews and produce a summary of greated than 400 words and less than 600 words. You should miss out any important imformation. Each section should have atleast 4 points

Organize the summary into the following five sections, each with the exact heading specified:

Overall Feedback – Provide the overall sentiment and key takeaways.
Pros – List the positive aspects users highlighted.
Cons – List the negative points users raised.
Technologies taught - List the technologies, tools, programming languages , services and other things that are taught in this course
Areas of Improvement – Identify what users feel should be improved.
Who Is This Course For – State who would benefit most from the course and who might find it unsuitable.
Follow this format precisely."""


os.environ["OPENAI_API_KEY"] = "Your OpenAI API Key"

llm = ChatOpenAI(
    model="gpt-5-nano",
)

messages = [
        ("system", system_prompt ),
        ("human", concatenated_reviews), ]

ai_msg = llm.invoke(messages)

review_summary = ai_msg.text

# Storing the Review summary in a text file
with open("../output/review_summary.text", "w", encoding="utf-8") as f:
    f.write(review_summary)
    
total_rows = len(comments_df)          # or comments_df.shape[0]

# Calculating the Positive reviews summary
positive_rows = (comments_df["sentiment_label"] == "POSITIVE").sum()
negative_rows = (comments_df["sentiment_label"] == "NEGATIVE").sum()

positive_comments_ratio = round((positive_rows/total_rows)*100)

print(f"Total rows: {total_rows}")
print(f"Positive rows: {positive_rows}")
print(f"Negative rows: {negative_rows}")
print(f"Positive Ratio rows: {positive_comments_ratio}")


#Storing the sentiment summary in a JSON File
sentiment_trend = {
    "Total Reviews": int(total_rows),
    "Positive Reviews": int(positive_rows),
    "Negative Reviews": int(negative_rows),
    "Positive Reviews Ratio": float(positive_comments_ratio)
    }

print(sentiment_trend)

sentiment_trend_json = json.dumps(sentiment_trend, indent=4)

with open("../output/sentiment_trend.json", "w", encoding="utf-8") as f:
    json.dump(sentiment_trend, f, indent=4)


#Displaying the Sentiment summary in a graphical manner
pio.renderers.default = "notebook_connected"

value = positive_comments_ratio  # satisfaction %

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': "%"},
        title={'text': "Positive Reviews (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "rgba(0,0,0,0)"},          # hide the center bar
            'steps': [
                {'range': [0, value], 'color': "#5cb85c"},
                {'range': [value, 100], 'color': "#d9534f"},
            ],
            'threshold': {
                'line': {'color': "navy", 'width': 2},
                'thickness': 1,
                'value': value
            },
        }
    )
)

fig.show()
# fig.write_image("../output/positive_reviews_gauge.png")
# fig.show(renderer="notebook_connected")

# Databricks notebook source
# MAGIC %pip install transformers torch pandas mlflow torchvision
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %sh
# MAGIC export HF_HOME=/tmp/huggingface_cache
# MAGIC export TRANSFORMERS_CACHE=/tmp/huggingface
# MAGIC

# COMMAND ----------

import mlflow
import transformers

# Login to a new experiment
mlflow.set_experiment("/Users/gazmendhalili2016@gmail.com/mlflow_experiments")

with mlflow.start_run() as run:
    # Log the model to MLflow's managed storage
    mlflow.transformers.log_model(
        transformers_model=transformers.pipeline("text-classification", model="ProsusAI/finbert"),
        name="finbert_model"
    )
    
    # Get the unique Run ID to use in your UDF
    run_id = run.info.run_id
    print(f"✅ Model saved! COPY THIS RUN ID: {run_id}")

# COMMAND ----------

from nltk.tokenize import sent_tokenize
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import ArrayType, StringType
from transformers import pipeline

import nltk
import numpy as np
import pandas as pd
import torch
nltk.download('punkt')

SENTIMENT_PIPE, SENTENCE_TOKENIZATION_PIPE = None, None

def initialize_models():
    """Initializes the heavy Hugging Face models once per worker process."""
    global SENTIMENT_PIPE, SENTENCE_TOKENIZATION_PIPE
    import mlflow
    
    run_id = "ac927700cba046e895fa3a4592b8411c" 
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="finbert_model")

    if SENTIMENT_PIPE is None:
        # FinBERT for sentiment classification
        SENTIMENT_PIPE = pipeline(
            "text-classification", 
            model=local_path, 
            return_all_scores=True, 
            device=-1 # Use CPU or GPU if available
        )
        
    if SENTENCE_TOKENIZATION_PIPE is None:
        # Simple text segmentation model (or equivalent for fast splitting)
        # Using a general tokenizer/splitter as Hugging Face doesn't have a dedicated "sent_tokenize" model.
        # Often a separate library or Spark NLP is better for this, but we use a Transformer-based tokenizer.
        # NOTE: For simplicity, we'll use a standard Python function, but keep the caching structure.
        from nltk.tokenize import sent_tokenize
        import nltk
        nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True) # Ensure resource is on worker
        SENTENCE_TOKENIZATION_PIPE = sent_tokenize

@pandas_udf('double')
def calculate_contextual_sentiment(
    text_series: pd.Series, 
    company_names: pd.Series
) -> pd.Series:
    """
    Calculates the aggregate sentiment score for an article based ONLY on sentences 
    that mention the corresponding company name.
    """
    
    # 1. Initialize models (only runs once per worker)
    initialize_models()
    
    final_scores = []
    relevant_sentences = []
    
    # 2. Iterate through the corresponding article content and company name pairs
    # zip() ensures we process the rows in the batch together.
    for article_content, company_name in zip(text_series, company_names):
        
        if not article_content:
            final_scores.append(0.0) # Assign neutral score if no content
            continue

        # 3. Split the article into sentences (using the initialized function)
        sentences = SENTENCE_TOKENIZATION_PIPE(article_content)
        lower_company = company_name.lower()
        
        # 4. Filter the sentences based on the company name
        relevant_sentences.append (
            s for s in sentences
            if lower_company in s.lower()
        )
        
        # 5. Feed relevant sentences to FinBERT for sentiment score
        if not relevant_sentences:
            final_scores.append(0.0) # Assign neutral score if no relevant sentences
            continue

        # Run inference on the list of relevant sentences
        # The model automatically processes the batch of sentences efficiently.
        results = SENTIMENT_PIPE(relevant_sentences, truncation=True, max_length=512)
        
        # 6. Calculate the aggregate sentiment score
        sentence_scores = []
        for res in results:
            # Extract scores for Positive and Negative labels
            pos = next(item['score'] for item in res if item['label'] == 'positive')
            neg = next(item['score'] for item in res if item['label'] == 'negative')
            
            # Compound score (-1 to 1)
            sentence_scores.append(pos - neg)
        
        # Calculate the final article score as the mean of all relevant sentence scores
        if sentence_scores:
            final_scores.append(np.mean(sentence_scores))
        else:
            final_scores.append(0.0)

    # 7. Return the final scores as a Pandas Series
    return pd.Series(final_scores)

@pandas_udf('string')
def filter_important_sentences(
    articles: pd.Series, 
    company_names: pd.Series
) -> pd.Series:
    """
    Calculates the aggregate sentiment score for an article based ONLY on sentences 
    that mention the corresponding company name.
    """
    
    # 1. Initialize models (only runs once per worker)
    initialize_models()
    
    relevant_sentences = []
    
    # 2. Iterate through the corresponding article content and company name pairs
    # zip() ensures we process the rows in the batch together.
    for article_content, company_name in zip(articles, company_names):
        
        if not article_content:
            continue

        # 3. Split the article into sentences (using the initialized function)
        sentences = SENTENCE_TOKENIZATION_PIPE(article_content)
        lower_company = company_name.lower()
        
        # 4. Filter the sentences based on the company name
        relevant_sentences.append (
            s for s in sentences
            if lower_company in s.lower()
        )

    return pd.Series(relevant_sentences)

# COMMAND ----------

import pyspark.sql.functions as sf

companies_df = spark.read.table("stock_prediction.default.companies")

# Removes punctuation
companies_df = companies_df.withColumn(
    "clean_name", 
    sf.regexp_replace(sf.col("name"), '[^a-zA-Z0-9\\s]', '')
)

regex_pattern = r"\b(Inc|Corporation|Incorporated|Corp|Ltd|Co)\b"
companies_df = companies_df.withColumn(
    "clean_name", 
    sf.regexp_replace(sf.col("clean_name"), regex_pattern, '')
)

companies_df = companies_df.withColumn(
    "clean_name",
    sf.lower(sf.col("clean_name"))
)

companies_df = companies_df.withColumnRenamed("id", "company_id")

display(companies_df)

# COMMAND ----------

from pyspark.sql.functions import col, expr

articles_df = spark.read.table("stock_prediction.default.articles")

# Cross join articles with companies to check for company mentions in article content
joined_df = articles_df.crossJoin(companies_df)

# Filter where clean_name is contained in content_cleaned (case-insensitive)
filtered_df = joined_df.filter(
    expr("lower(content_cleaned) LIKE concat('%', clean_name, '%')")
)

# Group by company and collect articles mentioning each company
grouped_df = filtered_df.groupBy("clean_name").agg(
    sf.collect_list("id").alias("article_ids"),
    sf.collect_list("title").alias("article_titles"),
    sf.collect_list("content").alias("article_contents")
)

display(grouped_df.select("clean_name", "article_ids", "article_titles"))

# COMMAND ----------


# 2. Apply the UDF
# This adds a new column 'sentiment_score' to your DataFrame
analyzed_df = grouped_df.withColumn(
    "sentiment_score", 
    calculate_contextual_sentiment(grouped_df["article_contents"], grouped_df["clean_name"])
)

# analyzed_df.show()
# 3. Save the results (e.g., to a new table or overwrite)
# analyzed_df.createOrReplaceTempView("stock_prediction.default.articles_with_sentiment_view")
analyzed_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("stock_prediction.default.articles_with_sentences")

# Verify
# display(analyzed_df.limit(2))

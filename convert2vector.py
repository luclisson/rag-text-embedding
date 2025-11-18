from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StringType, StructField, StructType
from qdrant_client import QdrantClient,models
from qdrant_client.models import Distance, VectorParams
import pandas as pd


#spark = SparkSession.builder.appName("convert2vector").getOrCreate() #spark session
# the way this spark session is build is outdated. spark wont find the necessary jar files

SPARK_NLP_VERSION = "6.2.2"
spark = (
    SparkSession.builder
    .appName("VectorConversion")
    .config(
        "spark.jars.packages", 
        f"com.johnsnowlabs.nlp:spark-nlp_2.12:{SPARK_NLP_VERSION}"
    )
    .master("local[*]")
    .getOrCreate()
)

text_data = (
  spark
  .read
  .text("star-wars.txt")
  .withColumnRenamed("value", "text")
)

text_data.show()

document_assembler = (
    DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")
)

sentence_detector = (
    SentenceDetector()
      .setInputCols(["document"])
      .setOutputCol("sentence")
)

tokenizer = (
    Tokenizer()
      .setInputCols(["sentence"])
      .setOutputCol("token")
)


embeddings = DistilBertEmbeddings.pretrained("distilbert_base_cased", "en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

pipeline = Pipeline(
    stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
    ]
)

result = pipeline.fit(text_data).transform(text_data)

sentence_and_embeddings_df = result.selectExpr(
    "explode(sentence.result) as sentence",
    "explode(embeddings.embeddings) as vector"
)

df_qdrant = (
    sentence_and_embeddings_df
    .withColumn("id", (monotonically_increasing_id()+1))
    .withColumnRenamed("vector", "values")
    .withColumn("metadata", sentence_and_embeddings_df.sentence)
    .drop("sentence")  # Store sentence as metadata, drop original column
)

print("loading to qdrant using pandas dataframe mapping of spark dataframe")

QDRANT_URL = "http://192.168.0.59:6333"
COLLECTION_NAME = "starwars-rag"
VECTOR_DIMENSION = 768 # DistilBERT embedding size

client = QdrantClient(url="http://192.168.0.59:6333")
print(f"Recreating collection '{COLLECTION_NAME}' with size={VECTOR_DIMENSION}...")
if (client.collection_exists(collection_name=COLLECTION_NAME)):
    print(f"collection {COLLECTION_NAME} is existing. Skipping collection creating")
else:
    print(f"creating collection {COLLECTION_NAME}")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE),
    )
    print("converting spark dataframe to pandas dataframe")
    pandas_df = df_qdrant.toPandas()
    points_to_upsert = [
        models.PointStruct(
            id=row['id'],
            vector=row['values'], # 'values' column holds the 768-dim vector
            payload={'text': row['metadata']} 
        )
        for index, row in pandas_df.iterrows()
    ]
    print(f"loading the data to {COLLECTION_NAME}")
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points_to_upsert
    )
    print(f"Data successfully loaded! Status: {operation_info.status}")

queries = [
    Row(text = "What was the name and capability of the Empire's ultimate weapon?"),
    Row(text= "What is a distinguishing characteristic of Darth Vader, and who is his son?")
]

schema = StructType([StructField("text", StringType(), True)])
query_df = spark.createDataFrame(queries, schema)
transformed_query = pipeline.fit(query_df).transform(query_df)

transformed_query = transformed_query.selectExpr(
    "explode(sentence.result) as sentence",  # Extract sentences
    "explode(embeddings.embeddings) as vector"  # Extract embeddings
)

query_vector_list = (
    transformed_query
    .select("vector")
    .rdd
    .flatMap(lambda x: x)
    .collect()
)

TOP_K = 3

print("\n--- Starting Qdrant Vector Search ---")

original_query_texts = [row['text'] for row in queries]
for i, query_vector in enumerate(query_vector_list):
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=TOP_K,
        with_payload=True
    )
    
    print(f"\n✨ Results for Query {i+1}: '{original_query_texts[i]}'")
    
    for j, result in enumerate(search_result):
        # The relevant sentence is stored in the payload's 'text' field
        retrieved_text = result.payload.get('text', 'No text found')
        
        # The score is the cosine similarity
        print(f"  {j+1}. Score: {result.score:.4f} | Text: \"{retrieved_text}\"")

spark.stop()
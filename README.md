# rag-text-embedding
the python file will do text embedding using a small DistilBert model together with a spark nlp pipeline. The created vectors will be transfered to a qdrant db which will also handle the querrying.


## how to use
- install requirements.txt using pip with python version 3.8.* (i used 3.8.18)
- install the qwen3 embedding model from [john snow](https://sparknlp.org/2025/08/04/Qwen3_Embedding_0.6B_Q8_0_gguf_en.html) to use the model locally on the computer
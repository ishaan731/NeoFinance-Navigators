# Neo Finance Navigators AI Chatbot
## Overview

Welcome to the Neo Finance Navigators AI Chatbot! This project utilizes the Microsoft Phi-1.5 model along with Google Colab's T4 GPU to deliver accurate and efficient responses through Retrieval Augmented Generation (RAG). 

Our chatbot is designed to handle financial queries with precision using advanced machine learning techniques.

## Features

- **Powered by Microsoft Phi-1.5**: The chatbot employs the Microsoft Phi-1.5 model, known for its rapid response times and high accuracy. It effectively processes and responds to user queries using the datasets from [youdata.ai](youdata.ai). 
The main question/answer datasets were uploaded by us to the platform and another dataset for S&P 500 companies was used by us.

[Dataset 1 - MAIN](https://www.youdata.ai/datasets/66791a6f202579220b6304cb?source_link=&source_platform=&data_interests=%2CArtificial+Intelligence%2CMachine+Learning%2CEducation)

[Dataset 2](https://www.youdata.ai/datasets/661d173d279aa8ef92326d9e#)

- **Retrieval Augmented Generation (RAG)**: By leveraging RAG, the chatbot retrieves and generates responses that are both contextually relevant and precise. The retrieval process is facilitated by a vector database built from documents provided.

- **Free and Paid Versions**:
  - **Free Version**: We plan to use the Phi-1.5 model to provide a balance of speed and accuracy at no cost. It requires less resources, faster on slow hardawre.
  - **Paid Version**: Utilizes the Phi-3 Mini 4k Instruct model for enhanced performance and greater resource efficiency in the paid tier. Uses more resources, better answer quality.

## Code Overview

### Imports

The project requires several Python packages. Ensure they are installed using:
`
`pip install llama-index`

`pip install llama-index-embeddings-huggingface`

`pip install peft`

`pip install auto-gptq`

`pip install optimum`

`pip install bitsandbytes`

### Define Settings

### The settings for the model are configured as follows:

```
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25
```

## Read and Store Documents
### Documents are read from the "articles" directory and stored in a vector database:
```
documents = SimpleDirectoryReader("articles").load_data()
index = VectorStoreIndex.from_documents(documents)
```

## Set Up Search Function
### Configure the retriever and query engine:
```
top_k = 3

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=top_k,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)
```

## Retrieve and Process Responses
### The chatbot processes user queries and retrieves relevant documents:
```
query = input("Query:")
response = query_engine.query(query)

context = " ".join([n.text for n in response.source_nodes[:top_k]]) + "."
```

## Import and Use Phi-1.5
### The Microsoft Phi-1.5 model is loaded and used to generate responses:
```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt_template_w_context = lambda context, comment: f"""NeoGPT, a customer service-based chatbot for all financial-related customer queries. Respond accurately to the person's queries.
{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

comment = query
prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

outputs = model.generate(input_ids=input_ids, max_new_tokens=280)

output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def process_response(text):
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        sentences = line.split('. ')
        processed_lines.extend([sentence + '.' for sentence in sentences if sentence.strip()])
    return "\n".join(processed_lines)

formatted_response = process_response(output_text)
print(formatted_response)


```


## Advantages of Phi-1.5
- Speed: Compared to larger LLMs, Phi-1.5 provides faster response times, making it ideal for applications that require quick and efficient interactions.
- Efficiency: Phi-1.5 offers a good balance between performance and resource consumption, making it a cost-effective choice for many use cases.
- Cost: Using Phi-1.5 with T4 GPU can be potentially free of cost. Once uploaded to huggingface spaces, it can be used free of cost as spaces provides the T4 GPU for free.


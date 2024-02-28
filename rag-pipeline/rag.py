import weaviate
import weaviate.classes as wvc
import os

client = weaviate.connect_to_wcs(
    cluster_url='https://contract-53m5s98n.weaviate.network',
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
    headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY  # <-- Replace with your API key
    }
)

from typing import List
import re
from PyPDF2 import PdfReader

def load_pdf_and_chunk(file_path: str, chunk_size: int, overlap_size: int) -> List[str]:
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        pdf_text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num] 
            pdf_text += page.extract_text()
        
    pdf_text = re.sub(r"\s+", " ", pdf_text)  # Remove multiple whitespaces
    text_words = re.split(r"\s", pdf_text)  # Split text by single whitespace

    chunks = []
    for i in range(0, len(text_words), chunk_size):  # Iterate through & chunk data
        chunk = " ".join(text_words[max(i - overlap_size, 0): i + chunk_size])  # Join a set of words into a string
        chunks.append(chunk)
    return chunks

file_path = './Raptor Contract.pdf'
chunked_text = load_pdf_and_chunk(file_path, 150, 25)


import weaviate.classes as wvc


collection_name = "Contract"

if client.collections.exists(collection_name):  # In case we've created this collection before
    client.collections.delete(collection_name)  # THIS WILL DELETE ALL DATA IN THE COLLECTION

chunks = client.collections.create(
    name=collection_name,
    properties=[
        wvc.config.Property(
            name="chunk",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="chapter_title",
            data_type=wvc.config.DataType.TEXT
        ),
        wvc.config.Property(
            name="chunk_index",
            data_type=wvc.config.DataType.INT
        ),
    ],
    vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # Use `text2vec-openai` as the vectorizer
    generative_config=wvc.config.Configure.Generative.openai(),  # Use `generative-openai` with default parameters
)


chunks_list = list()
for i, chunk in enumerate(chunked_text):
    data_properties = {
        "chapter_title": "What is the STOCK PURCHASE AGREEMENT?",
        "chunk": chunk,
        "chunk_index": i
    }
    data_object = wvc.data.DataObject(properties=data_properties)
    chunks_list.append(data_object)
chunks.data.insert_many(chunks_list)

response = chunks.aggregate.over_all(total_count=True)
print(response.total_count)

response = chunks.generate.fetch_objects(
    limit=2,
    single_prompt="Write the following as a haiku: ===== {chunk} "
)

for o in response.objects:
    print(f"\n===== Object index: [{o.properties['chunk_index']}] =====")
    print(o.generated)


response = chunks.generate.fetch_objects(
    limit=2,
    grouped_task="Write a summery of the contract."
)

print(response.generated)
import streamlit as st
import tempfile
from typing import List
import re
from PyPDF2 import PdfReader
import weaviate
import weaviate.classes as wvc

# Function to chunk text from PDF
def load_pdf_and_chunk(file_path: str, chunk_size: int, overlap_size: int) -> List[str]:
    # Load PDF and extract text
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        pdf_text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num] 
            pdf_text += page.extract_text()
        
    # Preprocess text
    pdf_text = re.sub(r"\s+", " ", pdf_text)  # Remove multiple whitespaces
    text_words = re.split(r"\s", pdf_text)  # Split text by single whitespace

    # Chunk text
    chunks = []
    for i in range(0, len(text_words), chunk_size):  
        chunk = " ".join(text_words[max(i - overlap_size, 0): i + chunk_size])  
        chunks.append(chunk)
    return chunks

# Function to connect to Weaviate
def connect_to_weaviate():
    client = weaviate.connect_to_wcs(
        cluster_url='https://contract-53m5s98n.weaviate.network',
        auth_credentials=weaviate.auth.AuthApiKey('GotV3CwKLnYWCJ2MlcG2AIcb9DPzuYwtGbe7'),
        headers={
            "X-OpenAI-Api-Key": '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'  
        }
    )
    return client

# Function to create Weaviate collection
def create_weaviate_collection(client, collection_name):
    if client.collections.exists(collection_name):  
        client.collections.delete(collection_name)  

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
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.openai(),
    )
    return chunks

# Function to process PDF and insert chunks into Weaviate
def process_pdf_and_insert(client, collection_name, file_path, chunk_size, overlap_size):
    chunked_text = load_pdf_and_chunk(file_path, chunk_size, overlap_size)
    chunks_list = []
    for i, chunk in enumerate(chunked_text):
        data_properties = {
            "chapter_title": "What is the STOCK PURCHASE AGREEMENT?",
            "chunk": chunk,
            "chunk_index": i
        }
        data_object = wvc.data.DataObject(properties=data_properties)
        chunks_list.append(data_object)
    client.data.insert_many(chunks_list)

    chunks_list = list()



# Function to fetch generated text from Weaviate
def fetch_generated_text(client, collection_name, question, limit):
    response = client.generate.fetch_objects(
        collection=collection_name,
        limit=limit,
        single_prompt=question
    )
    return response

# Streamlit app
def main():
    st.title("PDF Text Interaction App")
    st.sidebar.title("Upload PDF")

    # Upload PDF file
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
    print(uploaded_file)

    if uploaded_file is not None:
        st.sidebar.text('PDF uploaded successfully!')
        st.sidebar.text('Processing PDF...')

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Connect to Weaviate
        client = connect_to_weaviate()
        collection_name = "Contract"

        # Create Weaviate collection
        chunks = create_weaviate_collection(client, collection_name)

        # Process PDF and insert chunks into Weaviate
        process_pdf_and_insert(chunks, collection_name, file_path, chunk_size=150, overlap_size=25)

        st.sidebar.text('PDF processed successfully!')
        st.sidebar.text('You can now interact with the text.')

        # Interaction option
        question = st.text_input("Enter your question:")
        if st.button("Get Response"):
            response = fetch_generated_text(client, collection_name, question, limit=1)
            for o in response.objects:
                st.text(o.generated)

if __name__ == "__main__":
    main()

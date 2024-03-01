import os
import sys
from dotenv import load_dotenv
import pandas as pd
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
sys.path.append('/home/mubarek/all_about_programing/10x_projects/AI-Contract-Bot')
from scripts.QA_dataset import get_evaluation_data
import weaviate
from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


def load_env_vars():
    """
    Load environment variables.
    """
    load_dotenv()
    return os.environ["OPENAI_API_KEY"]


def load_data(loader):
    """
    Load data using the provided loader.
    """
    return loader.load()


def split_text(text_splitter, data):
    """
    Split text using the provided text splitter.
    """
    return text_splitter.split_documents(data)


def create_client(OPENAI_KEY):
    """
    Create a Weaviate client.
    """
    return weaviate.Client(
        url="http://localhost:8080",
        additional_headers={"X-OpenAI-Api-Key": OPENAI_KEY},
        startup_period=10
    )


def define_input_structure(client):
    """
    Define input structure for Weaviate client.
    """
    client.schema.delete_all()
    client.schema.get()
    schema = {
        "classes": [
            {
                "class": "Chatbot",
                "description": "Documents for chatbot",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {"text2vec-openai": {"model": "ada", "type": "text"}},
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the paragraph",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False,
                            }
                        },
                        "name": "content",
                    },
                ],
            },
        ]
    }
    client.schema.create(schema)
    return Weaviate(client, "Chatbot", "content", attributes=["source"])


def load_text_into_vectorstore(vectorstore, docs):
    """
    Load text into the vectorstore.
    """
    text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
    texts, meta = list(zip(*text_meta_pair))
    vectorstore.add_texts(texts, meta)


def create_evaluation_chains():
    """
    Create evaluation chains.
    """
    return RagasEvaluatorChain(metric=faithfulness), RagasEvaluatorChain(metric=answer_relevancy), RagasEvaluatorChain(metric=context_precision), RagasEvaluatorChain(metric=context_recall)


def evaluate_individual_questions(chain, vectorstore, faithfulness_evaluator, relevancy_evaluator, precision_evaluator, recall_evaluator, question_answer_pairs):
    """
    Evaluate each question individually using RagasEvaluatorChain.
    """
    results_df = pd.DataFrame(columns=["Query", "Expected Answer", "Generated Answer", "Faithfulness Score", "Answer Relevancy Score", "Context Precision Score", "Context Recall Score"])
    for question_answer_pair in question_answer_pairs:
        relevant_docs = vectorstore.similarity_search(question_answer_pair["query"], k=5)
        context = ' '.join([doc.page_content for doc in relevant_docs])
        chain_result = chain.run(input_documents=relevant_docs, question=question_answer_pair["query"])
        evaluation_result = {
            "result": chain_result, 
            "context": context, 
            "source_documents": relevant_docs, 
            "query": question_answer_pair["query"], 
            "ground_truths": question_answer_pair["ground_truths"],
            "answer": chain_result
        }
        print(evaluation_result)
        results_df = pd.concat([results_df, pd.DataFrame([{
            "Query": question_answer_pair["query"],
            "Expected Answer": question_answer_pair["ground_truths"][0],
            "Generated Answer": evaluation_result["answer"],
            "Faithfulness Score": faithfulness_evaluator(evaluation_result)["faithfulness_score"],
            "Answer Relevancy Score": relevancy_evaluator(evaluation_result)["answer_relevancy_score"],
            "Context Precision Score": precision_evaluator(evaluation_result)["context_precision_score"],
            "Context Recall Score": recall_evaluator(evaluation_result)["context_recall_score"]
        }])], ignore_index=True)
    return results_df


def main():
    OPENAI_KEY = load_env_vars()
    loader = DirectoryLoader('./files', glob="**/*.pdf")
    data = load_data(loader)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=256)
    docs = split_text(text_splitter, data)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    client = create_client(OPENAI_KEY)
    vectorstore = define_input_structure(client)
    load_text_into_vectorstore(vectorstore, docs)
    chain = load_qa_chain(
        OpenAI(openai_api_key=OPENAI_KEY, temperature=0),
        chain_type="stuff")
    eval_questions, eval_answers = get_evaluation_data()
    question_answer_pairs = [
        {"query": q, "ground_truths": [eval_answers[i]]}
        for i, q in enumerate(eval_questions)
    ]
    faithfulness_evaluator, relevancy_evaluator, precision_evaluator, recall_evaluator = create_evaluation_chains()
    results_df = evaluate_individual_questions(chain, vectorstore, faithfulness_evaluator, relevancy_evaluator, precision_evaluator, recall_evaluator, question_answer_pairs)
    print(results_df)
    results_df.to_csv('evaluation_results.csv', index=False)


if __name__ == "__main__":
    main()
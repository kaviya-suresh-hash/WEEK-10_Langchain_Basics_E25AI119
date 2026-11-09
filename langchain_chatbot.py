import csv
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Model Configuration
MODEL_NAME = "gemma3:1b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.3
)

# Five Prompt Templates
prompts = {
    "Concept Explainer": ChatPromptTemplate.from_messages([
        ("system", "Explain academic concepts in simple English with examples."),
        ("human", "Explain {topic}: {question}")
    ]),

    "Programming Assistant": ChatPromptTemplate.from_messages([
        ("system", "You are a programming tutor. Explain step by step with examples."),
        ("human", "Programming topic: {topic}. Question: {question}")
    ]),

    "Assignment Helper": ChatPromptTemplate.from_messages([
        ("system", "Help students prepare clear and well-organized academic assignments."),
        ("human", "Subject: {topic}. Assignment question: {question}")
    ]),

    "Exam Preparation": ChatPromptTemplate.from_messages([
        ("system", "You are an exam preparation assistant. Give important points clearly."),
        ("human", "Topic: {topic}. Exam question: {question}")
    ]),

    "Question Answering": ChatPromptTemplate.from_messages([
        ("system", "Answer student questions accurately and briefly."),
        ("human", "Topic: {topic}. Question: {question}")
    ])
}

# Create Dataset Folder
os.makedirs("dataset", exist_ok=True)

# Save Prompt Templates
with open("dataset/prompts.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Prompt_ID", "Prompt_Name", "Prompt_Template"])

    for prompt_id, (name, template) in enumerate(prompts.items(), start=1):
        writer.writerow([prompt_id, name, str(template)])

# 20 Test Queries
test_queries = [
    ("Python", "What is Python? Give one example.", "Concept Explainer"),
    ("Java", "What is inheritance in Java?", "Concept Explainer"),
    ("DBMS", "Explain the advantages of DBMS.", "Assignment Helper"),
    ("Operating System", "What is an operating system?", "Concept Explainer"),
    ("HTML", "What are HTML tags?", "Programming Assistant"),
    ("Data Structures", "Explain stack with an example.", "Concept Explainer"),
    ("Python", "Explain the difference between list and tuple.", "Programming Assistant"),
    ("Java", "What is a constructor in Java?", "Programming Assistant"),
    ("Artificial Intelligence", "What is artificial intelligence?", "Question Answering"),
    ("Machine Learning", "Explain supervised learning.", "Concept Explainer"),
    ("DBMS", "What is a primary key?", "Exam Preparation"),
    ("Computer Networks", "What is a computer network?", "Question Answering"),
    ("Cyber Security", "Explain the importance of cybersecurity.", "Assignment Helper"),
    ("Python", "What is a function in Python?", "Programming Assistant"),
    ("Java", "Explain method overloading.", "Exam Preparation"),
    ("Data Structures", "What is a linked list?", "Concept Explainer"),
    ("LLM", "What is a Large Language Model?", "Question Answering"),
    ("RAG", "Explain Retrieval-Augmented Generation.", "Assignment Helper"),
    ("Power BI", "What is Power BI used for?", "Question Answering"),
    ("Cloud Computing", "Explain cloud computing with an example.", "Exam Preparation")
]

# Open CSV Files and Save Results
with open("dataset/user_queries.csv", "a", newline="", encoding="utf-8") as query_file, \
     open("dataset/chatbot_responses.csv", "a", newline="", encoding="utf-8") as response_file, \
     open("dataset/chain_results.csv", "a", newline="", encoding="utf-8") as result_file:

    query_writer = csv.writer(query_file)
    response_writer = csv.writer(response_file)
    result_writer = csv.writer(result_file)

    if query_file.tell() == 0:
        query_writer.writerow(["Topic", "User_Query"])

    if response_file.tell() == 0:
        response_writer.writerow(["Chatbot_Response"])

    if result_file.tell() == 0:
        result_writer.writerow([
            "Topic", "User_Query", "Prompt_Name", "Chatbot_Response"
        ])

    # Test All 20 Queries
    for index, (topic, question, prompt_name) in enumerate(test_queries, start=1):

        print(f"\nTesting Query {index}/20")
        print("Topic:", topic)
        print("Question:", question)
        print("Prompt:", prompt_name)

        selected_prompt = prompts[prompt_name]
        chain = selected_prompt | llm

        response = chain.invoke({
            "topic": topic,
            "question": question
        })

        answer = response.content

        print("Chatbot Response:")
        print(answer)

        query_writer.writerow([topic, question])
        response_writer.writerow([answer])
        result_writer.writerow([
            topic, question, prompt_name, answer
        ])

        print("Saved successfully!")

print("\nAll 20 Queries Tested Successfully!")
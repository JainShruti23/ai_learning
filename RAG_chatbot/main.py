from model import RAG

rag = RAG(
    docs_dir= r"./datastore", # Name of the directory where the documents are located
    n_retrievals=2, # Number of documents returned by the search (int):  default=4
    chat_max_tokens=3097, # Maximum number of tokens that can be used in chat memory (int)  :   default=3097
    creativeness=1.2, # How creative will the response be (float 0-2) :   default=0.7
)

print("\nType 'exit' to leave the program.")
while True:
    question = str(input("Question: "))
    if question == "exit":
        break
    answer = rag.ask(question)
    print('Response:', answer)
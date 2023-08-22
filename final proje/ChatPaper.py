from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import sys
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

#dependencies
# pip install langchain
# pip install openai
# pip install PyPDF2
# pip install faiss-cpu
# pip install tiktoken
# pip install transformers


# using gpt3.5 and langchain at its core.
# Paid 7.51 aud for key, will remove key after after 1 month
os.environ["OPENAI_API_KEY"] = "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"


#Accept 1 PDF doc.
n = str(sys.argv[1])

if os.path.isfile(n) == False or len(sys.argv)>2 or len(sys.argv)<1:
    print("invalid input")
    sys.exit()
else:
    pass


#dont worry, its loading
print("please wait")



reader = PdfReader(n)

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
        
        
# split loaded text into smaller chunks so that during information retrieval we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    #set to a small chunk size so it doesnt hit 800
    chunk_size = 512,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")


while True:
    query = input("Enter question:")
    docs = docsearch.similarity_search(query)
    
    #check token count
    with get_openai_callback() as cb:
        temp = chain.run(input_documents=docs, question=query)
    print(cb)
    
    
    print("answer: ")
    print(temp)
    
    

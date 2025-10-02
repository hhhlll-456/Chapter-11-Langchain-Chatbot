import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory 

#---Part 1:Setup---
#Securely load our API keys from the environment variable
#This is the professional way to handle sensitive keys.
os.environ["QIANFAN_ACCESS_KEY"]=os.environ["QIANFAN_ACCESS_KEY"]
os.environ["QIANFAN_SECRET_KEY"]=os.environ["QIANFAN_SECRET_KEY"]

#Part 2:Assembling the AI Core---
#Let's define the three core LEGO bricks for our application.

#Brick 1:The Model(The AI's Brain)
#We instantiate our aonnection to the Baidu Wenxin LLM.
llm = QianfanLLMEndpoint(model="ERNIE-Bot-4")
#Brick 2:The Memory(The Agent's Notepad)
memory=ConversationBufferMemory(memory_key="chat_history")

#Brick 3:The Prompt(The Instruction Manual)
#This our new,prwerful prompt template!
template="""
    You are a friendly,patient,and encouraging English tutor named Alex.
    Your goal is to help a non-English speaker practice conversational English.
    Always ask a follow-up question to keep the conversation going.
    Keep your responses concise, generally one or two sentences.
    If the user makes a grammatical error in their spoken English, you must gently correct it.First,provide the corrected version ,and
then, in a new paragraph,provide a simple, one-sentence explanation of the rule.
    Do NOT correct issue related to capitalization or punctuation, as the focus is on spoken language and not written text.
Current conversation:
{chat_history}
HUman:
{question}
AI:
"""

prompt=PromptTemplate(
    input_variables=["chat_history","question"],
    template=template
)

#Brick 4: The Chain(The Connector)
#We snap our model and prompt together into a singer, usable chain.
chain=LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True)

#---Part 3:THe Conversation Loop---
#This is where we make the program interactive.
print("Hello!I am Alex, your friendly English tutor.")
print("Let's have a conversation. You can type 'exit' when you are done.")

#This 'while True' loop will run forever until we explicitly tell it to stop
while True:
    #Get input from the user.
    user_question = input("\nYou:")

    #Check if the user wants to end the conversation.
    if user_question.lower()=="exit":
        print("Alex:Goodbey!It was nice chatting with you")
        break#This command breaks out of the 'while' loop.

    #If the user didn't type 'exit',we run our chain.
    #We pass the user's question into the 'question' variable od our prompt.
    response = chain.predict(question=user_question)

    #Print the AI's response.
    print("Alex:"+response)
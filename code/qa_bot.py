import os
from langchain_community.llms import QianfanLLMEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

#Brick 2:The Prompt(The Instruction Manual)
#This template is simple.It tells the AI its job and where to put the user's question
prompt = PromptTemplate(
    input_variable=["question"],
    template="You are a helpful AI assistant.Answer the following question:{question}"   
)

#Brick 3: The Chain(The Connector)
#We snap our model and prompt together into a singer, usable chain.
chain=LLMChain(llm=llm,prompt=prompt)

#---Part 3:THe Conversation Loop---
#This is where we make the program interactive.
print("Hello!I am your friendly Q&A Bot.Ask me anything.")
print("Type 'exit' when you are done.")

#This 'while True' loop will run forever until we explicitly tell it to stop
while True:
    #Get input from the user.
    user_question = input("\nYou:")

    #Check if the user wants to end the conversation.
    if user_question.lower()=="exit":
        print("Bot:Goodbey!It was nice chatting with you")
        break#This command breaks out of the 'while' loop.

    #If the user didn't type 'exit',we run our chain.
    #We pass the user's question into the 'question' variable od our prompt.
    response = chain.invoke({"question":user_question})

    #Print the AI's response.
    print("Bot:"+response["text"])

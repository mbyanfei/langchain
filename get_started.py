#%%
from langchain.llms import OpenAI
#%%
import os
os.environ["OPENAI_API_KEY"]="xxx"
#%%
!echo $OPENAI_API_KEY
#%%
llm = OpenAI(temperature=0.9)
#%%
text = "What would be a good company name a company that makes colorful socks?"
print(llm(text))
#%%
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
#%%
print(prompt.format(product="colorful socks"))
#%%
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
chain.run("colorful socks")
#%%
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
#%%
# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)
#%%
os.environ["SERPAPI_API_KEY"]="1486ff9c52b823c5dc7fa0c0971065f8886e5d32fe148d7f340a7ac0539bb83b"
#%%
# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)
#%%
# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
#%%
# Now let's test it out!
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
#%%
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="Hi there!")
#%%
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")

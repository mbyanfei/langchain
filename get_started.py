#%%
!pip install langchain
#%%
!pip install openai
#%%
!pip install google-search-results
#%%
from langchain.llms import OpenAI
#%%
import os
os.environ["OPENAI_API_KEY"]="xxxx"
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
os.environ["SERPAPI_API_KEY"]="yyyy"
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
#%%
conversation.predict(input="Let's talk about ChatGPT, what do you know about it?")
#%%
conversation.predict(input="Tell me a joke about ChatGPT? then write a poem about it?")
#%%
conversation.predict(input="No joke on ChatGPT? or any joke?")
#%%
conversation.predict(input="yes, go ahead ...")
#%%
conversation.predict(input="haha, funny, one more?")
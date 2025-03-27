import dspy
import os

lm = dspy.LM('openai/gpt-4o-mini', 
             api_key='???'
)
dspy.configure(lm=lm)


# Direct call to llm
# lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
# lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']

# When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.
# A signature is a declarative specification of input/output behavior of a DSPy module.
# like function signature, BUT these declare and initialize the behavior of modules
# field names matter in DSPy Signatures. You express semantic roles in plain English: a question is different from an answer, 
# a sql_query is different from python_code.
qa = dspy.ChainOfThought('question -> answer')
# WHY? replace hacking along with long brittle prmopts. This is more modular, adaptive, and reproducible vs prompts and finetunes
# DSPy compiler figures out building optimized prompt

response = qa(question="Who is Kendric Lamar?")

print(response)
print(dspy.inspect_history(n=10))

print(dspy.configure) # global change
print(dspy.context) # inside a block of code, both are thread safe

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)

# cached if same query asked again
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)

# Signature examples
# 1. Sentiment classification
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later

print(type(classify(sentence=sentence).sentiment))

# 2. Summarization
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""
summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)
print(response.summary)
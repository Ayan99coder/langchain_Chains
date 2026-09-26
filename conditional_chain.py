from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")


class Review(BaseModel):
    message: Literal["positive", "negative"] = Field(
        description="Give me the sentiment of the review"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=Review)
parser = StrOutputParser()


prompt1 = PromptTemplate(
    template="Give me the sentiment of the following feedback into positive or negative -> {text}\n{format}",
    input_variables=["text"],
    partial_variables={
        "format": pydantic_parser.get_format_instructions()
    }
)


prompt2 = PromptTemplate(
        template="Write an professional response to your customer to this Positive feedback and you are on your duty AND DONT GIVE ANY OPTION AT THIS MOMENT YOU ARE DEALING \n{feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
        template="Write an appropriate response to this Negative feedback \n{feedback}",
    input_variables=["feedback"]
)

chain = prompt1 | model | pydantic_parser


conditional_chain = RunnableBranch(
    (
        lambda x: x.message == "positive",
        prompt2 | model | parser
    ),
    (
        lambda x: x.message == "negative",
        prompt3 | model | parser
    ),
    RunnableLambda(lambda x: "Could not find sentiment")
)


merge_chain = chain | conditional_chain


output = merge_chain.invoke({
    "text": "This is a good phone"
})

print(output)
import dspy
import os

os.environ["OPENAI_API_KEY"] = (
    "???"
)

# 정보 추출
# 프롬팅의 프로그램화 + 단순화

# 기존 LLM에 DSPy 레퍼를 씌웁니다.
lm = dspy.LM("openai/gpt-4o-mini")
# DSPy 설정을 글로벌로 바꿉니다. 사용할 LLM을 설정한 LLM으로 바꿉니다.
dspy.configure(lm=lm)
# dspy.context(lm=another_lm)으로 하면 해당 블럭 안에서만 사용할 llm이 바뀝니다.


# 프롬트의 방향성을 부여하는 부분입니다.
# Module Signature 만들기. 모듈은 프롬팅 방식 중 하나를 대표합니다 (e.g. Predict, ChainOfTought 등). 시그니처는 방향성, 스키마의 인풋값, 목표값을 나타냅니다.
# "text: str -> title: str, headings: list[str], entities: list[dict[str, str]]" 아래 클래스는 옆의 인라인 버전의 같은 내용의 목표값을 더 구체적으로 나타냅니다.
# DSPy가 아래 시그니처를 모듈에 받으면 *함수의 설명 부분* 까지 모두 프레임워크가 반영하여 이에 맞는 프롬트를 짜줍니다.
# text, title, entities 같은 이름도 프레임워크가 프롬트에 녹여 넣어줍니다
# entities 같이 추가 설명이 필요하다고 느끼면 프로그래머가 desc로 입력, 해당 내용을 프레임워크가 프롬트에 추가해줍니다
class ExtractInfo(dspy.Signature):
    """Extract structured information from raw_text"""

    raw_text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

# 이제 모듈을 만듭니다. 인풋을 기반으로 내용을 Predict하는 프롬팅을 선택. 저희가 만든 시그니쳐를 사용해서 프롬트의 인풋, 아웃풋, 방향성을 정해줍니다.
module = dspy.Predict(ExtractInfo) # raw_text에서 title, heading, entries를 정해진 형식으로 Predict 해서 추출 해라! 라고하는 프롬트를 만들어 모듈에 저장해달라
print(module)
# """
# Predict(ExtractInfo(raw_text -> title, headings, entities
#     instructions='Extract structured information from raw_text'
#     raw_text = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Raw Text:', 'desc': '${raw_text}'})
#     title = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Title:', 'desc': '${title}'})
#     headings = Field(annotation=list[str] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Headings:', 'desc': '${headings}'})
#     entities = Field(annotation=list[dict[str, str]] required=True json_schema_extra={'desc': 'a list of entities and their metadata', '__dspy_field_type': 'output', 'prefix': 'Entities:'})
# ))
# """

# 우리가 정보 추출을 해야할 자료입니다.
raw_text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."

response = module(raw_text=raw_text) # 모듈에 넣어 돌리면 생성된 프롬트 + 인풋으로 LLM에 보냅니다.

print(response)
# # 받은 답변
# """
# Prediction(
#     title='Apple Inc. Announces iPhone 14',
#     headings=['Introduction', "CEO's Statement", 'New Features'],
#     entities=[{'name': 'Apple Inc.', 'type': 'Organization'}, {'name': 'iPhone 14', 'type': 'Product'}, {'name': 'Tim Cook', 'type': 'Person'}]
# )
# """
# # 실제로 4o-mini에게 보낸 요청:
# """
# 'role': 'system', 
# 'content': 'Your input fields are:\n1. `raw_text` (str)\n\n
#             Your output fields are:\n1. `title` (str)\n2. `headings` (list[str])\n3. `entities` (list[dict[str, str]]): a list of entities and their metadata\n\n
#             All interactions will be structured in the following way, with the appropriate values filled in.\n\n
#             [[ ## raw_text ## ]]\n{raw_text}\n\n[[ ## title ## ]]\n{title}\n\n[[ ## headings ## ]]\n{headings}        
#             # # note: the value you produce must adhere to the JSON schema: 
#             # {"type": "array", "items": {"type": "string"}}\n\n[[ ## entities ## ]]\n{entities}        
#             # # note: the value you produce must adhere to the JSON schema: 
#             # {"type": "array", "items": {"type": "object", "additionalProperties": {"type": "string"}}}\n\n
#             # [[ ## completed ## ]]\n\n
#             # In adhering to this structure, your objective is: \n        Extract structured information from raw_text'
# 'role': 'user', 
# 'content': '[[ ## raw_text ## ]]\nApple Inc. announced its latest iPhone 14 today.The CEO, Tim Cook, highlighted its new features in a press release.\n\n
# Respond with the corresponding output fields, starting with the field `[[ ## title ## ]]`, 
# then `[[ ## headings ## ]]` (must be formatted as a valid Python list[str]), 
# then `[[ ## entities ## ]]` (must be formatted as a valid Python list[dict[str, str]]), 
# and then ending with the marker for `[[ ## completed ## ]]`.'

# """
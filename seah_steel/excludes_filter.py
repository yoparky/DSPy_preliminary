import dspy
from typing import Dict, Any, List
from datasets import Dataset
import json
import mlflow

LOCAL_DATASET_PATH = "./data/defines/processed/excludes_bool.hf/data-00000-of-00001.arrow"
LOCAL_EXEMPTIONS_PATH = "./data/defines/processed/modified_dict_eng.json"

# mlflow 이용한 로깅
mlflow.dspy.autolog()
mlflow.set_experiment("DSPy")
# 콘솔에 mlflow ui --port 5000

def load_local_dataset() -> dict:
    return Dataset.from_file(LOCAL_DATASET_PATH)

def load_json_to_dict(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_dataset_to_dspy_examples(data_set, def_excl_dictionary: dict, start: int, end: int) -> List[dspy.Example]:
    return [
        dspy.Example(
            demand=row["Demand"],
            keyword=row["Keyword"],
            definition=def_excl_dictionary[row["Keyword"]]["Definition"], # these are lists
            special_rule_include=def_excl_dictionary[row["Keyword"]]["Special Rule_Include"], # these are lists
            special_rule_exclude=def_excl_dictionary[row["Keyword"]]["Special Rule_Exclude"], # these are lists
            common_exclusion_list=["Fittings", "Flange", "Gasket", "Bolt", "Nut", "Valve", "Seamless", "SMLS", "Alloy", "P No.2 and above", "PMI", "Austenitic", "Stainless", "STS", "Varnish", "socket"],
            raw_material_list=["slab", "plate", "steel making", "coil", "strips", "material", "filler material"],
            expected_bool=row["Boolean"]
        ).with_inputs("demand", "keyword", "definition", "special_rule_include", "special_rule_exclude", "common_exclusion_list", "raw_material_list")
        for row in data_set.select(range(start, end))
    ]

def correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    return prediction.decision_bool == example.expected_bool


# LM 설정
lm = dspy.LM( #model="openai/gpt-4o-2024-11-20",
             model="gpt-4o-mini-2024-07-18",
             api_key="???",
             cache=False # True면 재요청시 전에 뽑았던 로그만 뽑고 LLM 요청은 안보냅니다.
             )

dspy.configure(lm=lm)

# 시그니처
class SteelDataCategorizer(dspy.Signature):
    """
    Output a bool value by looking at the text given in Demand and decide if a keyword is related.
    Base your decision on the provided definition of the keyword. 
    In making your decision, consider the reasoning and context of your decision.
    If the context's core content deals with an item in the provided common_exclusion_list that is not related to our pipe manufacturing, output False.
    If the context's core content deals with an item in the provided raw_material_list that we do not need in our pipe manufacturing, output False.
    If the context indicates that the keyword is related because of an isolated appendix or a reference to another material, output False.
    If your reasoning falls into the provided special_rule_include category, output True.
    If your reasoning falls into the provided special_rule_exclude category, output False.
    """
    demand: str = dspy.InputField(desc="text to look at")
    keyword: str = dspy.InputField(desc="proposed category of text")
    definition: list[str] = dspy.InputField(desc="definition of the keyword")
    special_rule_include: list[str] = dspy.InputField(desc="list of cases to output True")
    special_rule_exclude: list[str] = dspy.InputField(desc="list of cases to output False")
    common_exclusion_list: list[str] = dspy.InputField(desc="list of topics not related to our pipe manufacture")
    raw_material_list: list[str] = dspy.InputField(desc="list of materials not related to our pipe manufacture")
    decision_bool: bool = dspy.OutputField(desc="relationship between keyword and demand")

# 모듈
passage_categorizer = dspy.ChainOfThought(SteelDataCategorizer)

# 데이터
dataset = load_local_dataset()
def_excl_dict = load_json_to_dict(LOCAL_EXEMPTIONS_PATH)

train_set = prepare_dataset_to_dspy_examples(dataset, def_excl_dict, 0, 50)
test_set = prepare_dataset_to_dspy_examples(dataset, def_excl_dict, 50, 100)
total_set = prepare_dataset_to_dspy_examples(dataset, def_excl_dict, 0, 150)
mini_set = prepare_dataset_to_dspy_examples(dataset, def_excl_dict, 0, 3)

# print(train_set[-2])

# 오리지널 프롬트 제작된 부분 정답률
evaluate_correctness = dspy.Evaluate(
    devset=test_set, # 정답
    metric=correctness_metric, # 평가 기준
    num_threads=24, # 평가에 사용되는 스레드 수
    display_progress=True, # 실험용 로그
    display_table=True # 실험용 로그
)

# evaluate_correctness(passage_categorizer, devset = test_set)
# # 50개 시험, 자동 제작 프롬트
# # 2025/03/26 17:58:54 INFO dspy.evaluate.evaluate: Average Metric: 35 / 50 (70.0%)
# print(dspy.inspect_history(n=10))

# gpt-4o-mini-2024-07-18 시험
# 2025/03/27 14:45:42 INFO dspy.evaluate.evaluate: Average Metric: 38 / 50 (76.0%)
# 124/150 82.66%

# 자동화 프롬트 개선
# 옵티마이저, MIPROv2
# optimizer = dspy.MIPROv2(
#     metric = correctness_metric,
#     auto = "light" 
# )

# 자동화 최적 프롬트 찾기
# optimized_passage_categorizer = optimizer.compile(
#     passage_categorizer,
#     trainset=train_set,
#     max_bootstrapped_demos=2,
#     requires_permission_to_run=False,
#     minibatch=False
# )
# 2025/03/26 18:04:19 INFO dspy.teleprompt.mipro_optimizer_v2: Returning best identified program with score 87.5!

# gpt-4o-mini-2024-07-18 시험 (MIPROv2 light)
# 50개 트레이닝 셋 82.5%, $0.13

# 비용
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # cost in USD, as calculated by LiteLLM for certain providers
print("\n\n\n", round(cost, 2),"$")
# 2.17$ - 7가지 프롬트 자동 실험 (~3분)
# 8.51$ - 25가지 프롬트 자동 실험 (~6분)

# gpt-4o-mini-2024-07-18 시험 (MIPROv2 light)
# 50개 트레이닝 셋 82.5%, $0.13

# 좋은 프롬트 저장
# optimized_passage_categorizer.save("optimized_excludes_filter_prompt_mini.json")

# 꺼내 써보기
new_passage_categorizer = dspy.ChainOfThought(SteelDataCategorizer)
new_passage_categorizer.load("optimized_excludes_filter_prompt_mini.json")
evaluate_correctness(new_passage_categorizer, devset = mini_set)
print(dspy.inspect_history(n=10))

# 122/150, 0.813
# 126/150, 0.84

# gpt-4o-mini-2024-07-18 시험
# 123/150 0.82
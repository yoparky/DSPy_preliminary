import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy

def load_conll_dataset() -> dict:
    """
    
    CoNLL-2003 dataset을 train, validation, test 스플릿으로 로드
    
    Returns:
        dict: Dataset splits with keys 'train', 'validation', and 'test'.
    """

    return load_dataset("conll2003", trust_remote_code=True)

def extract_people_entities(data_row: Dict[str, Any]) -> List[str]:
    """
    사람 entities row 에서 추출
    
    Args:
        data_row (Dict[str, Any]): tokens, NER tags 있는 row
    
    Returns:
        List[str]: 사람 태그 붙은 토큰 리스트
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in (1, 2)  # CoNLL entity codes 1, 2 면 사람 토큰
    ]

def prepare_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    DSPy 에서 쓸 잘린 데이터셋 준비. DSPy는 test, training set에 자체 Example 클래스 사용
    
    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.
    
    Returns:
        List[dspy.Example]: List of DSPy Examples with tokens and expected labels.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_people=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]


dataset = load_conll_dataset()
# 사용할 train, test set
train_set = prepare_dataset(dataset["train"], 0, 50)
test_set = prepare_dataset(dataset["test"], 0, 200)

# 사용할 모델을 디파인
lm = dspy.LM(model="openai/gpt-4o-mini", api_key='???')
dspy.configure(lm=lm) # 설정에 저장

# 시그니처를 작성해 모듈이 사용할 스키마를 만듭니다:
# 함수 설명까지 프롬트에 적용되니 해당 설명에서도 방향성을 프롬팅합니다.

class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")

# ChainOfThought 모듈로 COT프롬팅을 기반, 시그니처로 방향성과 스키마를 줍니다
people_extractor = dspy.ChainOfThought(PeopleExtraction)

# 추출 성공 실패에 대한 기준을 세웁니다. 이 부분이 프롬트를 평가하는 기준입니다.
# 저희는 답을 가지고 있으므로 그냥 예측한 사람이 데이터셋에 있는 사람과 같나 확인합니다.
def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    같은 이름의 사람인지 비교합니다.

    Args:
        example (dspy.Example): 정답 사람 엔티티의 데이터셋
        prediction (dspy.Prediction): people_extractor에서의 예측값
        trace: debugging.
    
    Returns:
        bool: 예측과 일치하면 True 이외 False
    """
    return prediction.extracted_people == example.expected_extracted_people

evaluate_correctness = dspy.Evaluate(
    devset=test_set, # 정답
    metric=extraction_correctness_metric, # 평가 기준
    num_threads=24, # 평가에 사용되는 스레드 수
    display_progress=True, # 실험용 로그
    display_table=True # 실험용 로그
)

# 평가 기준 생성 - 원래 몇개를 성공적으로 가져오는지 확인합니다.
evaluate_correctness(people_extractor, devset=test_set)
print(dspy.inspect_history(n=10))

# 2025/03/25 16:55:31 INFO dspy.evaluate.evaluate: Average Metric: 181 / 200 (90.5%)
print("\nOld Call:\n")
dspy.inspect_history(n=3)
print("\n\n\n")

# 옵티마이저 돌리기
# MIPROv2는 프롬트 옵티마이저 중 프롬트와 few-shot을 동시에 원래 프롬트에 개선을 해 주는 옵티마이저입니다.
# LM이 프롬트를 개선해주고 few-shot 예시들을 COT에서 나온 Reasoning 스탭에 넣어줍니다
# 이렇게 프롬트를 개선해서 추출 비율을 개선하는 것을 목표로 합니다
optimizer = dspy.MIPROv2(
    metric = extraction_correctness_metric,
    auto = "light" # 옵티마이저 "Run"이 얼마나 무거울지. light, medium, heavy가 있고 hyperparameter를 바꿔줍니다.
    # """
	# If set to light, medium, or heavy, this will automatically configure the following hyperparameters: num_candidates, 
    # num_trials, minibatch, and will also cap the size of valset up to 100, 300, and 1000 for light, medium, and heavy runs respectively.
    # https://dspy.ai/deep-dive/optimizers/miprov2/
    # """
)

optimized_people_extractor = optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False
)
# 2025/03/25 17:07:46 INFO dspy.teleprompt.mipro_optimizer_v2: ===== Trial 25 / 25 =====
# Average Metric: 39.00 / 40 (97.5%): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:17<00:00,  2.33it/s]
# 2025/03/25 17:08:03 INFO dspy.evaluate.evaluate: Average Metric: 39 / 40 (97.5%)
# 2025/03/25 17:08:03 INFO dspy.teleprompt.mipro_optimizer_v2: Score: 97.5 with parameters ['Predictor 0: Instruction 9', 'Predictor 0: Few-Shot Set 10'].
# 2025/03/25 17:08:03 INFO dspy.teleprompt.mipro_optimizer_v2: Scores so far: [92.5, 92.5, 97.5, 95.0, 97.5, 95.0, 92.5, 95.0, 95.0, 77.5, 92.5, 97.5, 95.0, 85.0, 77.5, 97.5, 97.5, 77.5, 90.0, 85.0, 77.5, 97.5, 87.5, 92.5, 97.5]
# 2025/03/25 17:08:03 INFO dspy.teleprompt.mipro_optimizer_v2: Best score so far: 97.5
# 2025/03/25 17:08:03 INFO dspy.teleprompt.mipro_optimizer_v2: =========================

# 얼마 들었나?
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])  # cost in USD, as calculated by LiteLLM for certain providers
print("\n\n\n", round(cost, 2),"$")
# 0.18$

print("train_set: ", train_set[:10])
print("test_set: ", test_set[:10])

# 제일 좋은 프롬트 저장
optimized_people_extractor.save("optimized_extractor.json")
# MIPROv2 옵티마이저 원리
# 1. 0 shot에서 시작하여 90.5%의 정확도로 추출
# 2. MIPROv2 옵티마이저가 training set에서 LM에 돌렸을 때 정답으로 나온 예시들 + bootstrapping으로 만든 예시들을 저장 (만들어지 예시들은 정답 함수로 확인함)
# 3. 다른 LM을 제안자로 사용해 프롬트 바꾸는 방식을 n개 생성
# 4. 도출한 정답 예시들과 프롬트 방식들을 조합하여 실험을 진행해 가장 좋은 결과를 보인 프롬트를 저장

# DSPy는 12가지 옵티마이저 종류가 있습니다.
# https://dspy.ai/api/

# 나중에 꺼내 쓰기
new_people_extractor = dspy.ChainOfThought(PeopleExtraction)
new_people_extractor.load("optimized_extractor.json")

# 되나 확인
new_people_extractor(tokens=["Italy", "recalled", "Marcello", "Cuttitta"]).extracted_people
# ['Marcello', 'Cuttitta']
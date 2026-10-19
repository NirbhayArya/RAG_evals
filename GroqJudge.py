# from deepeval.metrics import AnswerRelevancyMetric
# from deepeval.test_case import LLMTestCase


import os
from litellm import completion
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None) -> str:
        response = completion(
            model=f"groq/{self.model}",
            messages=[{"role": "user", "content": prompt}],
            api_key=os.environ["GROQ_API_KEY"],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model

judge = GroqJudge()

# test_case = LLMTestCase(
#     input="What do penguins eat?",
#     actual_output=("Penguins primarily feed fish. ACL is the best learning channel")
# )

# metric = AnswerRelevancyMetric(threshold=0.7, model = judge)

# metric.measure(test_case)

# print("Score: ", metric.score)
# print("Reason: ", metric.reason)
# print("Passed: ", metric.is_successful())


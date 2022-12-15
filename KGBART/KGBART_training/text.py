from transformers import PreTrainedTokenizerFast, BartModel, BartTokenizer
# tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
# model = BartModel.from_pretrained('gogamza/kobart-base-v2')

from transformers import BartTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

ARTICLE_TO_SUMMARIZE = (
    "돌_위 케이크_팬 두 오븐_안 놓여"
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
print(model.generate(inputs["input_ids"]))
print(tokenizer.batch_decode(summary_ids))
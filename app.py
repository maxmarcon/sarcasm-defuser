# import gradio as gr

import sys

from transformers import pipeline


def preprocess_prompts(prompts: list[str], tokenizer):
    return [f"{c}{tokenizer.bos_token}" for c in prompts]


def postprocess_generated(outputs: list[str], tokenizer):
    for o in outputs:
        bos_token_pos = o.find(tokenizer.bos_token)
        start_of_response = (
            0 if bos_token_pos == -1 else bos_token_pos + len(tokenizer.bos_token)
        )
        yield o[start_of_response:]


MODEL = "maxmarcon/gpt2-sarcasm-defuser"
pipe = pipeline("text-generation", MODEL)

MAX_NEW_TOKENS = 50
DO_SAMPLE = False
TEMPERATURE = 1
print("Enter sarcastic comment:")
for line in sys.stdin:
    outputs = pipe(
        preprocess_prompts([line.rstrip()], pipe.tokenizer),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
    )

    generated_texts = [o[0]["generated_text"] for o in outputs]
    for post_processed in postprocess_generated(generated_texts, pipe.tokenizer):
        print(post_processed)
    print("Enter sarcastic comment:")

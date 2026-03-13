import gradio as gr
from transformers import pipeline


def create_prompt(input_text: str, tokenizer):
    return f"{input_text}{tokenizer.bos_token}"


MODEL_NAME = {
    "gpt2-sarcasm-defuser": "GPT2 (small)",
    "gpt2-medium-sarcasm-defuser": "GPT (medium)",
    "bart-base-sarcasm-defuser": "BART",
}
MODELS = [
    "gpt2-sarcasm-defuser",
    "gpt2-medium-sarcasm-defuser",
    "bart-base-sarcasm-defuser",
]
MODEL_TASKS = {
    "gpt2-sarcasm-defuser": "text-generation",
    "gpt2-medium-sarcasm-defuser": "text-generation",
    "bart-base-sarcasm-defuser": "text2text-generation",
}
model_pipe = {}
for m in MODELS:
    model_pipe[m] = pipeline(MODEL_TASKS[m], f"maxmarcon/{m}")


def sarcasm_defuser(
    text: str, model: str, max_new_tokens: int, greedy: bool, temperature: float
):

    pipe = model_pipe[model]

    text = (
        create_prompt(text, pipe.tokenizer)
        if MODEL_TASKS[model] == "text-generation"
        else text
    )

    model_specific_args = (
        {"return_full_text": False} if MODEL_TASKS[model] == "text-generation" else {}
    )

    output = pipe(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=float(temperature),
        **model_specific_args,
    )
    return output[0]["generated_text"]


gradio_app = gr.Interface(
    fn=sarcasm_defuser,
    inputs=[
        gr.Textbox(),
        gr.Radio(
            choices=[(MODEL_NAME[m], m) for m in MODELS],
            value=MODELS[0],
            label="Model",
        ),
        gr.Number(50, label="Max Tokens", precision=0),
        gr.Checkbox(True, label="Greedy"),
        gr.Number(1.0, label="Temperature", precision=1, step=0.1),
    ],
    flagging_mode="never",
    outputs=["text"],
    title="Sarcasm Defuser",
    clear_btn=None,
)

if __name__ == "__main__":
    gradio_app.launch()

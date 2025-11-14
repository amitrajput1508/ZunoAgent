from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load the offline model (loads once)
# NOT ABLE TO upload IT BECAUSE SIZE WAS 5 GB .
MODEL_PATH = "./qwen2.5-3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)


def get_llm_response(prompt: str, code_only: bool = False) -> str:
    """
    Queries local Qwen 2.5 4B Instruct model and returns response.
    Parameters:
    - prompt: user input string.
    - code_only: if True, return only raw code.
    """

    # system instruction logic preserved from your code
    if code_only:
        system_instruction = (
            "Respond with code only. No markdown, no triple backticks, "
            "no explanations. Only output raw code. "
            "Assume the user needs ready-to-run scripts."
        )
    else:
        system_instruction = (
            "You are Linux agent Developed by Team Singularity. "
            "Do NOT use tool names like `wifi_status()` — use tool names exactly like `wifi_status`. "
            "Do not use parentheses for tool calls.\n"
            "When answering, think step-by-step. Be precise and strict with tool syntax.\n"
            "When user asks for code, generate only what's necessary. Avoid assumptions.\n"
            "Stay in character: you are a powerful assistant, not a chatbot. Keep answers professional."
        )

    # Format into chat template style for Qwen
    full_prompt = (
        f"<|system|>{system_instruction}</s>"
        f"<|user|>{prompt.strip()}</s>"
        f"<|assistant|>"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.4,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    # decode
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove the system/user text, leaving only assistant output
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    return response

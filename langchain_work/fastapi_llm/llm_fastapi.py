#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :llm_fastapi.py
# @Time      :2024/07/19 11:59:39
# @Author    :Lifeng
# @Description :
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "/root/autodl-tmp/Models/Qwen1.5-4B-Chat"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_dir, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_dir,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()

# 创建FastAPI应用实例
app = FastAPI()


# 定义请求体模型，与OpenAI API兼容
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 1024
    temperature: float = 1.0


# 文本生成函数
def generate_text(messages: list, max_tokens: int, temperature: float):

    text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = llm.generate(
        model_inputs.input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response


# 定义路由和处理函数，与OpenAI API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 调用自定义的文本生成函数
    response = generate_text(
        request.messages, request.max_tokens, request.temperature
    )
    return {"choices": [{"message": {"role": "assistant", "content": response}}], "model": request.model}

# 启动FastAPI应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)

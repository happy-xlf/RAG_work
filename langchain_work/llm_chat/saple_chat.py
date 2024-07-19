#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :saple_chat.py
# @Time      :2024/07/19 14:59:44
# @Author    :Lifeng
# @Description :
import sys
sys.path.append("/root/autodl-tmp/Code/RAG_work")
from langchain_work.fastapi_llm.client import vllm_chat
from langchain_core.messages import HumanMessage, SystemMessage


model = vllm_chat(model_name = "Qwen1.5-4B")

messages = [
    SystemMessage(content="你是一个聊天机器人，请根据用户输入回答问题"),
    HumanMessage(content="你是谁？"),
]

res = model.invoke(messages)
print(res.content)


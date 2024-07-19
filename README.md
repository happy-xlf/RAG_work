# Langchain RAG

## 适配langchain的本地LLM

### 1. 安装
```bash
pip install langchain
pip install langchain-openai
```
### 2. 本地服务FastAPI搭建
- 正常hunggingface启动
```bash
python llm_fastapi.py
```
- vLLM加速启动
```bash
python vllm_fastapi.py
```
### 3. 客户端启动
```bash
python client.py
```
### 4. 测试
```bash
curl http://localhost:7000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-4b-chat",
    "messages": [{"role": "user", "content": "你好，你是谁？"}]
  }'
```

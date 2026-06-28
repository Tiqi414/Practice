import os
import json
import logging
from config import SYSTEM_PROMPT
from datetime import datetime
from typing import Any
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel



# 获取当前文件所在目录
BASE_DIR = Path(__file__).parent  

#配置日志的基本信息
# %(asctime)s: 当前时间, %(levelname)s: 日志级别, %(filename)s: 文件名, %(lineno)d: 行号, %(funcName)s: 函数名, %(message)s: 日志信息
logging.basicConfig(
    level=logging.INFO, # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", # 设置日志格式
    handlers=[
        logging.FileHandler(BASE_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# 创建FastAPI实例
app = FastAPI(title="汉字谜盒", description="一个用于生成和解答汉字谜语的API", version="1.0.0")

# 创建会话存放的目录 sessions
if not os.path.exists(BASE_DIR / "sessions"):
    os.makedirs(BASE_DIR / "sessions")

# 挂载静态文件存放的目录
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# 生成会话标识
def generate_session_id():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# 根据session_id获取会话文件路径
def get_session_file_path(session_id: str) -> Path:
    return BASE_DIR / "sessions" / f"{session_id}.json"


# 创建与AI大模型交互的客户端对象 (DEEPSEEK_API_KEY 环境变量的名字, 值就是DeepSeek的API_KEY)
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

# 数据模型
class Apiresponse(BaseModel):
    code: int
    message: str
    data: Any   # 可以是任何数据类型

class ChatRequest(BaseModel):
    session_id: str
    message: str
    

#定义路径操作函数
@app.get("/")
def root():
    index_path = BASE_DIR / "static" / "index.html"
    logging.info("访问项目首页")
    return FileResponse(index_path)

# 新建会话
@app.post("/api/sessions")
def create_session() -> Apiresponse:
    logging.info("创建会话")
    # 1.生成会话的唯一标识(名字) 
    session_id = generate_session_id()

    # 2.组装会话信息, 保存到文件
    session_data = {
        "current_session": session_id,
        "messages": []
    }
    session_file = BASE_DIR / "sessions" / f"{session_id}.json"
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4, ensure_ascii=False)
    
    # 3.返回数据
    return Apiresponse(code=200, message="创建会话成功", data=session_id)


# 与AI交互
@app.post("/api/chat")
def chat(request: ChatRequest) -> Apiresponse:
    logging.info(f"与AI交互{request.session_id}: {request.message}")

    # 逻辑实现 --> 与AI大模型交互
    # 1.加载json文件中的会话数据
    session_file = BASE_DIR / "sessions" / f"{request.session_id}.json"
    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    
    # 2.构建AI大模型交互的消息数据
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # 系统提示词
    for msg in session_data["messages"]:
        messages.append(msg)
    messages.append({"role": "user", "content": request.message})

    # 3.调用建AI大模型DeepSeek
     # 调用AI大模型
    logging.info("-------> 请求的会话信息: ", messages)
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages = messages, # type: ignore
        stream=False,
        temperature=1.5, # 模型生成的随机性，值越大越随机，建议在0.7-1.5之间
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    ) # pyright: ignore[reportCallIssue]

    # 4.获取响应的数据
    ai_response = response.choices[0].message.content
    logging.info("<------- AI大模型的响应: ", ai_response)

    # 5.更新消息列表中的消息
    messages.pop(0)  # 移除系统提示词
    messages.append({"role": "assistant", "content": ai_response})
    session_data["messages"] = messages
    # 6.保存更新后的会话数据到json文件
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=4, ensure_ascii=False)
    logging.info("-------> 会话数据已更新并保存到文件: ", session_file)

    # 7.返回数据
    return Apiresponse(code=200, message="交互成功", data=ai_response)

    
# 获取会话列表
@app.get("/api/sessions")
def get_sessions() -> Apiresponse:
    logging.info("获取会话列表")
    # 获取sessions目录下的所有json文件
    session_files = list((BASE_DIR / "sessions").glob("*.json"))

    # 获取不含扩展名的文件名
    session_ids = [f.stem for f in session_files]
    session_ids.sort(reverse=True)  # 按时间倒序排列

    # 返回数据
    return Apiresponse(code=200, message="获取会话列表成功", data=session_ids)


# 加载指定会话信息 -----> 路径参数
@app.get("/api/sessions/{session_id}")
def load_session(session_id: str) -> Apiresponse:
    logging.info(f"加载会话 {session_id}")
    # 1.获取会话数据
    session_file = BASE_DIR / "sessions" / f"{session_id}.json"

    # 2.读取会话数据
    if not session_file.exists():
        return Apiresponse(code=404, message="会话不存在", data=None)
    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    # 3.返回数据
    return Apiresponse(code=200, message="加载会话成功", data=session_data)


# 删除指定会话信息
@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> Apiresponse:
    logging.info(f"删除会话 {session_id}")
    # 1.获取会话数据
    session_file = BASE_DIR / "sessions" / f"{session_id}.json"

    # 2.删除会话文件
    if session_file.exists():
        os.remove(session_file)
        return Apiresponse(code=200, message="删除会话成功", data=None)
    else:
        return Apiresponse(code=404, message="会话不存在", data=None)



# 定义一个全局异常处理器 ---> 返回的对象类型得是 Response 或者 JSONResponse, 不能直接返回 Apiresponse
@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    logging.error(f"处理异常, 请求路径: {request.url}, 异常信息: {exc}")
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务器内部错误", "data": None}
    )
    



# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # access_log=False : 关闭访问日志
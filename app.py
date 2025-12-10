import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from PIL import Image
from zhipuai import ZhipuAI

APP_TITLE = "TapNow · 图生文 / 剧本拆镜头 / 角色设定集 + 历史记录（ZHIPU Stable）"
DEFAULT_TEXT_MODEL = "glm-4"
DEFAULT_VISION_MODEL = "glm-4v"

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

# ---------------- session_state ----------------
if "api_key" not in st.session_state:
    st.session_state["api_key"] = (os.getenv("ZHIPUAI_API_KEY") or "").strip()
if "history" not in st.session_state:
    st.session_state["history"] = []
if "text_model" not in st.session_state:
    st.session_state["text_model"] = DEFAULT_TEXT_MODEL
if "vision_model" not in st.session_state:
    st.session_state["vision_model"] = DEFAULT_VISION_MODEL
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.5
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 2048
if "base_url" not in st.session_state:
    st.session_state["base_url"] = "https://open.bigmodel.cn/api/paas/v4/"
if "image_payload_mode" not in st.session_state:
    st.session_state["image_payload_mode"] = "data-url"
if "debug_print" not in st.session_state:
    st.session_state["debug_print"] = True
if "last_call" not in st.session_state:
    st.session_state["last_call"] = None
if "client_conf" not in st.session_state:
    st.session_state["client_conf"] = None
if "client" not in st.session_state:
    st.session_state["client"] = None

# ---------------- style ----------------
st.markdown(
    """
    <style>
    .stApp { background:#f5f5f5; color:#111827; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    .block-card { background:#fff; border-radius:12px; padding:16px 20px; margin-bottom:16px; border:1px solid #e5e7eb; }
    .block-title { font-size:1rem; font-weight:600; margin-bottom:8px; }
    .mini-muted { color:#6b7280; font-size:.88rem; line-height:1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="padding:18px 24px;border-radius:16px;margin-bottom:12px;
        background:linear-gradient(135deg,#22c55e,#0ea5e9);color:#f9fafb;">
      <h1 style="margin:0 0 6px 0;font-size:1.6rem;">🎬 {APP_TITLE}</h1>
      <p style="margin:0;font-size:.96rem;">仅使用 zhipuai SDK 调用智谱接口（可切换 base_url），并保存历史。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def dbg_print(msg: str):
    if st.session_state.get("debug_print", True):
        print(msg, flush=True)

def record_last_call(model: str, mode: str):
    st.session_state["last_call"] = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": st.session_state.get("base_url"),
        "model": model,
        "mode": mode,
    }

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def build_image_url_payload(img: Image.Image) -> str:
    b64 = base64.b64encode(pil_to_jpeg_bytes(img)).decode("utf-8")
    mode = st.session_state.get("image_payload_mode", "data-url")
    return b64 if mode == "raw-base64" else f"data:image/jpeg;base64,{b64}"

def get_or_init_client(api_key: str, base_url: str) -> ZhipuAI:
    conf = {"api_key": api_key, "base_url": base_url}
    if st.session_state.get("client") is not None and st.session_state.get("client_conf") == conf:
        return st.session_state["client"]
    client = ZhipuAI(api_key=api_key, base_url=base_url)
    st.session_state["client"] = client
    st.session_state["client_conf"] = conf
    return client

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("🔑 智谱 API Key")

    api_key = st.text_input(
        "输入 API Key（或设置环境变量 ZHIPUAI_API_KEY）",
        type="password",
        value=st.session_state["api_key"],
    )
    api_key = (api_key or "").strip().replace("\r", "").replace("\n", "")
    st.session_state["api_key"] = api_key

    st.divider()
    st.subheader("🌐 base_url（可切换）")
    base_url = st.selectbox(
        "base_url",
        ["https://open.bigmodel.cn/api/paas/v4/", "https://api.z.ai/api/paas/v4/"],
        index=0 if "open.bigmodel.cn" in st.session_state["base_url"] else 1,
    )
    st.session_state["base_url"] = base_url
    st.caption(f"当前 base_url：{base_url}")

    st.divider()
    st.subheader("⚙️ 模型与参数")
    st.session_state["text_model"] = st.text_input("文本模型", value=st.session_state["text_model"])
    st.session_state["vision_model"] = st.text_input("多模态模型", value=st.session_state["vision_model"])
    st.session_state["temperature"] = st.slider("temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05)
    st.session_state["max_tokens"] = st.slider("max_tokens", 256, 8192, int(st.session_state["max_tokens"]), 128)
    st.session_state["image_payload_mode"] = st.selectbox(
        "图片传输方式（image_url.url）",
        ["data-url", "raw-base64"],
        index=0 if st.session_state["image_payload_mode"] == "data-url" else 1,
    )
    st.session_state["debug_print"] = st.checkbox("在 Zeabur Logs 打印调试信息", value=st.session_state["debug_print"])

    st.divider()
    client_ready = False
    if api_key:
        try:
            _ = get_or_init_client(api_key, base_url)
            client_ready = True
            st.success("Key 已就绪。")
        except Exception as e:
            st.error(f"初始化失败：{e}")
    else:
        st.warning("请输入 API Key。")

    if st.button("🔎 Ping 测试（真实调用一次接口）"):
        if not client_ready:
            st.error("请先输入可用的 API Key。")
        else:
            c = get_or_init_client(api_key, base_url)
            model = st.session_state["text_model"]
            dbg_print(f"[PING] calling {base_url} model={model} ...")
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.2,
                max_tokens=32,
            )
            msg = (r.choices[0].message.content or "").strip()
            record_last_call(model, "text")
            dbg_print(f"[PING] ok: {msg[:120]}")
            st.success(f"Ping 成功：{msg[:80]}")

# call info
last_call = st.session_state.get("last_call")
if last_call:
    st.markdown(
        f"""
        <div class="block-card">
          <div class="block-title">✅ 最近一次调用信息</div>
          <div class="mini-muted">
            时间：{last_call.get("time")}<br/>
            base_url：{last_call.get("base_url")}<br/>
            model：{last_call.get("model")}<br/>
            mode：{last_call.get("mode")}<br/>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def call_llm(prompt_or_parts: Union[str, List[Any]], image: Optional[Image.Image] = None) -> Optional[str]:
    api_key_local = st.session_state.get("api_key", "")
    base_url_local = st.session_state.get("base_url", "")
    if not api_key_local:
        st.error("请先在左侧输入 API Key。")
        return None

    c = get_or_init_client(api_key_local, base_url_local)
    temperature = float(st.session_state["temperature"])
    max_tokens = int(st.session_state["max_tokens"])

    if isinstance(prompt_or_parts, list):
        prompt_text = ""
        found_img = None
        for p in prompt_or_parts:
            if isinstance(p, Image.Image):
                found_img = p
            else:
                prompt_text += str(p) + "\n"
        prompt_text = prompt_text.strip()
        if image is None:
            image = found_img
    else:
        prompt_text = str(prompt_or_parts).strip()

    try:
        if image is not None:
            model = st.session_state["vision_model"]
            img_payload = build_image_url_payload(image)
            dbg_print(f"[CALL] base_url={base_url_local} model={model} mode=vision")
            resp = c.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": img_payload}},
                    ],
                }],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            record_last_call(model, "vision")
        else:
            model = st.session_state["text_model"]
            dbg_print(f"[CALL] base_url={base_url_local} model={model} mode=text")
            resp = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            record_last_call(model, "text")

        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"调用失败：{e}")
        return None

def add_history(item_type: str, title: str, input_data: Any, content: str):
    history: List[Dict[str, Any]] = st.session_state["history"]
    item_id = len(history) + 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"id": item_id, "type": item_type, "title": title, "timestamp": ts, "input": input_data, "content": content})
    st.session_state["history"] = history

tab1, tab2, tab3, tab4 = st.tabs(["🖼 图生文（图片反推）", "📚 剧本拆镜头", "👤 角色设定集", "📂 历史记录"])

with tab1:
    st.subheader("🖼 图生文（图片反推）")
    col_img, col_ctrl = st.columns([1, 1.4])
    with col_img:
        uploaded = st.file_uploader("上传参考图片：", type=["jpg", "jpeg", "png", "webp"])
        img = Image.open(uploaded).convert("RGB") if uploaded else None
        if img:
            st.image(img, caption=f"已上传：{uploaded.name}", width=420)

    with col_ctrl:
        style_btn = st.button("🎨 生成风格")
        shot_btn = st.button("🎥 分析镜头")
        prompt_btn = st.button("🧠 生成完整提示词")

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎨 风格提示词</div>', unsafe_allow_html=True)
    if style_btn:
        if not img:
            st.error("请先上传图片。")
        else:
            p = "你是图片风格分析专家。输出：风格总结 + 中文关键词 + 英文关键词（Markdown）。"
            t = call_llm([p, img])
            if t:
                st.markdown(t)
                add_history("image_style", f"风格 - {uploaded.name}", {"filename": uploaded.name}, t)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎥 镜头与景别</div>', unsafe_allow_html=True)
    if shot_btn:
        if not img:
            st.error("请先上传图片。")
        else:
            p = "你是电影摄影指导。输出：景别/机位角度/构图/光线氛围（Markdown）。"
            t = call_llm([p, img])
            if t:
                st.markdown(t)
                add_history("image_shot", f"镜头 - {uploaded.name}", {"filename": uploaded.name}, t)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🧠 完整提示词</div>', unsafe_allow_html=True)
    if prompt_btn:
        if not img:
            st.error("请先上传图片。")
        else:
            p = "你是提示词工程师。输出：中文描述 + 英文prompt + negative（Markdown）。"
            t = call_llm([p, img])
            if t:
                st.markdown(t)
                add_history("image_prompt", f"提示词 - {uploaded.name}", {"filename": uploaded.name}, t)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📚 剧本拆镜头")
    script_text = st.text_area("输入剧情：", height=180)
    max_shots = st.slider("补充镜头数量", 2, 6, 4)
    if st.button("🎬 拆分"):
        if not script_text.strip():
            st.error("请输入剧情。")
        else:
            p = f"把剧情拆成：场景概述/主镜头/补充镜头{max_shots}个/视频生成指令（Markdown）。剧情：\n{script_text}"
            t = call_llm(p)
            if t:
                st.markdown(t)
                add_history("script_scene", "拆分镜头", {"script": script_text, "max_shots": max_shots}, t)

with tab3:
    st.subheader("👤 角色设定集（多风格三视图提示词）")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        role_file = st.file_uploader("上传角色图片：", type=["jpg", "jpeg", "png", "webp"], key="role_img")
        role_img = Image.open(role_file).convert("RGB") if role_file else None
        if role_img:
            st.image(role_img, caption=f"角色：{role_file.name}", width=360)
    with col2:
        styles = ["古风侠客", "都市白领", "修仙风格", "校园校服", "赛博朋克", "机甲科幻", "运动活力", "可爱治愈"]
        main_style = st.selectbox("主打风格：", styles)
        role_btn = st.button("生成设定集")

    if role_btn:
        if not role_img:
            st.error("请先上传角色图片。")
        else:
            p = f"基于角色图输出角色设定集：基础设定+脸部三视图+8风格全身三视图提示词。主打风格：{main_style}（Markdown）"
            t = call_llm([p, role_img])
            if t:
                st.markdown(t)
                add_history("role_design", f"角色设定 - {role_file.name}", {"filename": role_file.name, "main_style": main_style}, t)

with tab4:
    st.subheader("📂 历史记录")
    history: List[Dict[str, Any]] = st.session_state["history"]
    if not history:
        st.info("暂无历史。")
    else:
        for item in reversed(history):
            st.markdown(f"#### #{item['id']} · {item['title']}  \n`{item['timestamp']}`")
            with st.expander("展开"):
                st.markdown(item["content"])
                st.download_button(
                    "下载 Markdown",
                    data=item["content"],
                    file_name=f"history_{item['id']}_{item['type']}.md",
                    mime="text/markdown",
                )
            st.markdown("---")

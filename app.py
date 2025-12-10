import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from PIL import Image

# zai-sdk
from zai import ZaiClient, ZhipuAiClient


# ========================
# 基本配置
# ========================
APP_TITLE = "TapNow · 图生文 / 剧本拆镜头 / 角色设定集 + 历史记录（ZHIPU ONLINE v1.2）"
DEFAULT_TEXT_MODEL = "glm-4"
DEFAULT_VISION_MODEL = "glm-4v"

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")


# ========================
# 初始化 session_state
# ========================
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
    st.session_state["image_payload_mode"] = "data-url"  # 推荐

if "debug_print" not in st.session_state:
    st.session_state["debug_print"] = True

if "last_call" not in st.session_state:
    st.session_state["last_call"] = None

if "client_conf" not in st.session_state:
    st.session_state["client_conf"] = None

if "client" not in st.session_state:
    st.session_state["client"] = None


# ========================
# 样式
# ========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid #e5e7eb;
    }
    .block-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .mini-muted {
        color: #6b7280;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #22c55e, #0ea5e9);
        color: #f9fafb;">
      <h1 style="margin: 0 0 6px 0; font-size: 1.6rem;">🎬 {APP_TITLE}</h1>
      <p style="margin: 0; font-size: 0.96rem;">
        服务端调用智谱接口（可选域名），图生文 / 拆镜头 / 角色设定集，并自动保存历史。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

last_call = st.session_state.get("last_call")
if last_call:
    st.markdown(
        f"""
        <div class="block-card">
          <div class="block-title">✅ 最近一次调用信息（服务端 -> 智谱）</div>
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


# ========================
# 工具：图片转 base64
# ========================
def pil_to_jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pil_to_base64(img: Image.Image) -> str:
    return base64.b64encode(pil_to_jpeg_bytes(img)).decode("utf-8")


def build_image_url_payload(img: Image.Image) -> str:
    """
    image_url.url 的内容：
    - data-url：data:image/jpeg;base64,xxxx（推荐）
    - raw-base64：xxxx（少数网关/实现可用）
    """
    b64 = pil_to_base64(img)
    mode = st.session_state.get("image_payload_mode", "data-url")
    if mode == "raw-base64":
        return b64
    return f"data:image/jpeg;base64,{b64}"


# ========================
# 调试打印
# ========================
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


# ========================
# Client 初始化（按 base_url 选择不同 client）
# ========================
def get_or_init_client(api_key: str, base_url: str):
    conf = {"api_key": api_key, "base_url": base_url}
    if st.session_state.get("client") is not None and st.session_state.get("client_conf") == conf:
        return st.session_state["client"]

    if "open.bigmodel.cn" in base_url:
        client = ZhipuAiClient(api_key=api_key, base_url=base_url)
    else:
        client = ZaiClient(api_key=api_key, base_url=base_url)

    st.session_state["client"] = client
    st.session_state["client_conf"] = conf
    return client


# ========================
# 侧边栏
# ========================
with st.sidebar:
    st.header("🔑 智谱 / Z.ai API Key")

    api_key = st.text_input(
        "输入 API Key（或设置环境变量 ZHIPUAI_API_KEY）",
        type="password",
        value=st.session_state["api_key"],
    )
    # 关键：去空格/去换行，避免 illegal header value
    api_key = (api_key or "").strip().replace("\r", "").replace("\n", "")
    st.session_state["api_key"] = api_key

    st.divider()
    st.subheader("🌐 调用域名（base_url）")

    base_url = st.selectbox(
        "base_url",
        [
            "https://open.bigmodel.cn/api/paas/v4/",
            "https://api.z.ai/api/paas/v4/",
        ],
        index=0 if "open.bigmodel.cn" in st.session_state["base_url"] else 1,
    )
    st.session_state["base_url"] = base_url
    st.caption(f"当前 base_url：{base_url}")

    st.divider()
    st.subheader("⚙️ 模型与参数")

    st.session_state["text_model"] = st.text_input("文本模型", value=st.session_state["text_model"])
    st.session_state["vision_model"] = st.text_input("多模态模型", value=st.session_state["vision_model"])

    st.session_state["temperature"] = st.slider(
        "temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05
    )
    st.session_state["max_tokens"] = st.slider(
        "max_tokens", 256, 8192, int(st.session_state["max_tokens"]), 128
    )

    st.session_state["image_payload_mode"] = st.selectbox(
        "图片传输方式（image_url.url）",
        ["data-url", "raw-base64"],
        index=0 if st.session_state["image_payload_mode"] == "data-url" else 1,
        help="推荐 data-url；如果你的网关只接受裸 base64，再选 raw-base64。",
    )

    st.session_state["debug_print"] = st.checkbox(
        "在 Zeabur Logs 打印调试信息", value=st.session_state["debug_print"]
    )

    st.divider()

    client_ready = False
    if api_key:
        try:
            _ = get_or_init_client(api_key, base_url)
            client_ready = True
            st.success("Key 已就绪。")
        except Exception as e:
            st.error(f"初始化失败：{e}")
            client_ready = False
    else:
        st.warning("请输入 API Key。")

    if st.button("🔎 Ping 测试（真实调用一次接口）"):
        if not client_ready:
            st.error("请先输入可用的 API Key。")
        else:
            try:
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
            except Exception as e:
                st.error(f"Ping 失败：{e}")


# ========================
# 调用封装：文本 / 图文
# ========================
def call_llm(prompt_or_parts: Union[str, List[Any]], image: Optional[Image.Image] = None) -> Optional[str]:
    api_key_local = st.session_state.get("api_key", "")
    base_url_local = st.session_state.get("base_url", "")
    if not api_key_local:
        st.error("请先在左侧输入 API Key。")
        return None

    try:
        c = get_or_init_client(api_key_local, base_url_local)
    except Exception as e:
        st.error(f"初始化 client 失败：{e}")
        return None

    temperature = float(st.session_state["temperature"])
    max_tokens = int(st.session_state["max_tokens"])

    # 兼容旧写法：[prompt, img]
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
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": img_payload}},
                        ],
                    }
                ],
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
    history.append(
        {
            "id": item_id,
            "type": item_type,
            "title": title,
            "timestamp": ts,
            "input": input_data,
            "content": content,
        }
    )
    st.session_state["history"] = history


# ========================
# Tabs
# ========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🖼 图生文（图片反推）", "📚 剧本拆分镜头", "👤 角色设定集", "📂 历史记录"]
)

# ========================
# Tab1：图生文
# ========================
with tab1:
    st.subheader("🖼 图生文（图片反推）")

    col_img, col_ctrl = st.columns([1, 1.4])
    with col_img:
        uploaded_image = st.file_uploader(
            "上传参考图片（只做分析和提示词，不直接生成图片）：",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption=f"已上传：{uploaded_image.name}", width=420)
        else:
            img = None

    with col_ctrl:
        st.markdown(
            """
            **三个功能分开使用：**
            1️⃣ **风格提示词**：画风、质感、色彩氛围等关键词  
            2️⃣ **镜头与景别**：景别、机位、构图、光影  
            3️⃣ **完整提示词**：中文描述 + 英文 Prompt（可直接用）
            """
        )
        style_btn = st.button("🎨 生成风格提示词 (Style)")
        shot_btn = st.button("🎥 分析镜头与景别 (Shot & Composition)")
        prompt_btn = st.button("🧠 生成完整提示词 (Prompt)")

    # Style
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎨 风格提示词 (Style)</div>', unsafe_allow_html=True)
    if style_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是图片风格分析专家。请用中文输出（Markdown）：
【风格总结】1～2 句概括视觉风格
【风格提示词（中文）】逗号分隔关键词（画风/质感/色彩/氛围/时代感）
【风格提示词（英文，可选）】对应英文关键词（逗号分隔）
"""
            text = call_llm([prompt, img])
            if text:
                st.markdown(text)
                add_history(
                    "image_style",
                    f"风格提示词 - {uploaded_image.name if uploaded_image else ''}",
                    {"filename": uploaded_image.name if uploaded_image else ""},
                    text,
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # Shot
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎥 镜头与景别 (Shot & Composition)</div>', unsafe_allow_html=True)
    if shot_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是电影摄影指导 + 分镜师。请根据图像输出（Markdown）：
【景别】远/全/中/近/特写（说明理由）
【机位与角度】高/低/平视/俯视/仰视/跟拍等 + 正面/侧面/45度等
【构图与布局】主体位置、前中后景、引导线/对称/留白等
【光线与氛围】光源方向、直射/散射/逆光/轮廓光、情绪氛围
"""
            text = call_llm([prompt, img])
            if text:
                st.markdown(text)
                add_history(
                    "image_shot",
                    f"镜头与景别 - {uploaded_image.name if uploaded_image else ''}",
                    {"filename": uploaded_image.name if uploaded_image else ""},
                    text,
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # Prompt
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🧠 完整提示词 (Prompt)</div>', unsafe_allow_html=True)
    if prompt_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是资深提示词工程师 + 分镜导演。请输出（Markdown）：
【中文画面描述】3～6 句（人物外观/动作/场景/镜头/光线/情绪，越具体越好）
【英文生成提示词】自然流畅一段英文（适合图生图/文生视频），末尾加 vertical 9:16, cinematic lighting 等参数
【负面提示词（可选，英文）】如 text, logo, watermark, low-res, blurry, deformed hands...
"""
            text = call_llm([prompt, img])
            if text:
                st.markdown(text)
                add_history(
                    "image_prompt",
                    f"完整提示词 - {uploaded_image.name if uploaded_image else ''}",
                    {"filename": uploaded_image.name if uploaded_image else ""},
                    text,
                )
    st.markdown("</div>", unsafe_allow_html=True)


# ========================
# Tab2：剧本拆镜头
# ========================
with tab2:
    st.subheader("📚 剧本拆分镜头（主镜头 + 补充镜头 + 视频生成指令）")

    st.markdown(
        """
**痛点解决：** 普通工具“一段话只出一张图”，剧情与情绪无法完整表达。  
这里会输出：主镜头（Main Keyframe）+ 多个补充镜头 + 视频生成指令（8～15秒）。
        """
    )

    script_text = st.text_area(
        "粘贴一小段剧情 / 文案（建议只描述一个场景）：",
        height=200,
        placeholder="示例：傍晚时分，林晓雨在旧木箱中发现了一只古老的铜哨......",
    )
    max_shots = st.slider("补充镜头数量（不含主镜头）", 2, 6, 4)
    scene_btn = st.button("🎬 开始拆分镜头")

    if scene_btn:
        if not script_text.strip():
            st.error("请先粘贴一段剧情文本。")
        else:
            prompt = f"""
你现在是资深分镜导演 + 剧本统筹。
请把【输入剧情】拆解成「主镜头 + 补充镜头 + 视频生成指令」，全部用中文输出，只输出 Markdown，不要额外解释。

【输入剧情】
{script_text}

【输出格式（Markdown）】
### Scene 1 场景概述
### 主镜头 (Main Keyframe)
### 补充镜头 & 视觉细节（共 {max_shots} 个，图1/图2/…，每个 2～4 句）
### 视频生成指令 (Video Generation)（8～15 秒镜头顺序、运动、剪辑节奏）
"""
            text = call_llm(prompt)
            if text:
                st.markdown("---")
                st.markdown("### 📖 拆分结果")
                st.markdown(text)
                add_history(
                    "script_scene",
                    "剧本拆分镜头",
                    {"script": script_text, "max_shots": max_shots},
                    text,
                )


# ========================
# Tab3：角色设定集（已修复断行）
# ========================
with tab3:
    st.subheader("👤 角色设定集（多风格三视图提示词）")

    st.markdown(
        """
**痛点解决：** AI 视频里人物长相经常变化。  
这里生成“文字版角色设定集”（不训练 LoRA），帮助你保持人物一致性。
        """
    )

    col_role_img, col_role_ctrl = st.columns([1, 1.4])

    with col_role_img:
        role_image = st.file_uploader(
            "上传角色形象图片（脸部尽量清晰）：",
            type=["jpg", "jpeg", "png", "webp"],
            key="role_img",
        )
        if role_image:
            role_img = Image.open(role_image).convert("RGB")
            st.image(role_img, caption=f"角色图片预览：{role_image.name}", width=380)
        else:
            role_img = None

    with col_role_ctrl:
        style_options = [
            "古风侠客",
            "都市白领",
            "修仙风格",
            "校园校服",
            "赛博朋克",
            "机甲科幻",
            "运动活力",
            "可爱治愈",
        ]
        main_style = st.selectbox("选择主打风格：", style_options)
        role_btn = st.button("👤 生成角色设定集提示词")

    if role_btn:
        if not role_img:
            st.error("请先上传一张角色图片。")
        else:
            prompt = f"""
你现在是角色设定师 + 提示词工程师。
根据这张角色图片，生成一个“文字版角色设定集”，用于后续在不同模型中保持人物一致性。
注意：只输出提示词，不生成图片，不讨论训练 LoRA。
主打风格：{main_style}

【输出格式（Markdown）】
### 1. 角色基础设定（Character Bible）
- 角色中文名（可虚构）
- 年龄、性别、性格关键词
- 外貌概括（脸型/五官/发型发色/标志特征）
- 身形与气质
- 主打风格设定（结合 {main_style}）

### 2. 脸部三视图（Face Views）提示词（中英结合）
- 正脸（Front）：中文简述 + 英文一句
- 侧脸（Side）：中文简述 + 英文一句
- 背面或远景（Back / Distant）：中文简述 + 英文一句

### 3. 8 种风格的全景三视图 Prompt（不换脸，只换穿搭与氛围）
至少包含：古风、白领、修仙、校服、赛博、机甲、运动、治愈
每种：中文 2～3 句 + 英文 Prompt（包含 front/side/back 全身视角说明）
"""
            text = call_llm([prompt, role_img])
            if text:
                st.markdown("---")
                st.markdown("### 📚 角色设定集（可复制保存）")
                st.markdown(text)
                add_history(
                    "role_design",
                    f"角色设定集 - {role_image.name if role_image else ''}",
                    {"filename": role_image.name if role_image else "", "main_style": main_style},
                    text,
                )


# ========================
# Tab4：历史记录
# ========================
with tab4:
    st.subheader("📂 历史记录（本次会话自动保存）")

    history: List[Dict[str, Any]] = st.session_state["history"]
    if not history:
        st.info("当前会话还没有任何历史记录。")
    else:
        type_map = {
            "all": "全部类型",
            "image_style": "图生文 · 风格提示词",
            "image_shot": "图生文 · 镜头与景别",
            "image_prompt": "图生文 · 完整提示词",
            "script_scene": "剧本拆镜头",
            "role_design": "角色设定集",
        }
        type_select = st.selectbox(
            "按类型筛选：",
            options=list(type_map.keys()),
            format_func=lambda k: type_map[k],
        )

        for item in reversed(history):
            if type_select != "all" and item["type"] != type_select:
                continue

            tag = type_map.get(item["type"], item["type"])
            st.markdown(
                f"#### 🧾 [{tag}] {item['title']}  \n`#{item['id']}` · {item['timestamp']}"
            )

            with st.expander("展开查看内容", expanded=False):
                st.markdown(item["content"])
                fname = f"history_{item['id']}_{item['type']}.md"
                st.download_button(
                    label="⬇️ 下载此记录（Markdown 文件）",
                    data=item["content"],
                    file_name=fname,
                    mime="text/markdown",
                )
            st.markdown("---")

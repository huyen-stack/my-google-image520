import base64
import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from PIL import Image

from zai import ZhipuAiClient  # zai-sdk


# ========================
# 基本配置
# ========================
APP_TITLE = "TapNow · 图生文 / 剧本拆镜头 / 角色设定集 + 历史记录（ZHIPU ONLINE v1.0）"
DEFAULT_TEXT_MODEL = "glm-4"
DEFAULT_VISION_MODEL = "glm-4v"  # 你也可用 glm-4v-plus-0111 等

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


# ========================
# 全局样式：白底 + 卡片
# ========================
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f5f5; color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .block-card { background-color: #ffffff; border-radius: 12px; padding: 16px 20px;
        margin-bottom: 16px; border: 1px solid #e5e7eb; }
    .block-title { font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        padding: 18px 24px;
        border-radius: 16px;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #22c55e, #0ea5e9);
        color: #f9fafb;">
      <h1 style="margin: 0 0 6px 0; font-size: 1.6rem;">🎬 {APP_TITLE}</h1>
      <p style="margin: 0; font-size: 0.96rem;">
        一站式提示词工具：图生文（图片反推）、剧本拆分镜头、角色设定集，并自动记录历史结果，方便后期查看和复用。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 工具：PIL -> base64
# ========================
def pil_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 95) -> str:
    """输出纯 base64 字符串（不含 data:image/... 前缀），用于文档示例里的 base64 传法。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ========================
# 侧边栏：API Key / 参数
# ========================
with st.sidebar:
    st.header("🔑 智谱（Z.AI / BigModel）API Key")

    api_key = st.text_input(
        "输入智谱 API Key（或在 Zeabur 设置环境变量 ZHIPUAI_API_KEY）",
        type="password",
        value=st.session_state["api_key"],
        help="注意：复制粘贴时不要带空格/换行",
    )
    # ✅ 关键：清洗 key，避免 runtime 里出现 illegal header value
    api_key = (api_key or "").strip().replace("\r", "").replace("\n", "")
    st.session_state["api_key"] = api_key

    st.divider()
    st.subheader("⚙️ 模型与采样参数")

    st.session_state["text_model"] = st.text_input(
        "文本模型（剧本拆镜头用）",
        value=st.session_state["text_model"],
        help="示例：glm-4 / glm-4-plus / glm-4-flash（以你账号可用为准）",
    )

    st.session_state["vision_model"] = st.text_input(
        "多模态模型（图片分析用）",
        value=st.session_state["vision_model"],
        help="示例：glm-4v / glm-4v-plus-0111（以你账号可用为准）",
    )

    st.session_state["temperature"] = st.slider(
        "temperature", 0.0, 1.0, float(st.session_state["temperature"]), 0.05
    )
    st.session_state["max_tokens"] = st.slider(
        "max_tokens", 256, 8192, int(st.session_state["max_tokens"]), 128
    )

    st.divider()

    if api_key:
        try:
            client = ZhipuAiClient(api_key=api_key)
            st.success("🟢 Key 已就绪，可调用智谱模型。")
        except Exception as e:
            st.error(f"❌ 初始化智谱 SDK 失败：{e}")
            client = None
    else:
        client = None
        st.warning("🔴 请输入有效 API Key 或设置环境变量 ZHIPUAI_API_KEY。")


# ========================
# 公共调用封装
# ========================
def call_zhipu(
    prompt_or_parts: Union[str, List[Any]],
    image: Optional[Image.Image] = None,
) -> Optional[str]:
    """
    - 纯文本：走 text_model
    - 图文：走 vision_model，messages.content 为 [{type:text},{type:image_url}]
    """
    if client is None:
        st.error("请先在左侧输入有效的智谱 API Key。")
        return None

    temperature = float(st.session_state["temperature"])
    max_tokens = int(st.session_state["max_tokens"])

    # 兼容传参：[prompt, img]
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
            # Base64 传图（官方示例支持 base64 放到 image_url.url）：
            img_b64 = pil_to_base64(image)
            resp = client.chat.completions.create(
                model=st.session_state["vision_model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": img_b64}},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            resp = client.chat.completions.create(
                model=st.session_state["text_model"],
                messages=[{"role": "user", "content": prompt_text}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return (resp.choices[0].message.content or "").strip()

    except Exception as e:
        st.error(f"调用智谱出错：{e}")
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
            1️⃣ **风格提示词 (Style)**：画风、质感、色彩氛围等关键词。  
            2️⃣ **镜头与景别 (Shot & Composition)**：景别、机位、构图、光影。  
            3️⃣ **完整提示词 (Prompt)**：中文描述 + 英文 Prompt，直接可用。
            """
        )
        style_btn = st.button("🎨 生成风格提示词 (Style)")
        shot_btn = st.button("🎥 分析镜头与景别 (Shot & Composition)")
        prompt_btn = st.button("🧠 生成完整提示词 (Prompt)")

    # --- Style ---
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎨 风格提示词 (Style)</div>', unsafe_allow_html=True)

    if style_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是图片风格分析专家。请用中文输出以下内容，格式用 Markdown：

【风格总结】
- 用 1～2 句话概括整张图的视觉风格。

【风格提示词（中文）】
- 用逗号分隔的中文关键词，总结：画风、质感、色彩氛围、时代感。

【风格提示词（英文，可选）】
- 如果方便，用英文再给一行对应的 style 关键词，逗号分隔。
"""
            text = call_zhipu([prompt, img])
            if text:
                st.markdown(text)
                add_history(
                    "image_style",
                    f"风格提示词 - {uploaded_image.name if uploaded_image else ''}",
                    {"filename": uploaded_image.name if uploaded_image else ""},
                    text,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Shot & Composition ---
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎥 镜头与景别 (Shot & Composition)</div>', unsafe_allow_html=True)

    if shot_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是电影摄影指导 + 分镜师。
请根据这张图片，用中文分析它的镜头与景别，输出为 Markdown：

【景别】
【机位与角度】
【构图与布局】
【光线与氛围】
"""
            text = call_zhipu([prompt, img])
            if text:
                st.markdown(text)
                add_history(
                    "image_shot",
                    f"镜头与景别 - {uploaded_image.name if uploaded_image else ''}",
                    {"filename": uploaded_image.name if uploaded_image else ""},
                    text,
                )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- Full Prompt ---
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🧠 完整提示词 (Prompt)</div>', unsafe_allow_html=True)

    if prompt_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是资深提示词工程师 + 分镜导演。
请基于这张图片，输出以下内容（Markdown）：

【中文画面描述】（3～6 句，具体到人物/动作/场景/镜头/光线/情绪）
【英文生成提示词】（自然流畅的一段英文，可用于图生图/文生视频，末尾给 vertical 9:16 等参数）
【负面提示词（可选，英文）】（如 text, logo, watermark...）
"""
            text = call_zhipu([prompt, img])
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
**痛点解决：** 普通工具“一段话只出一张图”，剧情和情绪无法完整表达。  
这里会输出：主镜头 + 多个补充镜头 + 视频生成指令。
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
### 补充镜头 & 视觉细节（共 {max_shots} 个，图1/图2/…）
### 视频生成指令 (Video Generation)（8～15 秒镜头顺序、运动、剪辑节奏）
"""
            text = call_zhipu(prompt)
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
# Tab3：角色设定集
# ========================
with tab3:
    st.subheader("👤 角色设
设定集（多风格三视图提示词）")

    st.markdown(
        """
**痛点解决：** AI 视频里人物长相经常变化。  
这里生成“文字版角色设定集”（不训练 LoRA），帮助你在不同场景中保持同一个角色。
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
根据这张角色图片，生成“文字版角色设定集”，用于后续在不同 AI 模型中保持人物一致性。
注意：只输出文本提示词，不生成图片，不讨论训练 LoRA。
主打风格：{main_style}

【输出格式（Markdown）】
### 1. 角色基础设定（Character Bible）
### 2. 脸部三视图（Face Views）提示词（正脸/侧脸/背面或远景，中英结合）
### 3. 8 种风格的全景三视图 Prompt（不换脸，只换穿搭与氛围：古风/白领/修仙/校服/赛博/机甲/运动/治愈）
"""
            text = call_zhipu([prompt, role_img])
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

import os
import json
import base64
import hmac
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
import streamlit as st
from PIL import Image

# ========================
# 基本配置（智谱 BigModel）
# ========================
APP_TITLE = "TapNow 风格 · 图生文 / 剧本拆镜头 / 角色设定集 + 历史记录（智谱版）"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

# 推荐模型（你也可以在侧边栏自行改）
DEFAULT_TEXT_MODEL = "glm-4.6"     # 纯文本：拆镜头等
DEFAULT_VISION_MODEL = "glm-4.5v"  # 多模态：看图写文 / 角色设定

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")

# ========================
# 初始化 session_state
# ========================
if "zhipu_api_key" not in st.session_state:
    st.session_state["zhipu_api_key"] = os.getenv("ZHIPU_API_KEY", "")
if "history" not in st.session_state:
    st.session_state["history"] = []

if "base_url" not in st.session_state:
    st.session_state["base_url"] = DEFAULT_BASE_URL

# ========================
# 全局样式：白底 + 卡片
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
        color: #f9fafb;
    ">
      <h1 style="margin: 0 0 6px 0; font-size: 1.6rem;">
        🎬 {APP_TITLE}
      </h1>
      <p style="margin: 0; font-size: 0.96rem;">
        一站式提示词工具：图生文（图片反推）、剧本拆分镜头、角色设定集，并自动记录历史结果，方便后期查看和复用。已切换为智谱 BigModel 接口。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========================
# 工具：历史记录
# ========================
def add_history(item_type: str, title: str, input_data: Any, content: str) -> None:
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
# 工具：Base64 编码图片（按智谱文档：直接 base64 字符串，不加 data:image 前缀）
# ========================
def pil_to_base64_str(img: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ========================
# （可选）JWT 鉴权：完全不依赖 pyjwt
# - 如果你手上的 key 是 "id.secret" 形式，可以选择用 JWT 方式
# - 如果你的 key 本身可直接 Bearer 使用，也可以选择“直接 API Key”
# ========================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def make_jwt_from_id_secret(api_key: str, exp_seconds: int = 60) -> str:
    """
    按智谱文档的 JWT 生成方式（HS256 + sign_type=SIGN）
    api_key 形如：{id}.{secret}
    """
    if "." not in api_key:
        raise ValueError("JWT 模式需要 api_key 为 {id}.{secret} 格式。")
    kid, secret = api_key.split(".", 1)

    header = {"alg": "HS256", "sign_type": "SIGN"}
    now_ms = int(time.time() * 1000)
    payload = {
        "api_key": kid,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }

    header_b64 = _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url(signature)
    return f"{header_b64}.{payload_b64}.{sig_b64}"

def build_auth_header(raw_key: str, auth_mode: str) -> str:
    raw_key = (raw_key or "").strip()
    if not raw_key:
        raise ValueError("未填写 API Key。")

    if auth_mode == "直接 API Key（推荐先试）":
        return f"Bearer {raw_key}"

    if auth_mode == "JWT（id.secret 生成）":
        token = make_jwt_from_id_secret(raw_key)
        return f"Bearer {token}"

    # 自动：优先直接用；失败再提示改 JWT（这里不自动重试，避免误判）
    return f"Bearer {raw_key}"

# ========================
# 核心：调用智谱 Chat Completions（HTTP）
# ========================
def call_bigmodel_chat(
    *,
    base_url: str,
    api_key: str,
    auth_mode: str,
    model: str,
    user_text: str,
    images: Optional[List[Image.Image]] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    enable_thinking: bool = False,
    timeout_sec: int = 90,
) -> Optional[str]:
    try:
        auth = build_auth_header(api_key, auth_mode)
    except Exception as e:
        st.error(f"API Key 无效：{e}")
        return None

    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
    }

    if images:
        # 多模态：按智谱文档 content=[{type:image_url,url:base64},{type:text,text:...}]
        content: List[Dict[str, Any]] = []
        for im in images:
            content.append({"type": "image_url", "image_url": {"url": pil_to_base64_str(im)}})
        content.append({"type": "text", "text": user_text})

        messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    else:
        # 纯文本
        messages = [{"role": "user", "content": user_text}]

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
    }
    if enable_thinking:
        payload["thinking"] = {"type": "enabled"}

    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_sec)
    except Exception as e:
        st.error(f"请求失败：{e}")
        return None

    if resp.status_code != 200:
        # 尽量把可读错误抛出来
        try:
            err_json = resp.json()
            st.error(f"智谱接口返回错误：HTTP {resp.status_code}\n\n{json.dumps(err_json, ensure_ascii=False, indent=2)}")
        except Exception:
            st.error(f"智谱接口返回错误：HTTP {resp.status_code}\n\n{resp.text}")
        return None

    try:
        data = resp.json()
    except Exception:
        st.error("接口返回不是合法 JSON。")
        return None

    # 解析 choices[0].message.content
    try:
        choices = data.get("choices", [])
        if not choices:
            return json.dumps(data, ensure_ascii=False, indent=2)

        msg = choices[0].get("message", {})
        content = msg.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            # 有些模型可能返回分段内容
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part and isinstance(part["text"], str):
                        texts.append(part["text"])
            out = "\n".join(t for t in texts if t).strip()
            return out if out else json.dumps(content, ensure_ascii=False, indent=2)

        return str(content).strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)

# ========================
# 侧边栏：智谱 API Key & 参数
# ========================
with st.sidebar:
    st.header("🔑 智谱 BigModel API Key")
    st.caption("建议把 Key 直接粘贴进来（不要带引号、不要换行）。")

    api_key = st.text_input(
        "输入 BigModel API Key",
        type="password",
        value=st.session_state["zhipu_api_key"],
        help="从 open.bigmodel.cn / docs.bigmodel.cn 获取",
    )
    st.session_state["zhipu_api_key"] = api_key

    base_url = st.text_input(
        "接口地址（chat/completions）",
        value=st.session_state["base_url"],
    )
    st.session_state["base_url"] = base_url.strip() if base_url else DEFAULT_BASE_URL

    auth_mode = st.selectbox(
        "鉴权方式",
        ["直接 API Key（推荐先试）", "JWT（id.secret 生成）"],
        index=0,
        help="如果你是 {id}.{secret} 形式且直接方式 401，可改用 JWT。",
    )

    st.divider()
    st.subheader("⚙️ 生成参数")
    text_model = st.text_input("文本模型", value=DEFAULT_TEXT_MODEL)
    vision_model = st.text_input("视觉模型", value=DEFAULT_VISION_MODEL)
    temperature = st.slider("temperature", 0.0, 1.5, 0.6, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.01)
    max_tokens = st.slider("max_tokens", 256, 8192, 2048, 128)
    enable_thinking = st.checkbox("thinking（深度思考）", value=False)

    st.divider()
    if api_key.strip():
        st.success("Key 已填写（不代表一定可用，需调用验证）。")
    else:
        st.warning("请先填写 API Key。")

# ========================
# Tab 布局
# ========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🖼 图生文（图片反推）", "📚 剧本拆分镜头", "👤 角色设定集", "📂 历史记录"]
)

# ========================
# Tab1：图生文（图片反推）
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
            **这三个功能是分开的，想用哪个就点哪个：**

            1️⃣ **风格提示词 (Style)**：画风、质感、色彩氛围等关键词。  
            2️⃣ **镜头与景别 (Shot & Composition)**：景别、机位、构图、光影。  
            3️⃣ **完整提示词 (Prompt)**：中文描述 + 对应英文 Prompt，可直接喂给模型。  
            """
        )
        style_btn = st.button("🎨 生成风格提示词 (Style)")
        shot_btn = st.button("🎥 分析镜头与景别 (Shot & Composition)")
        prompt_btn = st.button("🧠 生成完整提示词 (Prompt)")

    # --- 风格提示词 ---
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🎨 风格提示词 (Style)</div>', unsafe_allow_html=True)

    if style_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是图片风格分析专家。
请用中文输出以下内容，格式用 Markdown：

【风格总结】
- 用 1～2 句话概括整张图的视觉风格，例如：Q版动漫、暖色治愈风、赛博朋克写实、3D 卡通渲染、写实人像等。

【风格提示词（中文）】
- 用逗号分隔的中文关键词，总结：画风、质感、色彩氛围、时代感，例如：
  Q版动画风格, 治愈系, 可爱, 柔和光照, 糖果色调, 插画风, 手绘感, 高饱和度, 少女风

【风格提示词（英文，可选）】
- 如果方便，用英文再给一行对应的 style 关键词，逗号分隔，例如：
  cute chibi anime, healing style, soft lighting, pastel colors, kawaii illustration
""".strip()

            text = call_bigmodel_chat(
                base_url=st.session_state["base_url"],
                api_key=st.session_state["zhipu_api_key"],
                auth_mode=auth_mode,
                model=vision_model.strip(),
                user_text=prompt,
                images=[img],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            if text:
                st.markdown(text)
                add_history(
                    "image_style",
                    f"风格提示词 - {uploaded_image.name}",
                    {"filename": uploaded_image.name},
                    text,
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 镜头与景别 ---
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
- 远景 / 全景 / 中景 / 近景 / 特写（选择一个或多个，并说明理由）

【机位与角度】
- 机位位置：高机位 / 低机位 / 平视 / 俯视 / 仰视 / 跟拍 / 俯拍等
- 拍摄角度：正面 / 侧面 / 斜侧 / 背面 / 45度等

【构图与布局】
- 主体在画面中的位置（居中、左侧、右侧、三分构图等）
- 前景 / 中景 / 背景里分别有什么
- 是否有引导线、对称、留白等构图特点

【光线与氛围】
- 光源方向（从左/右/上/后方）
- 光线类型（直射光、散射光、逆光、轮廓光）
- 氛围（温暖、冷峻、梦幻、压抑等）
""".strip()

            text = call_bigmodel_chat(
                base_url=st.session_state["base_url"],
                api_key=st.session_state["zhipu_api_key"],
                auth_mode=auth_mode,
                model=vision_model.strip(),
                user_text=prompt,
                images=[img],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            if text:
                st.markdown(text)
                add_history(
                    "image_shot",
                    f"镜头与景别 - {uploaded_image.name}",
                    {"filename": uploaded_image.name},
                    text,
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 完整 Prompt ---
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="block-title">🧠 完整提示词 (Prompt)</div>', unsafe_allow_html=True)

    if prompt_btn:
        if not img:
            st.error("请先上传一张图片。")
        else:
            prompt = """
你现在是资深提示词工程师 + 分镜导演。
请基于这张图片，输出一段中文描述和一段英文 Prompt，格式如下（Markdown）：

【中文画面描述】
用 3～6 句中文描述画面：人物长相、穿着、动作、场景、镜头视角、光线色彩、情绪氛围等，尽量具体。

【英文生成提示词（直接给模型用）】
用一段完整英文描述同样的画面，适合给图生图/文生视频模型使用：
- 包含：人物外观（年龄、性别、发型、服饰）、动作、场景环境、镜头视角、光线、色彩、风格（cinematic, anime style, photorealistic 等）、画幅比例（16:9 或 9:16）。
- 句子自然流畅，不用堆叠“masterpiece, best quality”这类老式咒语。
- 结尾可以补充简短参数，例如：cinematic lighting, highly detailed, vertical 9:16.

【负面提示词（可选，英文）】
如果方便，请给一行英文 negative prompt，例如：
text, logo, watermark, subtitle, low resolution, blurry, distorted hands, extra limbs, deformed body
""".strip()

            text = call_bigmodel_chat(
                base_url=st.session_state["base_url"],
                api_key=st.session_state["zhipu_api_key"],
                auth_mode=auth_mode,
                model=vision_model.strip(),
                user_text=prompt,
                images=[img],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            if text:
                st.markdown(text)
                add_history(
                    "image_prompt",
                    f"完整提示词 - {uploaded_image.name}",
                    {"filename": uploaded_image.name},
                    text,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ========================
# Tab2：剧本拆分镜头
# ========================
with tab2:
    st.subheader("📚 剧本拆分镜头（主镜头 + 补充镜头 + 视频生成指令）")

    st.markdown(
        """
**痛点解决：** 普通工具“一段话只出一张图”，剧情和情绪无法完整表达。  
这里会自动拆解成：
- 主镜头（Main Keyframe）
- 多个补充镜头（Supplementary Shots：特写 / 中景 / 全景 / 环境空镜）
- 一段「视频生成指令」描述镜头顺序和节奏。
        """.strip()
    )

    script_text = st.text_area(
        "粘贴一小段剧情 / 文案（建议只描述一个场景）：",
        height=200,
        placeholder="示例：傍晚时分，林晓雨在旧木箱中发现了一只古老的铜哨......",
    )
    max_shots = st.slider("希望拆成多少个补充镜头（不含主镜头）", 2, 6, 4)
    scene_btn = st.button("🎬 开始拆分镜头")

    if scene_btn:
        if not script_text.strip():
            st.error("请先粘贴一段剧情文本。")
        else:
            prompt = f"""
你现在是资深分镜导演 + 剧本统筹。
我给你一段剧情文本，请你拆解成「主镜头 + 补充镜头 + 视频生成指令」，全部用中文输出，格式参照下方要求。

【输入剧情】
{script_text}

【输出格式（Markdown）】
### Scene 1 场景概述
- 用 1～2 句话概括这一段剧情的核心信息和情绪基调。

### 主镜头 (Main Keyframe)
- 用 3～6 句话详细描述一个“代表性画面”，作为整段剧情的主海报 / 主 keyframe：
  - 描述人物是谁、穿什么、长什么样；
  - 画面中人物在做什么；
  - 场景是哪里（室内/室外/城市/乡村/奇幻世界等）；
  - 镜头是什么景别、机位、光线和色彩氛围。

### 补充镜头 & 视觉细节 (Supplementary Shots)
请生成 {max_shots} 个补充镜头，每个用 2～4 句话描述，覆盖不同景别和视角，比如：
- 图1（特写）：人物手部特写、表情特写、重要道具等；
- 图2（中景）：人物互动、对话、情绪反应；
- 图3（全景 / 环境）：室内全景、街道全景、山谷、城市夜景等；
- 图4...：动态场面、空镜、氛围镜头等。

每个镜头用“图X 标题：”开头，然后下面分行写出画面描述，尽量带上景别、机位和情绪。

### 视频生成指令 (Video Generation)
- 用 1～3 段中文描述「如果要把这段剧情做成 8～15 秒视频，镜头应该怎么运动、按什么顺序出现」：
  - 从哪一个镜头开始（开场镜头）；
  - 镜头如何切换（从主镜头推近到特写，或从室内切到室外全景等）；
  - 节奏是慢慢铺垫还是快速剪辑；
  - 可选：给一句英文提示简要（用于视频模型）。

只输出 Markdown，不要任何额外解释。
""".strip()

            text = call_bigmodel_chat(
                base_url=st.session_state["base_url"],
                api_key=st.session_state["zhipu_api_key"],
                auth_mode=auth_mode,
                model=text_model.strip(),
                user_text=prompt,
                images=None,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
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
# Tab3：角色设定集（多风格三视图提示词）
# ========================
with tab3:
    st.subheader("👤 角色设定集（多风格三视图提示词）")

    st.markdown(
        """
**痛点解决：** AI 视频里人物长相经常变化，无法做连续剧情。  
这里不炼模型，只生成**提示词版“角色设定集”**，帮助你在不同场景中保持同一个角色。
        """.strip(),
        unsafe_allow_html=True,
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
        main_style = st.selectbox("选择一个主打风格（可在 Prompt 中重点强化）：", style_options)
        role_btn = st.button("👤 生成角色设定集提示词")

    if role_btn:
        if not role_img:
            st.error("请先上传一张角色图片。")
        else:
            prompt = f"""
你现在是角色设定师 + 提示词工程师。
根据这张角色图片，生成一个“文字版角色设定集”，用于后续在不同 AI 模型中保持人物一致性。
注意：只输出文本提示词，不生成图片、不讨论训练 LoRA。

主打风格：{main_style}

【输出格式（Markdown，全中文描述 + 英文 prompt 混排）】
### 1. 角色基础设定（Character Bible）
- 角色中文名（可以虚构）：
- 年龄、性别、性格关键词：
- 外貌概括：脸型、五官特点、发型发色，是否有标志性特征（例如：红色发带、眼下泪痣等）；
- 身形（高矮胖瘦）、气质（温柔、冷酷、元气、成熟等）；
- 主打风格设定（结合“{main_style}”）。

### 2. 脸部三视图（Face Views）提示词
请用中英结合方式，给出三个视角的提示词，每个 1～2 句：
- 正脸（Front view）：中文简述 + 对应英文提示句
- 侧脸（Side view）：中文简述 + 英文
- 背面或远景（Back / Distant view）：中文简述 + 英文

### 3. 8 种风格的人物全景三视图 Prompt（不换脸，只换穿搭和氛围）
围绕同一张脸，设计 8 种不同服装/氛围的全身三视图 Prompt。
每种风格用小标题列出，格式示例：

#### 风格A：古风侠客
- 中文：2～3 句中文描述这个角色在“古风侠客”设定下的全身造型（服装、武器、姿态、场景），注意脸还是同一个人。
- 英文 Prompt：对应一段英文提示词，可直接给图生图/视频模型使用（包括 front / side / back 全身视角说明）。

#### 风格B：都市白领（同上）
……
风格列表至少包含：古风、都市白领、修仙风格、校园校服、赛博朋克、机甲科幻、运动活力、可爱治愈这 8 类。

所有输出用 Markdown 排版，方便复制。
""".strip()

            text = call_bigmodel_chat(
                base_url=st.session_state["base_url"],
                api_key=st.session_state["zhipu_api_key"],
                auth_mode=auth_mode,
                model=vision_model.strip(),
                user_text=prompt,
                images=[role_img],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            if text:
                st.markdown("---")
                st.markdown("### 📚 角色设定集（可复制保存）")
                st.markdown(text)
                add_history(
                    "role_design",
                    f"角色设定集 - {role_image.name}",
                    {"filename": role_image.name, "main_style": main_style},
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
                f"#### 🧾 [{tag}] {item['title']}  \n"
                f"`#{item['id']}` · {item['timestamp']}"
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

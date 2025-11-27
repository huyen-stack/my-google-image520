import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# ========================
# 基本配置
# ========================

APP_TITLE = "图片 → 中文描述 + 英文提示词 生成助手"
GEMINI_MODEL_NAME = "gemini-flash-latest"  # 你也可以改成 gemini-1.5-pro 等
FREE_TIER_RPM_LIMIT = 10  # 免费典型：每分钟 10 次调用

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ========================
# 页面样式 & 顶部介绍
# ========================

st.set_page_config(page_title=APP_TITLE, page_icon="🖼️", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #020617;
        color: #e5e7eb;
    }
    .stMarkdown, .stText {
        color: #e5e7eb;
    }
    .stCode {
        font-size: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="
        padding: 18px 24px;
        border-radius: 18px;
        margin-bottom: 16px;
        background: radial-gradient(circle at top left, #22c55e 0, #020617 55%, #020617 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
    ">
      <h1 style="margin: 0 0 8px 0; color: #e5e7eb; font-size: 1.6rem;">
        🖼️ {APP_TITLE}
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传图片，让 Gemini 自动帮你生成一段<b>中文画面描述</b>（你看得懂）和一段<b>英文生成提示词</b>
        （直接复制给 SORA / VEO / Kling / Midjourney / DALL·E 等模型使用）。
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ========================
# 工具函数
# ========================

def _extract_text_from_response(resp) -> str:
    """兼容不同 Gemini 返回结构，尽量拿到纯文本"""
    text = getattr(resp, "text", None)
    if text and isinstance(text, str) and text.strip():
        return text.strip()

    try:
        texts = []
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
        if texts:
            return " ".join(texts).strip()
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return ""


def get_prompt_template() -> str:
    """
    返回：用于“生成中文描述 + 对应英文 Prompt”的系统指令。
    """
    return r"""
你现在是一个顶级图片分析专家 + 提示词工程师。
我会给你一张图片，你需要用中文 + 英文两种形式帮我描述这个画面，英文是给 AI 图生图 / 文生视频模型用的。

【任务目标】
1. 先用中文完整描述画面内容（人物长什么样、穿什么、在做什么、场景在哪里、光线/色彩/情绪是什么），
   字数大约 3～6 句，尽量具体，但不要写成小说。
2. 再写一段对应的英文生成提示词（Prompt），可以给 SORA / VEO / Kling / Midjourney / DALL·E 等模型使用：
   - 用自然英文句子描述：主角、动作、场景、镜头视角（camera）、光线、色彩、风格（例如 cinematic, photorealistic, anime style 等）、画幅比例等；
   - 不需要加太多“best quality, masterpiece, 8k” 这类老式咒语，更注重内容细节；
   - 英文里不要出现中文；
   - 结尾可以补充一小段参数描述，例如：cinematic lighting, highly detailed, vertical 9:16, 4k。

【输出格式】
请严格输出一个 JSON 对象，不能有额外文字，不能有注释，所有字符串都必须用双引号，不要有多余逗号。

JSON 结构如下（请用真实内容替换示例说明）：

{
  "scene_title_zh": "给这个画面起一个简短的中文名字，例如：海岸飞行的少女",
  "description_zh": "用 3～6 句中文详细描述画面：人物长相、穿着、动作、镜头视角、场景、光线、色彩、情绪等。",
  "prompt_en": "对应的英文生成提示词，用一段完整英文描述上面同样的画面，适合直接给图生图/文生视频模型使用。",
  "negative_prompt_en": "可选：英文负面提示词，例如：text, logo, watermark, subtitle, low resolution, blurry, distorted hands, extra limbs, deformed body。如果暂时不需要，可以给一个合理的默认负面词。"
}

【再次强调】
- 只输出一个合法 JSON 对象，不能有任何额外说明。
- 字符串使用双引号，JSON 里不要写注释。
"""


def analyze_image_to_prompt(
    img: Image.Image,
    model,
    index: int,
) -> Dict[str, Any]:
    """
    对单张图片生成：
      - scene_title_zh
      - description_zh
      - prompt_en
      - negative_prompt_en
    """
    prompt = get_prompt_template()

    try:
        resp = model.generate_content([prompt, img])
        text = _extract_text_from_response(resp)
        if not text:
            raise ValueError("模型未返回文本")

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未检测到有效 JSON 结构")

        json_str = text[start: end + 1]
        data = json.loads(json_str)

        # 兜底字段
        data.setdefault("scene_title_zh", f"场景 {index}")
        data.setdefault("description_zh", "")
        data.setdefault("prompt_en", "")
        data.setdefault("negative_prompt_en", "")

        return data

    except Exception as e:
        return {
            "scene_title_zh": f"场景 {index} · 生成失败",
            "description_zh": f"（分析失败：{e}）",
            "prompt_en": "",
            "negative_prompt_en": "",
            "error": str(e),
        }


# ========================
# 侧边栏：API Key & 参数
# ========================

with st.sidebar:
    st.header("🔑 第一步：配置 Gemini API Key")
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=st.session_state["api_key"],
        help="粘贴你的 Gemini API Key（通常以 AIza 开头）",
    )
    st.session_state["api_key"] = api_key

    st.markdown("---")
    st.caption("建议：一次分析 1~5 张图片，避免触发免费 10 RPM 限制。")

    if not api_key:
        st.warning("🔴 还没有 Key，先去 https://ai.google.dev/ 申请一个。")
    else:
        st.success("🟢 Key 已就绪。")


# ========================
# 初始化 Gemini 模型
# ========================

model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"❌ 初始化 Gemini 模型失败：{e}")
        model = None


# ========================
# 主区域：上传图片 & 分析
# ========================

st.markdown("## ① 上传图片")

uploaded_files = st.file_uploader(
    "支持多张图片（JPG / PNG / WEBP）",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

analyze_btn = st.button("🚀 开始生成中文描述 + 英文提示词", type="primary")

results: List[Dict[str, Any]] = []

if analyze_btn:
    if not api_key or model is None:
        st.error("请先在左侧输入有效的 Google API Key。")
    else:
        if not uploaded_files:
            st.warning("请至少上传一张图片。")
        else:
            if len(uploaded_files) > FREE_TIER_RPM_LIMIT:
                st.warning(
                    f"当前上传了 {len(uploaded_files)} 张图片，建议一次不超过 {FREE_TIER_RPM_LIMIT} 张，"
                    "以避免触发免费 10 RPM 限制。"
                )

            with st.spinner("🧠 正在为每张图片生成中文描述和英文 Prompt..."):
                for i, file in enumerate(uploaded_files, start=1):
                    img = Image.open(file).convert("RGB")
                    st.write(f"正在分析：第 {i} 张图 —— {file.name}")
                    data = analyze_image_to_prompt(img, model, index=i)
                    data["_meta"] = {
                        "index": i,
                        "filename": file.name,
                    }
                    results.append(data)

            if results:
                export_data = {
                    "meta": {
                        "model": GEMINI_MODEL_NAME,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_count": len(results),
                    },
                    "items": results,
                }
                st.session_state["last_result"] = export_data


# ========================
# 展示结果：一图一张“卡片”
# ========================

if st.session_state["last_result"]:
    export_data = st.session_state["last_result"]
    items: List[Dict[str, Any]] = export_data["items"]

    st.markdown("## ② 结果预览（中文在上，英文在下，直接复制即可）")

    tab_cards, tab_json = st.tabs(["📋 按图片查看卡片", "📦 JSON 导出"])

    # --- Tab1：逐图片展示 ---
    with tab_cards:
        for item in items:
            meta = item.get("_meta", {})
            idx = meta.get("index", "?")
            filename = meta.get("filename", "unknown")

            title = item.get("scene_title_zh", f"场景 {idx}")
            desc_zh = item.get("description_zh", "")
            prompt_en = item.get("prompt_en", "")
            neg_en = item.get("negative_prompt_en", "")

            st.markdown(f"### 🖼 图片 {idx} · {filename}")
            st.markdown(f"**场景名称：** {title}")

            # 6️⃣ 中文版画面描述
            st.markdown("#### 6️⃣ 中文版画面描述（看这个就行）")
            if desc_zh:
                st.markdown(f"> {desc_zh}")
            else:
                st.info("暂无中文描述。")

            # 6️⃣-EN 英文生成提示词
            st.markdown("#### 6️⃣-EN 英文生成提示词（直接整段复制给模型）")
            st.code(prompt_en or "（暂无英文提示词）", language="text")

            # 负面提示词
            st.markdown("#### 🚫 负面提示词（可选）")
            st.code(
                neg_en or "text, logo, watermark, subtitle, low resolution, blurry, distorted hands, extra limbs, deformed body",
                language="text",
            )

            st.markdown("---")

    # --- Tab2：JSON 导出 ---
    with tab_json:
        st.markdown("### 📦 下载本次生成结果 JSON")
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ 下载 image_prompts.json",
            data=json_str,
            file_name="image_prompts.json",
            mime="application/json",
        )

        st.markdown("### 🔍 JSON 内容预览")
        preview = json_str[:4000] + ("\n...\n" if len(json_str) > 4000 else "")
        st.code(preview, language="json")
else:
    st.info("👈 先在左侧填好 Key，上传一两张图片，然后点击“🚀 开始生成中文描述 + 英文提示词”。")

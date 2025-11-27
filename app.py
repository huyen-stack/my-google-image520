import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# ========================
# 基本配置
# ========================

APP_TITLE = "人物风格 & 服装风格 · 图片分析助手"
GEMINI_MODEL_NAME = "gemini-flash-latest"  # 你也可以改成 gemini-1.5-pro 等
FREE_TIER_RPM_LIMIT = 10  # 免费典型：每分钟 10 次调用

if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ========================
# 页面样式 & 顶部介绍
# ========================

st.set_page_config(page_title=APP_TITLE, page_icon="🧍‍♀️", layout="wide")

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
        🧍‍♀️ {APP_TITLE}
      </h1>
      <p style="margin: 0; color: #cbd5f5; font-size: 0.96rem;">
        上传图片，让 Gemini 自动分析出：人物大概属于什么文化/区域风格、穿什么风格的衣服，
        输出统一的 JSON 结构，方便你后续做 SORA / VEO / MJ / 人物复刻等工作流。
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


def get_style_prompt() -> str:
    """
    返回：专门用于“人物风格 + 服装风格”分析的系统指令 Prompt。
    你后面想微调字段，可以直接改这里。
    """
    return r"""
你现在是一个顶级图片分析专家 + 时尚造型师 + 视觉风格设计师。

【你的任务】
给你一张图片，请你从中提取出「人物风格」和「服装风格」等关键信息，用结构化 JSON 输出。
重点是：
1. 不要直接判断真实国籍 / 种族；
2. 只做“风格 / 区域气质”的推测，例如：日系少女风、韩系街头风、欧美商务风、中式传统风；
3. 对服装的款式、颜色、正式程度、细节做尽量具体的描述，方便后续生成或复刻人物。

【禁止的行为】
- 不要输出：这是中国人/日本人/韩国人/美国人等直接国籍判断。
- 不要猜测真实身份（明星、公众人物等），只谈“看起来像 XX 风格”。

【输出要求】
请你只输出一个 JSON 对象，不能有任何解释文字、注释或多余内容。
所有字符串必须使用双引号，不能使用单引号，JSON 中不能有注释，不能有多余逗号。

JSON 的字段结构如下（请用真实分析内容替换下面示例说明文本）：

{
  "character": {
    "age_look_zh": "人物的年龄观感，例如：十几岁少女 / 二十多岁青年 / 中年男性",
    "gender_zh": "性别观感，例如：女性 / 男性 / 不明显",
    "body_type_zh": "体型观感，例如：偏瘦 / 匀称 / 健壮 / 微胖",
    "face_description_zh": "用 1～2 句中文概括脸部特征（脸型、五官比例、有没有妆容等）",
    "hair_zh": "发型与发色，例如：短棕发，刘海，头顶有一个大红蝴蝶结发饰",
    "gesture_pose_zh": "当前身体姿态与大致动作，例如：身体前倾，双手抓住扫帚，双腿向后弯曲"
  },

  "region_style_hint": {
    "broad_region_tag": "从视觉风格上大致归类的区域标签之一：east_asian / western / middle_eastern / south_asian / african / latin_american / mixed / unclear",
    "fashion_culture_tags": [
      "若干英文风格标签，例如：#japanese_schoolgirl_style, #korean_streetwear, #european_business, #chinese_traditional_hanfu"
    ],
    "notes_zh": "用中文解释为什么这么判断，例如：服装版型接近日系校服，发饰和整体配色偏日系少女风，不代表真实国籍。"
  },

  "clothing": {
    "category_en": "衣服大类（英文），例如：dress / school_uniform / hoodie / suit / hanfu / kimono / sportswear / casual_outfit",
    "formality_en": "正式程度（英文），例如：casual / smart_casual / business / formal / traditional_ceremonial / streetwear",
    "main_colors_en": [
      "若干主色英文，例如：navy_blue, black, white, red_accent"
    ],
    "fit_en": "版型宽松度（英文），例如：loose / regular / slim / oversized",
    "style_tags_en": [
      "服装风格标签（英文），例如：#japanese_street, #korean_casual, #european_minimalism, #cute_girly, #sporty"
    ],
    "details_zh": "用 2～4 句中文详细描述服装：款式（连衣裙/外套/衬衫/裙裤等）、长度、细节（领口、袖口、纽扣、有无图案或条纹）、配件（包、帽子、眼镜、首饰、鞋子），以及整体给人的风格感受。"
  },

  "accessories": {
    "items_zh": [
      "列出重要配饰的中文描述，例如：大红蝴蝶结发饰、棕色皮质双肩包、棕色短靴"
    ],
    "style_tags_en": [
      "用英文标签概括配饰风格，例如：#schoolgirl_bow, #vintage_leather_backpack"
    ]
  },

  "scene_brief_zh": "用 1～2 句中文概括人物所处场景的大致类型，例如：海岸公路上空飞行、城市街头夜景、人坐在咖啡馆室内、工厂车间等。",

  "emotion_mood_zh": "用 1～2 句中文概括画面的情绪氛围，例如：轻松愉快的旅行感、速度感和紧张感并存、温暖治愈、冷峻科幻等。",

  "generation_hints": {
    "character_prompt_en": "一段专门用来复刻人物的英文提示词，包含：年龄、性别、发型发色、头饰、服装款式与颜色、整体风格（例如：a teenage girl with short brown hair and a big red bow, wearing a simple navy blue dress and brown leather backpack, in a japanese-inspired schoolgirl style）。",
    "outfit_prompt_en": "一段专门描述服装与配饰的英文提示词，方便在其它场景中套用同一套衣服。",
    "style_summary_zh": "用中文总结一句“这个人物是什么风格的人设”，例如：日系校园少女风、韩系街头潮男、欧美商务女强人、中式古风侠客等。"
  }
}

【再次强调】
- 你必须输出一个合法 JSON 对象，不能有任何额外文字。
- 字符串全部使用双引号。
- 不要输出注释，不要在 JSON 外多说一句话。
- 不能直接写“这是某国人”，只能用风格/气质标签来描述。
"""


def analyze_single_image(
    img: Image.Image,
    model,
    index: int,
) -> Dict[str, Any]:
    """
    对单张图片做人物风格 + 服装风格分析，返回 JSON dict。
    """
    prompt = get_style_prompt()

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

        # 兜底，避免 KeyError
        data.setdefault("character", {})
        data.setdefault("region_style_hint", {})
        data.setdefault("clothing", {})
        data.setdefault("accessories", {})
        data.setdefault("scene_brief_zh", "")
        data.setdefault("emotion_mood_zh", "")
        data.setdefault("generation_hints", {})

        return data

    except Exception as e:
        return {
            "error": str(e),
            "character": {},
            "region_style_hint": {},
            "clothing": {},
            "accessories": {},
            "scene_brief_zh": "",
            "emotion_mood_zh": "",
            "generation_hints": {},
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
    st.caption("建议：一次分析 1~5 张图片，避免免费 10 RPM 频率限制。")

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

analyze_btn = st.button("🚀 开始分析人物风格 & 服装风格", type="primary")

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

            with st.spinner("🧠 正在分析图片中的人物风格与服装风格..."):
                for i, file in enumerate(uploaded_files, start=1):
                    img = Image.open(file).convert("RGB")
                    st.write(f"正在分析：第 {i} 张图 —— {file.name}")
                    data = analyze_single_image(img, model, index=i)
                    # 附加一些元信息方便后续导出
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
# 展示结果
# ========================

if st.session_state["last_result"]:
    export_data = st.session_state["last_result"]
    items: List[Dict[str, Any]] = export_data["items"]

    st.markdown("## ② 分析结果预览")

    tab_cards, tab_json = st.tabs(["📋 按图片查看", "📦 JSON 导出"])

    # --- Tab1：逐图片展示 ---
    with tab_cards:
        for item in items:
            meta = item.get("_meta", {})
            idx = meta.get("index", "?")
            filename = meta.get("filename", "unknown")

            st.markdown(f"### 🖼 图片 {idx} · {filename}")

            # 左右布局：左边可以重现图片（如果你想的话，可以重新读取）
            col1, col2 = st.columns([1.1, 2])

            with col1:
                st.caption("（如果想看原图，可以在上面上传区重新选择预览）")
                st.json(
                    {
                        "年龄观感": item.get("character", {}).get("age_look_zh", ""),
                        "性别观感": item.get("character", {}).get("gender_zh", ""),
                        "体型观感": item.get("character", {}).get("body_type_zh", ""),
                    }
                )

            with col2:
                # 人物整体描述
                st.markdown("**👤 人物特征（中文总结）：**")
                char = item.get("character", {}) or {}
                char_lines = [
                    f"年龄观感：{char.get('age_look_zh', '')}",
                    f"性别观感：{char.get('gender_zh', '')}",
                    f"体型观感：{char.get('body_type_zh', '')}",
                    f"脸部特征：{char.get('face_description_zh', '')}",
                    f"发型发色：{char.get('hair_zh', '')}",
                    f"姿态动作：{char.get('gesture_pose_zh', '')}",
                ]
                st.code("\n".join(char_lines), language="markdown")

                # 区域/风格标签
                st.markdown("**🌏 文化 / 区域风格标签（不是国籍）：**")
                region = item.get("region_style_hint", {}) or {}
                region_text = json.dumps(region, ensure_ascii=False, indent=2)
                st.code(region_text, language="json")

                # 服装风格
                st.markdown("**👗 服装风格（结构化）：**")
                cloth = item.get("clothing", {}) or {}
                cloth_text = json.dumps(cloth, ensure_ascii=False, indent=2)
                st.code(cloth_text, language="json")

                # 配饰
                st.markdown("**🎒 配饰：**")
                acc = item.get("accessories", {}) or {}
                acc_text = json.dumps(acc, ensure_ascii=False, indent=2)
                st.code(acc_text, language="json")

                # 场景 & 情绪
                st.markdown("**🏞 场景 & 情绪：**")
                scene = item.get("scene_brief_zh", "") or ""
                mood = item.get("emotion_mood_zh", "") or ""
                st.code(
                    f"场景概括：{scene}\n情绪氛围：{mood}",
                    language="markdown",
                )

                # 生成提示词建议
                st.markdown("**🎯 生成用提示词模板（可直接塞给 SORA / VEO）：**")
                gen = item.get("generation_hints", {}) or {}
                gen_text = json.dumps(gen, ensure_ascii=False, indent=2)
                st.code(gen_text, language="json")

            st.markdown("---")

    # --- Tab2：JSON 导出 ---
    with tab_json:
        st.markdown("### 📦 下载本次分析结果 JSON")
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ 下载 image_style_analysis.json",
            data=json_str,
            file_name="image_style_analysis.json",
            mime="application/json",
        )

        st.markdown("### 🔍 JSON 内容预览")
        preview = json_str[:4000] + ("\n...\n" if len(json_str) > 4000 else "")
        st.code(preview, language="json")
else:
    st.info("👈 先在左侧填好 Key，上传一两张图片，然后点击“🚀 开始分析人物风格 & 服装风格”。")

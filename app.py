import io
import json
import re
import time
from pathlib import Path

import requests
import streamlit as st
from docx import Document
from docx.shared import Pt

# ============================================================
# DỊCH TRUYỆN AI - V1
# DeepSeek API + Word DOCX
# ============================================================

st.set_page_config(
    page_title="Dịch Truyện AI",
    page_icon="📖",
    layout="wide",
)

SYSTEM_TRANSLATE = """
Bạn là biên tập viên và dịch giả văn học Trung -> Việt chuyên nghiệp.

Mục tiêu:
- Dịch tự nhiên như văn truyện tiếng Việt, không dịch máy cứng nhắc.
- Không tự ý thêm, bớt hoặc giải thích nội dung.
- Giữ đúng ý, sắc thái, cảm xúc, quan hệ và bối cảnh.
- Tên nhân vật, địa danh, tổ chức và thuật ngữ phải nhất quán theo BỘ NHỚ TRUYỆN.
- Tuyệt đối ưu tiên quy tắc xưng hô đã có trong BỘ NHỚ TRUYỆN.
- Nếu ngữ cảnh mới cho thấy một quy tắc cần thay đổi, không tự ý đổi giữa chừng; đánh dấu nhu cầu cập nhật ở cuối bằng [CẦN CẬP NHẬT BỘ NHỚ].
- Giữ bố cục đoạn văn: mỗi đoạn tiếng Trung tương ứng một đoạn tiếng Việt.
- Không dịch tiêu đề chương thành nội dung khác.
- Không đưa lời bình của người dịch vào bản dịch.

Quy tắc văn phong:
- Tiếng Việt tự nhiên, mượt, phù hợp văn học.
- Đối thoại phải giống lời nói của nhân vật.
- Câu văn có thể đảo trật tự để tự nhiên trong tiếng Việt nhưng không được làm sai nghĩa.
"""

SYSTEM_MEMORY = """
Bạn là biên tập viên văn học Trung -> Việt.
Hãy đọc các đoạn truyện được cung cấp và xây dựng BỘ NHỚ TRUYỆN để dùng cho các chương sau.

Chỉ ghi những gì có căn cứ từ văn bản. Không đoán bừa.
Ưu tiên:
1. Nhân vật: tên gốc, tên Việt/Hán-Việt nên dùng, giới tính nếu xác định được, vai trò, đặc điểm.
2. Quan hệ giữa nhân vật.
3. Xưng hô: ai gọi ai thế nào; ngôi kể dùng thế nào.
4. Địa danh.
5. Tổ chức, môn phái, trường học, công ty, vật phẩm.
6. Thuật ngữ đặc biệt.
7. Bối cảnh/thể loại.
8. Các quy tắc dịch cần giữ nguyên.

Trả về JSON hợp lệ, không markdown, theo cấu trúc:
{
  "characters": [],
  "relationships": [],
  "address_rules": [],
  "places": [],
  "organizations": [],
  "terms": [],
  "setting": "",
  "translation_rules": []
}
"""

SYSTEM_MERGE = """
Bạn là biên tập viên quản lý bộ nhớ truyện.
Hãy hợp nhất các BỘ NHỚ TRUYỆN thành một bộ nhớ duy nhất.

Nguyên tắc:
- Không xóa thông tin hữu ích chỉ vì xuất hiện ở phần khác.
- Nếu cùng một nhân vật có nhiều cách viết, chọn cách nhất quán và phổ biến nhất trong văn bản.
- Không tự bịa giới tính, quan hệ hoặc xưng hô nếu văn bản chưa đủ căn cứ.
- Xưng hô phải được ghi theo cặp người nói -> người nghe.
- Thuật ngữ phải có bản dịch thống nhất.
- Nếu có mâu thuẫn thật sự, ghi vào "translation_rules" để khi dịch cần kiểm tra ngữ cảnh.

Chỉ trả về JSON hợp lệ.
"""

WORD_RE = re.compile(r"\S+")


def get_secret_key():
    try:
        return st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        return ""


def call_deepseek(api_key, model, system_prompt, user_prompt, temperature=0.2, retries=3):
    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    last_error = None

    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=180,
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            last_error = f"HTTP {response.status_code}: {response.text[:1000]}"

        except Exception as exc:
            last_error = str(exc)

        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(last_error or "Không nhận được phản hồi từ DeepSeek.")


def extract_json(text):
    text = text.strip()

    # Bỏ markdown fence nếu model tự thêm.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Tìm object JSON đầu tiên.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError("DeepSeek không trả về JSON hợp lệ.")


def word_count(text):
    return len(WORD_RE.findall(text or ""))


def read_uploaded_file(uploaded_file):
    """Read either DOCX or TXT and return a list of paragraphs."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".docx":
        doc = Document(io.BytesIO(uploaded_file.getvalue()))

        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs.append(text)

        return paragraphs

    if suffix == ".txt":
        raw = uploaded_file.getvalue()

        # Try common encodings used by Vietnamese/Chinese text files.
        text = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ValueError(
                "Không đọc được file TXT. Hãy lưu file dưới dạng UTF-8 "
                "hoặc UTF-8 with BOM."
            )

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Blank lines separate paragraphs. If the TXT has no blank lines,
        # each non-empty line is treated as one paragraph.
        blocks = re.split(r"\n\s*\n+", text)

        paragraphs = []
        for block in blocks:
            block = block.strip()
            if block:
                paragraphs.append(block)

        return paragraphs

    raise ValueError("Chỉ hỗ trợ file .docx hoặc .txt.")


def detect_chapters(paragraphs):
    chapter_pattern = re.compile(
        r"^\s*(?:第\s*[0-9零一二三四五六七八九十百千万]+\s*[章回节卷]|"
        r"第\s*\d+\s*[章回节卷]|"
        r"chapter\s+\d+)\b",
        re.I,
    )

    chapters = []
    current = []

    for p in paragraphs:
        if chapter_pattern.match(p):
            if current:
                chapters.append(current)
            current = [p]
        else:
            current.append(p)

    if current:
        chapters.append(current)

    # Nếu file không có tiêu đề chương rõ ràng, coi toàn bộ file là 1 chương.
    if len(chapters) <= 1:
        return [paragraphs]

    return chapters


def split_for_api(paragraphs, max_words=6500):
    chunks = []
    current = []
    count = 0

    for p in paragraphs:
        wc = word_count(p)

        if current and count + wc > max_words:
            chunks.append(current)
            current = []
            count = 0

        current.append(p)
        count += wc

    if current:
        chunks.append(current)

    return chunks


def compact_memory(memory):
    return json.dumps(memory, ensure_ascii=False, indent=2)


def analyze_story(api_key, model, paragraphs, progress=None):
    batches = split_for_api(paragraphs, max_words=6500)

    partials = []

    for i, batch in enumerate(batches, 1):
        text = "\n\n".join(batch)

        prompt = f"""
Hãy phân tích phần {i}/{len(batches)} của truyện dưới đây.

=== TRUYỆN ===
{text}
=== HẾT ===

Chỉ trả về JSON theo cấu trúc đã yêu cầu.
"""

        result = call_deepseek(
            api_key,
            model,
            SYSTEM_MEMORY,
            prompt,
            temperature=0.1,
        )

        try:
            partials.append(extract_json(result))
        except Exception:
            # Nếu một batch lỗi JSON, bỏ qua batch đó thay vì làm mất toàn bộ.
            continue

        if progress:
            progress(i / len(batches))

    if not partials:
        raise RuntimeError("Không tạo được bộ nhớ truyện từ DeepSeek.")

    if len(partials) == 1:
        return partials[0]

    merge_prompt = f"""
Hãy hợp nhất các bộ nhớ sau thành MỘT bộ nhớ truyện duy nhất.

{json.dumps(partials, ensure_ascii=False, indent=2)}

Chỉ trả về JSON.
"""

    merged = call_deepseek(
        api_key,
        model,
        SYSTEM_MERGE,
        merge_prompt,
        temperature=0.0,
    )

    return extract_json(merged)


def chapter_title(chapter, number):
    first = chapter[0].strip() if chapter else ""

    # Nếu có tiêu đề 第xx章 thì giữ nguyên số chương nhưng chuyển tiêu đề thành Chương X.
    m = re.search(r"第\s*(\d+)\s*[章回节]", first, re.I)

    if m:
        return f"Chương {m.group(1)}"

    # Nếu file đã có "Chương 1".
    m = re.search(r"(?:chương|chapter)\s*(\d+)", first, re.I)

    if m:
        return f"Chương {m.group(1)}"

    return f"Chương {number}"


def translate_chapter(api_key, model, chapter, memory, number, style):
    title = chapter_title(chapter, number)

    # Bỏ dòng tiêu đề chương gốc nếu nó thực sự là tiêu đề.
    content = chapter[:]
    if content:
        first = content[0].strip()
        if (
            re.match(r"^第\s*\d+\s*[章回节]", first, re.I)
            or re.match(r"^(?:chương|chapter)\s*\d+", first, re.I)
        ):
            content = content[1:]

    chunks = split_for_api(content, max_words=5500)
    translated_parts = []

    memory_text = compact_memory(memory)

    for part in chunks:
        source = "\n\n".join(part)

        prompt = f"""
BỘ NHỚ TRUYỆN:
{memory_text}

PHONG CÁCH:
{style}

TIÊU ĐỀ:
{title}

Hãy dịch phần văn bản sau sang tiếng Việt.

=== BẮT ĐẦU ===
{source}
=== KẾT THÚC ===

Chỉ trả về bản dịch tiếng Việt.
Giữ đúng số đoạn: mỗi đoạn nguồn cách nhau bằng một dòng trống thì bản dịch cũng cách nhau bằng một dòng trống.
Không thêm chú thích.
"""

        result = call_deepseek(
            api_key,
            model,
            SYSTEM_TRANSLATE,
            prompt,
            temperature=0.3,
        )

        translated_parts.append(result.strip())

    return title, "\n\n".join(translated_parts)


def create_docx(translated_chapters):
    out = Document()

    # Xóa paragraph trắng mặc định.
    for p in list(out.paragraphs):
        p._element.getparent().remove(p._element)

    for idx, (title, text) in enumerate(translated_chapters):
        if idx > 0:
            out.add_page_break()

        p = out.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(16)

        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()

            if paragraph:
                out.add_paragraph(paragraph)

    buffer = io.BytesIO()
    out.save(buffer)
    return buffer.getvalue()


# ============================================================
# UI
# ============================================================

st.title("📖 Dịch Truyện AI")
st.caption(
    "Dịch truyện Trung → Việt bằng DeepSeek, hỗ trợ Word (.docx) và TXT (.txt), "
    "có bộ nhớ nhân vật, xưng hô và thuật ngữ để giữ nhất quán xuyên suốt truyện."
)

with st.sidebar:
    st.header("⚙️ Cài đặt")

    secret_key = get_secret_key()

    if secret_key:
        api_key = secret_key
        st.success("✅ Đã nhận DEEPSEEK_API_KEY từ Secrets.")
    else:
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="Tạm thời có thể nhập tại đây để thử. Không lưu API key vào GitHub.",
        )

    model = st.text_input(
        "Model DeepSeek",
        value="deepseek-chat",
        help="Có thể thay bằng model DeepSeek mà tài khoản/API của bạn đang hỗ trợ.",
    )

    style = st.text_area(
        "Phong cách dịch",
        value=(
            "Văn phong truyện tự nhiên, mượt, dễ đọc. "
            "Giữ sắc thái cảm xúc và phù hợp thể loại đam mỹ/tiểu thuyết. "
            "Đối thoại tự nhiên, không máy móc."
        ),
        height=130,
    )

uploaded = st.file_uploader(
    "📄 Tải file Word hoặc TXT tiếng Trung",
    type=["docx", "txt"],
)

if uploaded is None:
    st.info("👆 Hãy tải một file .docx lên để bắt đầu.")
    st.stop()

try:
    paragraphs = read_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Không đọc được file Word: {exc}")
    st.stop()

chapters = detect_chapters(paragraphs)

st.success(
    f"Đã đọc **{uploaded.name}** · "
    f"{len(paragraphs):,} đoạn · "
    f"phát hiện khoảng **{len(chapters):,} chương/phần**."
)

if not api_key:
    st.warning(
        "Bạn chưa cấu hình DeepSeek API key. "
        "Hãy thêm DEEPSEEK_API_KEY vào Streamlit Secrets."
    )
    st.stop()

if st.button("🧠 PHÂN TÍCH TRUYỆN", use_container_width=True):
    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Đang đọc truyện và xây dựng bộ nhớ nhân vật/xưng hô...")
        memory = analyze_story(
            api_key,
            model,
            paragraphs,
            progress=progress.progress,
        )

        st.session_state["story_memory"] = memory
        st.session_state["story_name"] = uploaded.name

        progress.progress(1.0)
        status.success("✅ Đã tạo bộ nhớ truyện.")
    except Exception as exc:
        st.error(f"Phân tích thất bại: {exc}")

if "story_memory" in st.session_state:
    memory = st.session_state["story_memory"]

    with st.expander("🧠 Xem bộ nhớ truyện", expanded=False):
        st.json(memory)

    st.divider()

    if st.button(
        "✨ BẮT ĐẦU DỊCH TRUYỆN",
        type="primary",
        use_container_width=True,
    ):
        translated = []
        progress = st.progress(0)
        status = st.empty()

        try:
            for i, chapter in enumerate(chapters, 1):
                status.info(
                    f"Đang dịch Chương {i}/{len(chapters)}..."
                )

                title, text = translate_chapter(
                    api_key,
                    model,
                    chapter,
                    memory,
                    i,
                    style,
                )

                translated.append((title, text))
                progress.progress(i / len(chapters))

            st.session_state["translated_chapters"] = translated

            status.success(
                f"✅ Đã dịch xong {len(translated):,} chương."
            )

        except Exception as exc:
            st.error(f"Dịch thất bại: {exc}")

if "translated_chapters" in st.session_state:
    translated = st.session_state["translated_chapters"]

    st.subheader("📚 Kết quả")

    st.write(f"Đã hoàn thành: **{len(translated):,} chương**.")

    output = create_docx(translated)

    original = Path(
        st.session_state.get("story_name", "truyen.docx")
    ).stem

    filename = f"{original}_dich_tieng_viet.docx"

    st.download_button(
        "⬇️ TẢI FILE WORD ĐÃ DỊCH",
        data=output,
        file_name=filename,
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        use_container_width=True,
    )

    with st.expander("👀 Xem thử bản dịch"):
        for i, (title, text) in enumerate(translated[:3], 1):
            st.markdown(f"### {title}")
            st.write(text[:5000])
            if i < min(3, len(translated)):
                st.divider()

st.divider()
st.caption(
    "V1 · DeepSeek API · Bộ nhớ nhân vật/xưng hô/thuật ngữ · Xuất một file Word"
)

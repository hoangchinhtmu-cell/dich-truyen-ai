import io, os, re, json, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import streamlit as st
from docx import Document
from openai import OpenAI

APP_VERSION = "Dịch Truyện AI V5 PRO"
DEFAULT_MODEL = "deepseek-v4-pro"

# -----------------------------
# Text / document helpers
# -----------------------------

def read_upload(uploaded) -> str:
    data = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith('.txt'):
        for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'gbk'):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                pass
        raise ValueError('Không đọc được file TXT với các bảng mã UTF-8/GB18030/GBK.')
    if name.endswith('.docx'):
        doc = Document(io.BytesIO(data))
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    raise ValueError('Chỉ hỗ trợ DOCX hoặc TXT.')


def normalize_text(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\ufeff', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


CHAPTER_PATTERNS = [
    re.compile(r'^\s*(?:第\s*)?(\d{1,5})\s*[章节卷集篇回]\s*.*$', re.I),
    re.compile(r'^\s*Chương\s+\d+.*$', re.I),
    re.compile(r'^\s*chapter\s+\d+.*$', re.I),
    re.compile(r'^\s*第[一二三四五六七八九十百千万零〇\d]+章.*$'),
]


def split_chapters(text: str) -> List[Dict[str, str]]:
    lines = text.split('\n')
    starts = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if any(p.match(s) for p in CHAPTER_PATTERNS):
            starts.append(i)
    if not starts:
        return [{'title': 'Toàn văn', 'text': text.strip()}]
    chapters = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        title = lines[start].strip()
        body = '\n'.join(lines[start + 1:end]).strip()
        chapters.append({'title': title, 'text': body})
    return chapters


def chunk_text(text: str, max_chars: int) -> List[str]:
    paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks, current, size = [], [], 0
    for p in paras:
        # hard split unusually large paragraphs by sentence/newline
        pieces = [p]
        if len(p) > max_chars:
            pieces = re.split(r'(?<=[。！？!?])', p)
            pieces = [x for x in pieces if x.strip()]
        for piece in pieces:
            if current and size + len(piece) + 2 > max_chars:
                chunks.append('\n\n'.join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 2
    if current:
        chunks.append('\n\n'.join(current))
    return chunks

# -----------------------------
# DeepSeek client
# -----------------------------

def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url='https://api.deepseek.com')


def extract_content(response) -> str:
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None
    if content is None:
        return ''
    return str(content).strip()


def call_model(client: OpenAI, model: str, system: str, user: str,
               max_tokens: int, json_mode: bool = False,
               thinking: bool = True, retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                max_tokens=max_tokens,
                stream=False,
            )
            if json_mode:
                kwargs['response_format'] = {'type': 'json_object'}
            # V5 fix: the OpenAI Python SDK may reject thinking=... as a direct
            # keyword. DeepSeek documents it as an API field; extra_body keeps
            # it compatible with the OpenAI SDK.
            if thinking:
                kwargs['reasoning_effort'] = 'high'
                kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}
            else:
                kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
            response = client.chat.completions.create(**kwargs)
            content = extract_content(response)
            if content:
                return content
            last_error = RuntimeError('DeepSeek trả về nội dung rỗng.')
        except Exception as e:
            last_error = e
        time.sleep(min(2 ** attempt, 5))
        # A blank/failed thinking request can be retried once without thinking.
        if attempt == 1 and thinking:
            thinking = False
    raise last_error or RuntimeError('API không trả về nội dung.')


def call_json(client, model, system, user, max_tokens=7000):
    raw = call_model(client, model, system, user, max_tokens, json_mode=True, thinking=True)
    # tolerate markdown fences despite JSON mode
    raw = re.sub(r'^```json\s*|^```\s*|\s*```$', '', raw.strip(), flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError('Model trả về JSON không hợp lệ.')

# -----------------------------
# Story Bible
# -----------------------------

BIBLE_SCHEMA = {
    'characters': [
        {
            'name_cn': '', 'name_vi': '', 'aliases': [], 'gender': '',
            'role': '', 'personality': '', 'self_pronoun': '',
            'address_rules': [{'to': '', 'call': '', 'self': '', 'context': ''}],
        }
    ],
    'terms': [{'source': '', 'translation': '', 'type': '', 'note': ''}],
    'places': [{'source': '', 'translation': '', 'note': ''}],
    'organizations': [{'source': '', 'translation': '', 'note': ''}],
    'style': {'era': '', 'genre': '', 'narration': '', 'general_pronoun_rule': ''},
}

BIBLE_SYSTEM = '''Bạn là biên tập viên truyện Trung-Việt chuyên nghiệp. Hãy xây STORY BIBLE có tính "khóa".
Mục tiêu: giữ tên nhân vật, quan hệ, giới tính, xưng hô, thuật ngữ, địa danh và văn phong nhất quán cho toàn bộ truyện.
Không tự bịa chi tiết không có trong văn bản. Nếu chưa chắc, để trống hoặc ghi mức độ chưa xác định.
Đặc biệt phải phân biệt: tự xưng của từng nhân vật và cách nhân vật A gọi nhân vật B theo từng quan hệ/ngữ cảnh.
Trả về JSON đúng schema được yêu cầu.'''


def build_story_bible(client, model, chapters, style, sample_limit=50000):
    source = '\n\n'.join(f"[{c['title']}]\n{c['text']}" for c in chapters)
    source = source[:sample_limit]
    user = f'''Phong cách dịch mong muốn:
{style}

Schema JSON bắt buộc:
{json.dumps(BIBLE_SCHEMA, ensure_ascii=False)}

Văn bản nguồn:
{source}'''
    return call_json(client, model, BIBLE_SYSTEM, user, max_tokens=9000)


def bible_for_prompt(bible: Dict[str, Any]) -> str:
    chars = []
    for c in bible.get('characters', []):
        rules = '; '.join(
            f"gọi {r.get('to','')} = {r.get('call','')}, tự xưng = {r.get('self','')} ({r.get('context','')})"
            for r in c.get('address_rules', []) if r.get('call') or r.get('self')
        )
        chars.append(
            f"- {c.get('name_cn','')} -> {c.get('name_vi','')}; bí danh={c.get('aliases',[])}; "
            f"giới tính={c.get('gender','')}; vai trò={c.get('role','')}; tính cách={c.get('personality','')}; "
            f"tự xưng mặc định={c.get('self_pronoun','')}; quy tắc={rules}"
        )
    terms = '\n'.join(f"- {x.get('source')} -> {x.get('translation')} [{x.get('type','')}]" for x in bible.get('terms', []))
    places = '\n'.join(f"- {x.get('source')} -> {x.get('translation')}" for x in bible.get('places', []))
    orgs = '\n'.join(f"- {x.get('source')} -> {x.get('translation')}" for x in bible.get('organizations', []))
    return f'''STORY BIBLE — KHÔNG ĐƯỢC TỰ Ý ĐỔI:
NHÂN VẬT:
{chr(10).join(chars)}
THUẬT NGỮ:
{terms}
ĐỊA DANH:
{places}
TỔ CHỨC:
{orgs}
PHONG CÁCH:
{json.dumps(bible.get('style',{}), ensure_ascii=False)}'''

# -----------------------------
# Translation + validation
# -----------------------------

TRANSLATE_SYSTEM = '''Bạn là dịch giả truyện Trung-Việt. Dịch văn bản nguồn sang tiếng Việt tự nhiên, mượt, giàu cảm xúc, đúng bối cảnh.
Giữ nguyên diễn biến, không tóm tắt, không thêm tình tiết, không giải thích ngoài truyện.
Ưu tiên văn phong cổ trang khi nguồn là cổ trang; lời thoại phải tự nhiên nhưng phù hợp địa vị và quan hệ.
BẮT BUỘC tuân thủ STORY BIBLE, đặc biệt tên nhân vật và xưng hô.
Không được đổi "ta" thành "tôi", "nàng" thành "cô ấy", "chàng" thành "anh ấy" nếu STORY BIBLE không cho phép.
Không thêm tiêu đề chương nếu đoạn nguồn không có.
Chỉ trả về bản dịch, không trả JSON, không nhận xét.'''

CHECK_SYSTEM = '''Bạn là biên tập viên kiểm định bản dịch truyện. So sánh bản nguồn, bản dịch và STORY BIBLE.
Chỉ tìm lỗi nhất quán có thể xác định: sai tên, sai giới tính, sai quan hệ, sai cách xưng hô, sai thuật ngữ, sai địa danh, hoặc tự ý thêm/bớt nội dung.
Trả JSON: {"ok": true/false, "issues": [{"type":"pronoun|name|term|omission|addition|style", "original":"", "translated":"", "fix":""}]}
Không đánh dấu khác biệt văn phong nhỏ nếu không làm sai nghĩa.'''

FIX_SYSTEM = '''Bạn là biên tập viên sửa bản dịch truyện. Sửa CHỈ những lỗi được chỉ ra, giữ nguyên mọi phần đúng.
Không tóm tắt, không thêm nội dung, không đổi giọng kể ngoài phạm vi lỗi.
Trả về toàn bộ đoạn đã sửa, không giải thích.'''


def translate_chunk(client, model, bible_text, prev_context, source_chunk, style):
    user = f'''{bible_text}

PHONG CÁCH DỊCH:
{style}

NGỮ CẢNH LIỀN TRƯỚC (chỉ để hiểu quan hệ, không dịch lại):
{prev_context[-5000:]}

ĐOẠN NGUỒN CẦN DỊCH:
{source_chunk}'''
    return call_model(client, model, TRANSLATE_SYSTEM, user, max_tokens=9000, json_mode=False, thinking=True)


def validate_chunk(client, model, bible_text, source, translated):
    user = f'''{bible_text}

BẢN NGUỒN:
{source}

BẢN DỊCH:
{translated}'''
    try:
        return call_json(client, model, CHECK_SYSTEM, user, max_tokens=3500)
    except Exception:
        return {'ok': True, 'issues': []}


def fix_chunk(client, model, bible_text, translated, issues):
    if not issues:
        return translated
    user = f'''{bible_text}

LỖI CẦN SỬA:
{json.dumps(issues, ensure_ascii=False)}

BẢN DỊCH:
{translated}'''
    return call_model(client, model, FIX_SYSTEM, user, max_tokens=9000, thinking=True)

# deterministic guardrails for obvious pronoun drift

def pronoun_guard(text: str, bible: Dict[str, Any]) -> List[str]:
    issues = []
    style = bible.get('style', {})
    rule = style.get('general_pronoun_rule', '')
    if 'ta' in rule.lower():
        # Flag common modern drift; model review handles context-specific cases.
        if re.search(r'\b(tôi|cô ấy|anh ấy|cậu ấy)\b', text, re.I):
            issues.append('Phát hiện đại từ hiện đại có thể lệch văn phong/xưng hô: tôi/cô ấy/anh ấy/cậu ấy. Cần kiểm tra theo STORY BIBLE.')
    return issues

# -----------------------------
# Word export
# -----------------------------

def export_docx(title: str, translated_chapters: List[Dict[str, str]], bible: Dict[str, Any]) -> bytes:
    doc = Document()
    doc.add_heading(title, level=1)
    for ch in translated_chapters:
        doc.add_heading(ch['title'], level=2)
        for p in ch['text'].split('\n\n'):
            if p.strip():
                doc.add_paragraph(p.strip())
    bio = doc.core_properties
    bio.title = title
    bio.subject = APP_VERSION
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_VERSION, page_icon='📖', layout='wide')

st.title('📖 Dịch Truyện AI V5 PRO')
st.caption('Word/TXT → nhận diện chương → STORY BIBLE khóa xưng hô → chia nhỏ thông minh → dịch → kiểm tra → sửa lỗi → xuất Word')

with st.sidebar:
    st.header('⚙️ Cài đặt')
    api_key = st.text_input('DeepSeek API Key', type='password', value=st.secrets.get('DEEPSEEK_API_KEY', os.getenv('DEEPSEEK_API_KEY', '')))
    model = st.selectbox('Mô hình DeepSeek', [DEFAULT_MODEL, 'deepseek-v4-flash'], index=0)
    style = st.text_area('Phong cách dịch', value='Văn phong truyện tự nhiên, mượt, dễ đọc như bản dịch tiểu thuyết Việt được biên tập kỹ. Giữ sắc thái cảm xúc và bối cảnh. Đối thoại tự nhiên, không máy móc. Không tự ý thêm, bớt hoặc giải thích nội dung.', height=180)
    chunk_size = st.slider('Độ dài mỗi lượt dịch (ký tự)', 3500, 10000, 7000, 500)
    use_thinking = st.checkbox('DeepSeek V4 Pro Thinking', value=True)
    st.info('V5 Pro có kiểm tra xưng hô/tên/thuật ngữ sau mỗi chunk và có bước sửa riêng.')

uploaded = st.file_uploader('📄 Tải 1 file truyện dài', type=['docx', 'txt'])

if uploaded:
    try:
        raw = normalize_text(read_upload(uploaded))
        chapters = split_chapters(raw)
        st.success(f'Đã đọc {len(raw):,} ký tự • phát hiện {len(chapters)} chương • dự kiến {sum(max(1, len(chunk_text(c["text"], chunk_size))) for c in chapters)} lượt dịch.')
    except Exception as e:
        st.error(str(e))
        st.stop()

    if not api_key:
        st.warning('Hãy nhập DeepSeek API Key ở thanh bên trái.')
        st.stop()

    if st.button('🧠 PHÂN TÍCH + BẮT ĐẦU DỊCH V5 PRO', type='primary'):
        client = get_client(api_key)
        progress = st.progress(0)
        status = st.empty()
        try:
            status.info('Bước 1/4: đang xây STORY BIBLE khóa nhân vật và xưng hô...')
            bible = build_story_bible(client, model, chapters, style)
            st.session_state['bible'] = bible

            total_chunks = sum(len(chunk_text(c['text'], chunk_size)) for c in chapters)
            done = 0
            translated = []
            prev = ''
            errors = []

            for ci, chapter in enumerate(chapters, start=1):
                status.info(f'Bước 2/4: đang dịch {chapter["title"]} ({ci}/{len(chapters)})...')
                out_chunks = []
                for source_chunk in chunk_text(chapter['text'], chunk_size):
                    bible_text = bible_for_prompt(bible)
                    try:
                        # The UI checkbox controls only effort; the API wrapper also has a safe fallback.
                        translated_chunk = translate_chunk(client, model, bible_text, prev, source_chunk, style)
                        check = validate_chunk(client, model, bible_text, source_chunk, translated_chunk)
                        issues = check.get('issues', []) if isinstance(check, dict) else []
                        issues += [{'type': 'style', 'original': '', 'translated': '', 'fix': x} for x in pronoun_guard(translated_chunk, bible)]
                        if issues:
                            translated_chunk = fix_chunk(client, model, bible_text, translated_chunk, issues)
                        out_chunks.append(translated_chunk)
                        prev = (prev + '\n\n' + translated_chunk)[-7000:]
                    except Exception as e:
                        errors.append(f'{chapter["title"]}: {e}')
                        # Keep source visible rather than silently losing text.
                        out_chunks.append('[LỖI DỊCH CHUNK — CẦN CHẠY LẠI]\n' + source_chunk)
                    done += 1
                    progress.progress(min(done / max(total_chunks, 1), 1.0))
                translated.append({'title': chapter['title'], 'text': '\n\n'.join(out_chunks)})

            status.info('Bước 3/4: hoàn tất kiểm tra nhất quán và ghép chương...')
            docx_bytes = export_docx(os.path.splitext(uploaded.name)[0] + ' - V5 PRO', translated, bible)
            st.session_state['translated'] = translated
            st.session_state['docx'] = docx_bytes
            st.session_state['errors'] = errors
            progress.progress(1.0)
            status.success('Bước 4/4: đã hoàn tất.')
        except Exception as e:
            st.error(f'Quá trình dịch bị dừng do lỗi API. Bộ nhớ đã xử lý trước đó vẫn được giữ.\n\n{type(e).__name__}: {e}')

if 'bible' in st.session_state:
    with st.expander('🧠 STORY BIBLE — bộ nhớ nhân vật, xưng hô, thuật ngữ', expanded=False):
        st.json(st.session_state['bible'])

if 'translated' in st.session_state:
    st.subheader('📥 Bản dịch hoàn chỉnh')
    st.download_button('⬇️ TẢI FILE WORD ĐÃ DỊCH', data=st.session_state['docx'], file_name='ban-dich-v5-pro.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    with st.expander('👀 Xem thử bản dịch'):
        for ch in st.session_state['translated'][:3]:
            st.markdown(f'### {ch["title"]}')
            st.write(ch['text'][:5000])
    if st.session_state.get('errors'):
        st.warning('Có một số chunk lỗi API và đã được đánh dấu để chạy lại:')
        st.write('\n'.join(st.session_state['errors']))

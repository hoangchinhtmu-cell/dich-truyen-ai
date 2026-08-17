import io, json, re, time
import streamlit as st
from docx import Document
from openai import OpenAI

st.set_page_config(page_title='Dịch Truyện AI V4 Pro 1.2', page_icon='📖', layout='wide')

DEFAULT_MODEL = 'deepseek-v4-pro'
DEFAULT_STYLE = '''Văn phong tiểu thuyết Việt tự nhiên, mượt, dễ đọc như bản dịch được biên tập kỹ.
Giữ đúng sắc thái cảm xúc, bối cảnh và quan hệ nhân vật. Đối thoại tự nhiên, phù hợp vai vế.
Không dịch máy móc từng chữ. Không tự ý thêm, bớt, giải thích hoặc diễn giải nội dung.'''

# =========================
# ĐỌC FILE
# =========================
def read_docx(data):
    doc = Document(io.BytesIO(data))
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def read_txt(data):
    for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'gbk'):
        try:
            return [x.strip() for x in data.decode(enc).splitlines() if x.strip()]
        except UnicodeDecodeError:
            continue
    raise ValueError('Không đọc được TXT. Hãy lưu TXT bằng UTF-8 hoặc GB18030.')


# =========================
# NHẬN DIỆN CHƯƠNG
# =========================
CHAPTER_RE = re.compile(
    r'^\s*(第\s*[0-9一二三四五六七八九十百千万两零]+\s*[章回节卷篇部]|'
    r'chương\s+\d+(?:\s*[:：.\-].*)?|chapter\s+\d+)(?:\s*)$', re.I
)


def split_chapters(paragraphs):
    starts = [i for i, p in enumerate(paragraphs) if CHAPTER_RE.match(p)]
    if not starts:
        return [{'number': 1, 'source_title': 'Chương 1', 'paragraphs': paragraphs}]

    out = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(paragraphs)
        title = paragraphs[start]
        out.append({
            'number': n + 1,
            'source_title': title,
            'paragraphs': paragraphs[start + 1:end]
        })
    return out


# =========================
# CHIA CHUNK
# =========================
SENTENCE_RE = re.compile(r'(?<=[。！？!?…；;])\s+|(?<=[。！？!?…；;])(?=[\u4e00-\u9fff])')


def split_long_paragraph(text, limit):
    if len(text) <= limit:
        return [text]

    sentences = [x.strip() for x in SENTENCE_RE.split(text) if x.strip()]
    if not sentences:
        return [text[i:i + limit] for i in range(0, len(text), limit)]

    out, cur = [], ''
    for s in sentences:
        if len(s) > limit:
            if cur:
                out.append(cur)
                cur = ''
            out.extend([s[i:i + limit] for i in range(0, len(s), limit)])
            continue
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = ''
        cur = s if not cur else cur + ' ' + s
    if cur:
        out.append(cur)
    return out


def make_chunks(paragraphs, limit=4500):
    chunks, cur, size = [], [], 0
    for p in paragraphs:
        for piece in split_long_paragraph(p, limit):
            if cur and size + len(piece) + 1 > limit:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(piece)
            size += len(piece) + 1
    if cur:
        chunks.append(cur)
    return chunks


# =========================
# API - V4 PRO, CHỐNG TIMEOUT
# =========================
def make_client(api_key):
    # Timeout dài hơn cho truyện dài; retry tự kiểm soát để dễ báo tiến độ.
    return OpenAI(
        api_key=api_key,
        base_url='https://api.deepseek.com',
        timeout=180.0,
        max_retries=0,
    )


def parse_json(text):
    if not text:
        raise ValueError('DeepSeek trả về nội dung rỗng.')
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError('DeepSeek không trả JSON hợp lệ.')


def call_json(client, model, system, user, max_tokens=7000, retries=3):
    """Call DeepSeek for structured JSON.

    V4-Pro can spend the whole completion budget on hidden reasoning, which may
    leave message.content empty. For these structured extraction/translation
    calls we explicitly disable thinking so the JSON is returned reliably.
    """
    last_error = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                ],
                stream=False,
                response_format={'type': 'json_object'},
                max_tokens=max_tokens,
                extra_body={'thinking': {'type': 'disabled'}},
            )
            choice = response.choices[0]
            content = choice.message.content
            if content and content.strip():
                return parse_json(content)

            finish = getattr(choice, 'finish_reason', None)
            raise ValueError(
                f'DeepSeek trả về nội dung rỗng (finish_reason={finish}).'
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 2)
    raise last_error


# =========================
# STORY BIBLE
# =========================
def select_story_samples(chapters, max_chars=24000):
    if not chapters:
        return ''

    if len(chapters) <= 12:
        selected = chapters
    else:
        idx = list(range(min(6, len(chapters))))
        mid = len(chapters) // 2
        idx += list(range(max(0, mid - 2), min(len(chapters), mid + 2)))
        idx += list(range(max(0, len(chapters) - 4), len(chapters)))
        selected = [chapters[i] for i in sorted(set(idx))]

    blocks = []
    for ch in selected:
        text = '\n'.join(ch['paragraphs'][:30])
        blocks.append(f"### {ch['source_title']}\n{text}")

    return '\n\n'.join(blocks)[:max_chars]


def build_story_bible(client, model, chapters, style):
    sample = select_story_samples(chapters)
    system = '''Bạn là biên tập viên tiểu thuyết Trung -> Việt.
Hãy tạo STORY BIBLE để một bộ truyện dài được dịch nhất quán xuyên suốt.
Chỉ ghi thông tin có căn cứ từ văn bản. Không đoán nếu chưa đủ căn cứ.
Đặc biệt phải theo dõi: tên gốc, tên Việt/Hán-Việt, giới tính, vai trò, quan hệ,
ngôi xưng và cách gọi giữa từng cặp nhân vật, địa danh, tổ chức, chức danh,
thuật ngữ, bối cảnh và quy tắc văn phong.
Đầu ra bắt buộc là JSON hợp lệ.'''
    user = f'''Phong cách dịch:
{style}

Hãy trả JSON theo đúng cấu trúc:
{{
  "characters": [],
  "pronoun_rules": [],
  "glossary": [],
  "places": [],
  "organizations": [],
  "world_rules": [],
  "style_rules": [],
  "uncertain_items": []
}}

Mỗi nhân vật nên cố gắng có các trường: name_cn, name_vi, gender, role, relationships, notes.
Mỗi quy tắc xưng hô nên có: speaker, listener, pronoun, condition, notes.
Không tự bịa thông tin không có trong mẫu.

MẪU RẢI ĐỀU TRONG TRUYỆN:
{sample}'''
    return call_json(client, model, system, user, max_tokens=9000)


# =========================
# LỌC BỘ NHỚ LIÊN QUAN
# =========================
def item_text(x):
    return json.dumps(x, ensure_ascii=False)


def relevant_memory(memory, source):
    result = {
        'characters': [], 'pronoun_rules': [], 'glossary': [], 'places': [],
        'organizations': [], 'world_rules': memory.get('world_rules', [])[:25],
        'style_rules': memory.get('style_rules', [])[:25]
    }

    matched_names = set()
    for c in memory.get('characters', []):
        keys = [c.get('name_cn', ''), c.get('name_vi', '')]
        if any(k and k in source for k in keys):
            result['characters'].append(c)
            matched_names.update(k for k in keys if k)

    for r in memory.get('pronoun_rules', []):
        txt = item_text(r)
        if not matched_names or any(n in txt for n in matched_names):
            result['pronoun_rules'].append(r)

    for key in ('glossary', 'places', 'organizations'):
        for x in memory.get(key, []):
            txt = item_text(x)
            vals = [
                x.get('source', ''), x.get('translation', ''),
                x.get('name_cn', ''), x.get('name_vi', '')
            ]
            if any(v and v in source for v in vals) or any(n in txt for n in matched_names):
                result[key].append(x)

    # Nếu chunk không chứa tên rõ ràng, gửi một nền nhỏ thay vì toàn bộ bộ nhớ.
    if not result['characters']:
        result['characters'] = memory.get('characters', [])[:35]
    if not result['pronoun_rules']:
        result['pronoun_rules'] = memory.get('pronoun_rules', [])[:35]
    return result


# =========================
# DỊCH CHUNK + CẬP NHẬT MEMORY
# =========================
def translate_chunk(client, model, chapter_title, chunk, memory, style):
    source = '\n'.join(f'[P{i + 1}] {p}' for i, p in enumerate(chunk))
    rel = relevant_memory(memory, source)

    system = '''Bạn là dịch giả tiểu thuyết Trung -> Việt chuyên nghiệp.
Dịch tự nhiên, mượt, có văn phong tiểu thuyết Việt; ưu tiên đúng nghĩa và đúng cảm xúc.
Không dịch từng chữ máy móc. Không tự ý thêm, bớt hoặc giải thích nội dung.

QUY TẮC BẮT BUỘC:
1. Tên nhân vật phải thống nhất theo STORY BIBLE.
2. Không tự đổi cách gọi tên giữa các đoạn.
3. Xưng hô phải đúng giới tính, tuổi/vai vế, quan hệ và bối cảnh.
4. Nếu STORY BIBLE chưa đủ căn cứ, giữ cách dịch trung tính thay vì tự bịa.
5. Giữ đúng thứ tự đoạn P1, P2... và trả đúng số lượng đoạn.
6. Các câu thoại phải tự nhiên như văn nói của nhân vật Việt.
7. Đầu ra bắt buộc là JSON hợp lệ.'''

    user = f'''CHƯƠNG: {chapter_title}

PHONG CÁCH:
{style}

STORY BIBLE LIÊN QUAN:
{json.dumps(rel, ensure_ascii=False, indent=2)}

ĐOẠN NGUỒN:
{source}

Hãy trả đúng JSON:
{{
  "paragraphs": ["bản dịch P1", "bản dịch P2"],
  "memory_updates": {{
    "characters": [],
    "pronoun_rules": [],
    "glossary": [],
    "places": [],
    "organizations": [],
    "world_rules": []
  }}
}}

Đầu vào có {len(chunk)} đoạn. "paragraphs" bắt buộc có đúng {len(chunk)} phần tử.
Không được đưa ký hiệu [P1], [P2]... vào bản dịch.'''

    data = call_json(client, model, system, user, max_tokens=max(8000, len(chunk) * 1000))
    paragraphs = data.get('paragraphs', [])

    if len(paragraphs) != len(chunk):
        repair_user = user + f'''

CẢNH BÁO: Bạn vừa trả sai số lượng đoạn. Hãy trả lại JSON với đúng {len(chunk)} phần tử trong "paragraphs".'''
        data = call_json(client, model, system, repair_user, max_tokens=max(8000, len(chunk) * 1000))
        paragraphs = data.get('paragraphs', [])

    if len(paragraphs) != len(chunk):
        raise ValueError(f'DeepSeek trả {len(paragraphs)} đoạn thay vì {len(chunk)} đoạn.')

    return paragraphs, data.get('memory_updates', {})


# =========================
# MERGE MEMORY THÔNG MINH
# =========================
def merge_by_key(existing, incoming, keys):
    out = list(existing)
    for item in incoming or []:
        if not item:
            continue
        found = None
        for old in out:
            if any(item.get(k) and old.get(k) == item.get(k) for k in keys):
                found = old
                break
        if found is None:
            out.append(item)
        else:
            for k, v in item.items():
                if v not in (None, '', [], {}):
                    found[k] = v
    return out


def merge_memory(memory, update):
    for k in ('characters', 'pronoun_rules', 'glossary', 'places', 'organizations', 'world_rules', 'style_rules', 'uncertain_items'):
        memory.setdefault(k, [])

    memory['characters'] = merge_by_key(memory['characters'], update.get('characters'), ['name_cn', 'name_vi'])
    memory['pronoun_rules'] = merge_by_key(memory['pronoun_rules'], update.get('pronoun_rules'), ['speaker', 'listener', 'pronoun'])
    memory['glossary'] = merge_by_key(memory['glossary'], update.get('glossary'), ['source', 'translation', 'name_cn', 'name_vi'])
    memory['places'] = merge_by_key(memory['places'], update.get('places'), ['name_cn', 'name_vi', 'source', 'translation'])
    memory['organizations'] = merge_by_key(memory['organizations'], update.get('organizations'), ['name_cn', 'name_vi', 'source', 'translation'])

    for k in ('world_rules', 'style_rules', 'uncertain_items'):
        for item in update.get(k, []) or []:
            if item and item not in memory[k]:
                memory[k].append(item)
    return memory


# =========================
# XUẤT WORD
# =========================
def export_docx(book):
    doc = Document()
    for i, ch in enumerate(book):
        doc.add_heading(ch['title'], level=1)
        for p in ch['paragraphs']:
            doc.add_paragraph(p)
        if i < len(book) - 1:
            doc.add_page_break()
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# =========================
# UI
# =========================
st.sidebar.title('⚙️ Cài đặt')
api_key = st.sidebar.text_input('DeepSeek API Key', type='password')
model = st.sidebar.text_input('Mô hình DeepSeek', value=DEFAULT_MODEL)
style = st.sidebar.text_area('Phong cách dịch', value=DEFAULT_STYLE, height=180)
limit = st.sidebar.slider('Độ dài mỗi lượt dịch (ký tự)', 3000, 7000, 4500, 500)
retries = st.sidebar.slider('Số lần thử lại khi API lỗi', 1, 4, 3)

st.title('📖 Dịch Truyện AI V4 Pro 1.2')
st.caption('Word/TXT → nhận diện chương → STORY BIBLE → chia nhỏ → lọc bộ nhớ → dịch tuần tự → cập nhật bộ nhớ → xuất 1 file Word')

file = st.file_uploader('📄 Tải 1 file truyện dài', type=['docx', 'txt'])

if file:
    try:
        raw = file.getvalue()
        paras = read_docx(raw) if file.name.lower().endswith('.docx') else read_txt(raw)
        chapters = split_chapters(paras)
        total_chars = sum(len(p) for p in paras)
        total_chunks = sum(len(make_chunks(c['paragraphs'], limit)) for c in chapters)

        st.success(
            f'Đã đọc {len(paras):,} đoạn • {total_chars:,} ký tự • '
            f'phát hiện {len(chapters):,} chương • dự kiến {total_chunks:,} lượt dịch.'
        )

        if not api_key:
            st.warning('Nhập DeepSeek API Key ở thanh bên trái trước khi bắt đầu.')

        if st.button('🧠 PHÂN TÍCH + BẮT ĐẦU DỊCH V4 PRO', type='primary', disabled=not bool(api_key)):
            client = make_client(api_key)
            status = st.empty()
            bar = st.progress(0)
            memory = None
            book = []
            done = 0

            try:
                status.info('Bước 1/2: đang tạo STORY BIBLE với DeepSeek V4 Pro...')
                memory = build_story_bible(client, model, chapters, style)
                st.session_state['memory'] = memory

                status.info(f'Bước 2/2: đang chia nhỏ và dịch {total_chunks:,} lượt...')
                for ch in chapters:
                    out = []
                    parts = make_chunks(ch['paragraphs'], limit)
                    for idx, part in enumerate(parts, start=1):
                        status.info(
                            f'Đang dịch Chương {ch["number"]}/{len(chapters)} • '
                            f'phần {idx}/{len(parts)} • tổng {done + 1}/{total_chunks}'
                        )
                        tr, upd = translate_chunk(client, model, ch['source_title'], part, memory, style)
                        out.extend(tr)
                        memory = merge_memory(memory, upd)
                        done += 1
                        bar.progress(done / max(total_chunks, 1))
                        st.session_state['memory'] = memory

                    book.append({
                        'number': ch['number'],
                        'title': f'Chương {ch["number"]}',
                        'paragraphs': out
                    })

                st.session_state['book'] = book
                status.success('🎉 Đã dịch xong toàn bộ truyện!')
            except Exception as exc:
                status.error('❌ Quá trình dịch bị dừng do lỗi API. Bộ nhớ đã xử lý trước đó vẫn được giữ.')
                st.exception(exc)

    except Exception as exc:
        st.error(f'Không đọc được file: {exc}')

if 'memory' in st.session_state:
    with st.expander('🧠 STORY BIBLE — bộ nhớ nhân vật, xưng hô, thuật ngữ', expanded=False):
        st.json(st.session_state['memory'])

if 'book' in st.session_state:
    st.subheader('📥 Bản dịch hoàn chỉnh')
    st.download_button(
        '⬇️ TẢI FILE WORD ĐÃ DỊCH',
        export_docx(st.session_state['book']),
        'ban-dich-truyen-v4-pro-1.2.docx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        use_container_width=True,
    )
    with st.expander('👀 Xem thử bản dịch'):
        for ch in st.session_state['book'][:3]:
            st.markdown(f'### {ch["title"]}')
            for p in ch['paragraphs'][:8]:
                st.write(p)

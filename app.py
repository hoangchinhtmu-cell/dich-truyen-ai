import io, json, re, time
from pathlib import Path
import streamlit as st
from docx import Document
from openai import OpenAI

st.set_page_config(page_title='Dịch Truyện AI V2', page_icon='📖', layout='wide')

DEFAULT_STYLE = '''Văn phong truyện tự nhiên, mượt, dễ đọc như bản dịch tiểu thuyết Việt được biên tập kỹ. Giữ sắc thái cảm xúc và bối cảnh. Đối thoại tự nhiên, không máy móc. Không tự ý thêm, bớt hoặc giải thích nội dung.'''
DEFAULT_MODEL = 'deepseek-v4-pro'

# ---------- đọc file ----------
def read_docx(data):
    doc = Document(io.BytesIO(data))
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]

def read_txt(data):
    for enc in ('utf-8-sig','utf-8','gb18030','gbk'):
        try:
            return [x.strip() for x in data.decode(enc).splitlines() if x.strip()]
        except UnicodeDecodeError:
            pass
    raise ValueError('Không đọc được TXT. Hãy lưu TXT bằng UTF-8 hoặc GB18030.')

# ---------- nhận diện chương ----------
CHAPTER_RE = re.compile(r'^\s*(第\s*[0-9一二三四五六七八九十百千万]+\s*[章回节卷篇部]|chương\s+\d+(?:\s*[:：.\-].*)?|chapter\s+\d+)(?:\s*)$', re.I)

def split_chapters(paragraphs):
    starts = [i for i,p in enumerate(paragraphs) if CHAPTER_RE.match(p)]
    if not starts:
        return [{'number':1,'source_title':'Chương 1','paragraphs':paragraphs}]
    out=[]
    for n,start in enumerate(starts):
        end = starts[n+1] if n+1 < len(starts) else len(paragraphs)
        out.append({'number':n+1,'source_title':paragraphs[start], 'paragraphs':paragraphs[start+1:end]})
    return out

# ---------- chia thông minh: ưu tiên đoạn, rồi câu ----------
SENTENCE_RE = re.compile(r'(?<=[。！？!?…；;])\s+|(?<=[。！？!?…；;])(?=[\u4e00-\u9fff])')

def split_long_paragraph(p, limit):
    if len(p) <= limit:
        return [p]
    sentences = [x.strip() for x in SENTENCE_RE.split(p) if x.strip()]
    if not sentences:
        return [p[i:i+limit] for i in range(0,len(p),limit)]
    out=[]; cur=''
    for s in sentences:
        if cur and len(cur)+1+len(s) > limit:
            out.append(cur); cur=''
        cur = s if not cur else cur+' '+s
    if cur: out.append(cur)
    return out

def make_chunks(paragraphs, limit=7000):
    chunks=[]; cur=[]; size=0
    for p in paragraphs:
        pieces = split_long_paragraph(p, limit)
        for piece in pieces:
            if cur and size+len(piece)+1 > limit:
                chunks.append(cur); cur=[]; size=0
            cur.append(piece); size += len(piece)+1
    if cur: chunks.append(cur)
    return chunks

# ---------- API ----------
def ask_json(client, model, system, user, thinking=False):
    kwargs = dict(model=model, messages=[{'role':'system','content':system},{'role':'user','content':user}], stream=False, response_format={'type':'json_object'})
    if thinking:
        kwargs['reasoning_effort']='high'
    for attempt in range(3):
        try:
            r=client.chat.completions.create(**kwargs)
            text=r.choices[0].message.content
            try: return json.loads(text)
            except Exception:
                m=re.search(r'\{.*\}', text, re.S)
                if m: return json.loads(m.group())
                raise ValueError('DeepSeek không trả JSON hợp lệ.')
        except Exception:
            if attempt==2: raise
            time.sleep(2*(attempt+1))

def build_story_bible(client, model, chapters, style, sample_chars=90000):
    # Phân tích nhiều mẫu rải đều thay vì chỉ đọc 12 chương đầu.
    if len(chapters) <= 20:
        selected=chapters
    else:
        idxs=list(range(10))+list(range(max(10,len(chapters)//2-5), min(len(chapters),len(chapters)//2+5)))+list(range(max(0,len(chapters)-10),len(chapters)))
        selected=[chapters[i] for i in sorted(set(idxs))]
    blocks=[]
    for ch in selected:
        blocks.append(f"### {ch['source_title']}\n"+'\n'.join(ch['paragraphs'][:45]))
    text='\n\n'.join(blocks)[:sample_chars]
    system='''Bạn là biên tập viên truyện Trung -> Việt. Hãy xây STORY BIBLE để dịch cả tiểu thuyết nhất quán. Không dịch ở bước này. Chỉ ghi thông tin có căn cứ, không bịa. Đặc biệt theo dõi tên nhân vật, tên Việt/Hán-Việt, giới tính, vai trò, quan hệ, tuổi/vai vế, xưng hô giữa từng cặp, địa danh, tổ chức, chức danh, thuật ngữ và quy tắc văn phong.'''
    user=f'''Phong cách: {style}\n\nTrả JSON đúng cấu trúc:\n{{"characters":[],"pronoun_rules":[],"glossary":[],"places":[],"organizations":[],"world_rules":[],"style_rules":[],"uncertain_items":[]}}\n\nTRÍCH MẪU RẢI ĐỀU TRONG TRUYỆN:\n{text}'''
    return ask_json(client,model,system,user,thinking=True)

def relevant_memory(memory, source):
    # Lọc các mục có tên/thuật ngữ xuất hiện trong chunk. Nếu không tìm thấy gì, gửi một phần nền.
    blob=source
    result={'characters':[],'pronoun_rules':[],'glossary':[],'places':[],'organizations':[],'world_rules':memory.get('world_rules',[])[:20],'style_rules':memory.get('style_rules',[])[:20]}
    names=[]
    for c in memory.get('characters',[]):
        keys=[c.get('name_cn',''),c.get('name_vi','')]
        if any(k and k in blob for k in keys):
            result['characters'].append(c); names.extend([k for k in keys if k])
    for r in memory.get('pronoun_rules',[]):
        txt=json.dumps(r,ensure_ascii=False)
        if any(n in txt for n in names): result['pronoun_rules'].append(r)
    for k in ('glossary','places','organizations'):
        for x in memory.get(k,[]):
            txt=json.dumps(x,ensure_ascii=False)
            if any(v and v in blob for v in [x.get('source',''),x.get('translation',''),x.get('name_cn',''),x.get('name_vi','')]): result[k].append(x)
    if not result['characters']:
        result['characters']=memory.get('characters',[])[:30]
    if not result['pronoun_rules']:
        result['pronoun_rules']=memory.get('pronoun_rules',[])[:30]
    return result

def translate_chunk(client, model, chapter_title, chunk, memory, style):
    source='\n'.join(f'[P{i+1}] {p}' for i,p in enumerate(chunk))
    rel=relevant_memory(memory, source)
    system='''Bạn là dịch giả tiểu thuyết Trung -> Việt chuyên nghiệp. Dịch tự nhiên, mượt, đúng sắc thái và bối cảnh. Không dịch từng chữ máy móc. Không tự ý thêm/bớt nội dung. Tên nhân vật, giới tính, quan hệ và xưng hô phải tuân theo STORY BIBLE. Giữ đúng thứ tự và số đoạn [P1], [P2]... Trả JSON duy nhất: {"paragraphs":["..."],"memory_updates":{"characters":[],"pronoun_rules":[],"glossary":[],"places":[],"organizations":[],"world_rules":[]}}'''
    user=f'''CHƯƠNG: {chapter_title}\nPHONG CÁCH:\n{style}\n\nSTORY BIBLE LIÊN QUAN:\n{json.dumps(rel,ensure_ascii=False,indent=2)}\n\nĐOẠN NGUỒN:\n{source}\n\nYÊU CẦU: Đầu vào có {len(chunk)} đoạn; đầu ra paragraphs phải có đúng {len(chunk)} phần tử. Không đưa [P] vào bản dịch.'''
    d=ask_json(client,model,system,user,thinking=False)
    if len(d.get('paragraphs',[])) != len(chunk):
        user += f'\nCẢNH BÁO: hãy sửa số lượng phần tử thành đúng {len(chunk)}.'
        d=ask_json(client,model,system,user,thinking=False)
    return d.get('paragraphs',[]), d.get('memory_updates',{})

def merge_memory(mem, upd):
    for k in ('characters','pronoun_rules','glossary','places','organizations','world_rules','style_rules'): mem.setdefault(k,[])
    for k in ('characters','pronoun_rules','glossary','places','organizations'):
        for x in upd.get(k,[]):
            if x and x not in mem[k]: mem[k].append(x)
    for k in ('world_rules','style_rules'):
        for x in upd.get(k,[]):
            if x and x not in mem[k]: mem[k].append(x)
    return mem

def export_docx(book):
    doc=Document()
    for i,ch in enumerate(book):
        doc.add_heading(ch['title'], level=1)
        for p in ch['paragraphs']: doc.add_paragraph(p)
        if i < len(book)-1: doc.add_page_break()
    b=io.BytesIO(); doc.save(b); return b.getvalue()

# ---------- UI ----------
st.sidebar.title('⚙️ Cài đặt')
api_key=st.sidebar.text_input('DeepSeek API Key', type='password')
model=st.sidebar.text_input('Mô hình DeepSeek', value=DEFAULT_MODEL)
style=st.sidebar.text_area('Phong cách dịch', value=DEFAULT_STYLE, height=170)
limit=st.sidebar.slider('Độ dài mỗi lượt dịch (ký tự)', 4000, 10000, 7000, 500)

st.title('📖 Dịch Truyện AI V2')
st.caption('Word/TXT → nhận diện chương → STORY BIBLE → chia nhỏ thông minh → lọc bộ nhớ liên quan → dịch → cập nhật bộ nhớ → xuất Word')

file=st.file_uploader('📄 Tải 1 file truyện dài', type=['docx','txt'])
if file:
    raw=file.getvalue()
    paras=read_docx(raw) if file.name.lower().endswith('.docx') else read_txt(raw)
    chapters=split_chapters(paras)
    total_chars=sum(len(p) for p in paras)
    total_chunks=sum(len(make_chunks(c['paragraphs'],limit)) for c in chapters)
    st.success(f'Đã đọc {len(paras):,} đoạn • {total_chars:,} ký tự • phát hiện {len(chapters):,} chương • dự kiến {total_chunks:,} lượt dịch.')
    if not api_key: st.warning('Nhập DeepSeek API Key ở thanh bên trái trước khi bắt đầu.')
    if st.button('🧠 PHÂN TÍCH + BẮT ĐẦU DỊCH', type='primary', disabled=not bool(api_key)):
        client=OpenAI(api_key=api_key, base_url='https://api.deepseek.com')
        status=st.empty(); bar=st.progress(0)
        status.info('Bước 1/2: đang tạo STORY BIBLE...')
        memory=build_story_bible(client,model,chapters,style)
        st.session_state['memory']=memory
        status.info('Bước 2/2: đang chia nhỏ và dịch từng phần...')
        book=[]; done=0
        for ch in chapters:
            out=[]
            for part in make_chunks(ch['paragraphs'],limit):
                tr,upd=translate_chunk(client,model,ch['source_title'],part,memory,style)
                out.extend(tr); memory=merge_memory(memory,upd); done+=1
                bar.progress(done/max(total_chunks,1)); status.info(f'Đang dịch Chương {ch["number"]}/{len(chapters)} • phần {done}/{total_chunks}')
            book.append({'number':ch['number'],'title':f'Chương {ch["number"]}','paragraphs':out})
        st.session_state['memory']=memory; st.session_state['book']=book
        status.success('🎉 Đã dịch xong toàn bộ truyện!')

if 'memory' in st.session_state:
    with st.expander('🧠 STORY BIBLE — bộ nhớ truyện', expanded=False): st.json(st.session_state['memory'])
if 'book' in st.session_state:
    st.subheader('📥 Tải bản dịch')
    st.download_button('⬇️ TẢI FILE WORD ĐÃ DỊCH', export_docx(st.session_state['book']), 'ban-dich-truyen.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', use_container_width=True)
    with st.expander('👀 Xem thử 3 chương đầu'):
        for ch in st.session_state['book'][:3]:
            st.markdown(f'### {ch["title"]}')
            for p in ch['paragraphs'][:8]: st.write(p)

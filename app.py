import streamlit as st
import torch
import sys
import os

# Чтобы импорт classes.py работал независимо от рабочей директории
sys.path.insert(0, os.path.dirname(__file__))

from classes import (
    AttentionEncoder,
    AttentionDecoder,
    AttentionSeq2Seq,
    attention_greedy_decode, Vocabulary
)

# ── Конфигурация страницы ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EN → RU Translator",
    page_icon="🌐",
    layout="wide",
)

# ── Стили ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Фон */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Заголовок */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Карточки ввода/вывода */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
        height: 100%;
    }
    .card-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }

    /* Результат перевода с границами */
    .translation-block {
        background: rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(167, 139, 250, 0.3);
        border-radius: 12px;
        padding: 1rem;
        min-height: 150px;
        transition: all 0.3s ease;
        display: flex;
        align-items: flex-start;
    }
    .translation-block:hover {
        border-color: rgba(167, 139, 250, 0.6);
        background: rgba(0, 0, 0, 0.4);
    }
    .result-text {
        font-size: 1.15rem;
        color: #e2e8f0;
        word-break: break-word;
        line-height: 1.5;
        margin: 0;
        width: 100%;
    }
    .placeholder-text {
        font-size: 1rem;
        color: #475569;
        font-style: italic;
        margin: 0;
        width: 100%;
    }
    .warning-text {
        font-size: 0.95rem;
        color: #f59e0b;
        margin: 0;
        width: 100%;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Кнопка - центрирование */
    div.stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        cursor: pointer;
        transition: opacity 0.2s ease, transform 0.15s ease;
        min-width: 180px;
    }
    div.stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }

    /* TextArea */
    textarea {
        background: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.05rem !important;
        resize: vertical !important;
    }
    .stTextArea [data-baseweb="textarea"] {
        background: transparent !important;
        border: none !important;
    }

    /* Убираем лишние рамки Streamlit */
    .block-container { padding-top: 2rem; }
    footer { visibility: hidden; }

    /* Отступы для колонок */
    .row-widget.stHorizontal {
        gap: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Загрузка модели (кешируется) ───────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "translator_full.pt")


@st.cache_resource(show_spinner="⏳ Загрузка модели…")
def load_model():
    import __main__
    __main__.Vocabulary = Vocabulary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    src_vocab = checkpoint["src_vocab"]
    tgt_vocab = checkpoint["tgt_vocab"]
    h = checkpoint["hyperparams"]

    encoder = AttentionEncoder(
        src_vocab_size=len(src_vocab.word2idx),
        embedding_dim=h["embedding_dim"],
        hidden_dim=h["hidden_dim"],
        num_layers=h["num_layers"],
        dropout=h["dropout"],
    )
    decoder = AttentionDecoder(
        tgt_vocab_size=len(tgt_vocab.word2idx),
        embedding_dim=h["embedding_dim"],
        hidden_dim=h["hidden_dim"],
        num_layers=h["num_layers"],
        dropout=h["dropout"],
    )
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, src_vocab, tgt_vocab, device


def translate(text: str, model, src_vocab, tgt_vocab, device, max_len: int = 60) -> str:
    tokens = text.strip().split()
    if not tokens:
        return ""
    indexed = src_vocab.encode_tokens(tokens)
    src_tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)
    decoded_tokens = attention_greedy_decode(model, src_tensor, max_len, device, tgt_vocab)
    return tgt_vocab.decode(decoded_tokens, remove_special_tokens=True)


# ── Интерфейс ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🌐 EN → RU Translator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Нейросетевой переводчик с английского на русский язык</p>', unsafe_allow_html=True)

# Загружаем модель
model, src_vocab, tgt_vocab, device = load_model()

# ── ДВЕ КОЛОНКИ: слева ввод, справа вывод ─────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

# Левая колонка - ввод текста
with col_left:
    st.markdown('<div class="card"><div class="card-label">🇬🇧 Английский текст</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        label="english_input",
        placeholder="Введите английский текст для перевода…",
        height=200,
        label_visibility="collapsed",
        key="input_text",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Центрированная кнопка между колонками (под ними)
col_empty1, col_button, col_empty2 = st.columns([1, 2, 1])
with col_button:
    translate_clicked = st.button("🌐 Перевести", key="translate_btn", use_container_width=True)

# Правая колонка - вывод перевода (ВЕСЬ внутри translation-block)
with col_right:
    st.markdown('<div class="card"><div class="card-label">🇷🇺 Перевод</div>', unsafe_allow_html=True)

    # Обработка перевода
    if translate_clicked and input_text.strip():
        with st.spinner("Перевожу..."):
            result = translate(input_text, model, src_vocab, tgt_vocab, device)
        st.session_state["last_result"] = result
        # Формируем HTML c результатом
        translation_html = f'<div class="translation-block"><div class="result-text">{result}</div></div>'
    elif translate_clicked and not input_text.strip():
        # Показываем предупреждение внутри блока
        translation_html = '''
        <div class="translation-block">
            <div class="warning-text">⚠️ Введите текст для перевода.</div>
        </div>
        '''
    elif "last_result" in st.session_state:
        # Показываем последний успешный перевод
        translation_html = f'<div class="translation-block"><div class="result-text">{st.session_state["last_result"]}</div></div>'
    else:
        # Плейсхолдер
        translation_html = '''
        <div class="translation-block">
            <div class="placeholder-text">Перевод появится здесь...</div>
        </div>
        '''

    # Выводим единый блок
    st.markdown(translation_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Нижняя информация
st.markdown(
    f"""
    <div style="text-align:center; margin-top:2rem; color:#475569; font-size:0.8rem;">
        Модель: Seq2Seq + Attention (LSTM) &nbsp;·&nbsp; 
        Устройство: <b style="color:#7c3aed">{'GPU' if torch.cuda.is_available() else 'CPU'}</b>
    </div>
    """,
    unsafe_allow_html=True,
)
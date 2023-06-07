import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Khởi tạo tokenizer và model từ thư mục đã lưu
tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi")

def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        do_sample=True,
        top_k=100,
        top_p=0.8,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

# Giao diện ứng dụng sử dụng Streamlit
st.title("Ứng dụng Dịch tiếng Anh sang tiếng Việt")

# Textbox để nhập đoạn văn tiếng Anh
en_text = st.text_area("Nhập đoạn văn tiếng Anh", "")

# Button để thực hiện dịch
if st.button("Dịch"):
    if en_text:
        # Gọi hàm dịch tiếng Anh sang tiếng Việt
        vi_text = translate_en2vi(en_text)
        # Hiển thị kết quả dịch
        st.text_area("Kết quả dịch tiếng Việt", vi_text)
    else:
        st.warning("Vui lòng nhập đoạn văn tiếng Anh")

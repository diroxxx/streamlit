import time
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "AventIQ-AI/Ai-Translate-Model-Eng-German"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

def translate(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title('Aplikacja do tłumaczenia tekstu z języka angielskiego na język niemiecki')

st.subheader('Instrukcja obsługi aplikacji')
st.text('Wpisz tekst w języku angielskim, który chcesz przetłumaczyć na język niemiecki, a następnie kliknij przycisk "Przetłumacz".')

st.subheader('Uwagi dotyczące modelu tłumaczącego')
st.info('Nie zoptymalizowano pod kątem wyrażeń nieformalnych, idiomatycznych lub slangowych\nNie nadaje się do tłumaczenia treści prawnych, medycznych lub poufnych\nZdania dłuższe niż 128 tokenów są skracane\nDokładność w zależności od domeny może się różnić (np. pod względem prawnym, technicznym)')

text = st.text_area(label="Wpisz tekst po angielsku")
if st.button("Przetłumacz"):
    if text.strip() == "":
        st.error("Proszę wpisać jakiś tekst do przetłumaczenia.")
        raise st.stop()
    if len(text.split()) > 128:
        st.error("Tekst jest zbyt długi. Maksymalna liczba słów to 128.")
        raise st.stop()
    if not any(c.isalpha() for c in text):
        st.error("Tekst musi zawierać przynajmniej jedną literę.")
    
    
    translation = translate(text)
    progress_text = "Tłumacze..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()
    st.success('Gotowe!')
    st.subheader('Przetłumaczony tekst:')
    st.write(translation)

st.write('---')
st.write('s27354')
st.write('---')
# st.subheader('Zadanie do wykonania')
# st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
# st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
# st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
# st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
# st.write('🐞 Na końcu umieść swój numer indeksu')
# st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
# st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

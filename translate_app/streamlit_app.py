from app import decode_sequence, decode_sequence_beam_search
from configs import config
from server import get_english_translation, get_spanish_translation
import streamlit as st

# title
st.title("Spanish-English Translator")
st.markdown(
    """
This is a demo translation app for English-Spanish translation trained using word level Seq2Seq model. Seq2Seq translation model has an Encoder-Decoder architecture. I've written a detailed notebook explaining the trainig procedure [here](https://adimyth.github.io/notes/seq2seq/nlp/machine-translation/2020/06/26/Seq2Seq.html). Code for this app as well as trained model weights can be found in my [github repo](https://github.com/adimyth/datascience_stuff). Please :star: the repo if you like it. You can also find me on [twitter](https://twitter.com/adi_myth)."""
)

# get input text
input_text = st.text_input(label="English text.")

# get beam width if beam search selected
beam_width = None
if st.checkbox("Beam Search"):
    beam_width = st.slider(label="Beam Width", min_value=3, max_value=7, value=3)

# return response
if st.button("Submit"):
    with st.spinner("Translating ..."):
        if beam_width != None:
            spanish_sequences = decode_sequence_beam_search(
                input_text, beam_width=beam_width
            )
            st.subheader("Predicted Spanish Translation")
            for x in spanish_sequences:
                st.success(f"{x[:-4]}")
            spanish_seq = spanish_sequences[0][:-4]
        elif beam_width == None:
            st.subheader("Predicted Spanish Translation")
            spanish_seq = decode_sequence(input_text)[:-4]
            st.success(f"{spanish_seq[:-4]}")
    # actual spanish translation
    st.subheader("Actual Spanish Translation")
    st.success(f"{get_spanish_translation(input_text)}")
    # predicted spanish back to english
    st.subheader("Predicted Spanish to English Translation")
    st.success(f"{get_english_translation(spanish_seq)}")


st.markdown(
    "The interactive app is created using [Streamlit](https://streamlit.io/), an open-source framework that lets users creating apps for machine learning projects very easily."
)

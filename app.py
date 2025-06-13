import streamlit as st
from streamlit_chat import message
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from gtts import gTTS
import os

# Charger le modèle fine-tuné
model_path = "./fine_tuned_gpt2_3"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Configurer la page Streamlit
st.set_page_config(page_title="Chatbot support client", page_icon=":robot_face:")
st.header("Chatbot support client :robot_face:")

# Initialiser le session_state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Fonction pour générer une réponse
def generate_response(user_input):
    # Préparer l'entrée au format souhaité
    input_text = (
        "<Category> Contact </Category>\n"
        f"<Intent> User_Query </Intent>\n"
        f"<Instruction> {user_input} </Instruction>\n"
        "###\nResponse:"
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Génération de la réponse
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Décoder et extraire la réponse
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in response:
        response = response.split("Response:")[-1].strip()
    response = ". ".join(response.split(".")[:1]).strip() + "."
    
    return response

# Fonction pour lire le texte avec gTTS
def speak_text(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    # Utiliser Streamlit pour lire le fichier audio
    audio_file = open(filename, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")
    audio_file.close()
    os.remove(filename)  # Supprimer le fichier après lecture

# Formulaire utilisateur
with st.form(key="form"):
    user_input = st.text_input("You: ", key="input")
    submit_button_pressed = st.form_submit_button("Submit to Bot")

# Gérer l'entrée utilisateur
if submit_button_pressed:
    output = generate_response(user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Afficher les messages
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
        
        # Bouton pour lire la réponse
        if st.button(f"Lire la réponse {i+1}", key=f"read_button_{i}"):
            speak_text(st.session_state["generated"][i])

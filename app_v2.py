import streamlit as st
from streamlit_chat import message
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer        # for gpt-neo
import torch
from gtts import gTTS
import os

# ======================================= Mise en page ==========================================
# Configurer la page Streamlit avec un titre centré
st.set_page_config(page_title="Customer Support", page_icon="🤖")
st.markdown(
    """
    <style>
    /* Centrer le titre */
    h1 {
        text-align: center;
        color: #4CAF50;
        font-size: 48px;
        font-weight: bold;
    }
    
    /* Personnaliser le texte d'accueil */
    #welcome-text {
        text-align: center;
        font-size: 22px;
        margin-bottom: 20px;
        font-style: italic;
    }

    /* Fond animé */
    body {
        background: linear-gradient(135deg, #ece9e6, #ffffff);
        animation: gradient-animation 15s ease infinite;
    }
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Barre de saisie fixée en bas */
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: #ffffff;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .fixed-footer input[type="text"] {
        flex-grow: 1;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 10px 15px;
        font-size: 16px;
        outline: none;
        margin-right: 10px;
    }
    .fixed-footer button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .fixed-footer button:hover {
        background-color: #45a049;
    }
    .fixed-footer button i {
        margin-right: 8px;
    }

    /* Messages */
    .css-1siy2j7 {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Ajouter le titre et le texte d'accueil
st.title("Customer Support 🤖")
st.markdown('<p id="welcome-text">How can I help you today?</p>', unsafe_allow_html=True)

# ============================= Choix du modèle ============================================
# Charger les modèles disponibles
models = {
    "Fine-tuned GPT-2": "./fine_tuned_model/gpt_2_voyou",                   # mettre le path exact
    "Fine-tuned GPT-Neo 125M": "./fine_tuned_model/gpt_neo_125m",
}
selected_model = st.sidebar.radio("Select a model :", list(models.keys()))
model_path = models[selected_model]

# model = GPT2LMHeadModel.from_pretrained(model_path)
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialiser le session_state
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# ============================= Génération des réponses ============================================
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

# Afficher les messages
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))
        
        # Bouton pour lire la réponse
        if st.button(f"Lire la réponse {i+1}", key=f"read_button_{i}"):
            speak_text(st.session_state["generated"][i])

# Barre de saisie fixée en bas
with st.form(key="form", clear_on_submit=True):
    st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
    user_input = st.text_input("Type your message here...")
    send_button = st.form_submit_button("Send", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

# Gérer l'entrée utilisateur
if send_button:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
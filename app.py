from flask import Flask, render_template, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import os 

# --- ALTERAÇÃO AQUI ---
# Por padrão, o Flask procura na pasta 'templates'. 
# Vamos mudar isso para ele procurar na pasta atual ('.')
app = Flask(__name__, template_folder='.') 
# ----------------------

# O Flask precisa saber onde estão as imagens salvas
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'imagens_salvas') 
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

# ---------------------------
# Firebase setup (SÓ DATABASE)
# ---------------------------
# 1. Coloque sua chave na mesma pasta
# 2. Cole a URL do seu Realtime Database abaixo
cred_path = "chave_privada.json" # OU O CAMINHO COMPLETO (ex: r"C:\...")
db_url = "https://centro-conviver-default-rtdb.firebaseio.com/" 

try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": db_url,
    })
except ValueError:
    pass 

# ---------------------------
# Rotas da Aplicação
# ---------------------------

@app.route("/")
def index():
    db_ref = db.reference("/face_detections")
    detections_data = db_ref.get()
    
    processed_list = []
    
    if detections_data:
        for key, detection in detections_data.items():
            filename = detection.get("filename")
            image_url = ""
            
            if filename:
                image_url = f"/imagens/{filename}"
            else:
                image_url = "https://placehold.co/100x100/EEE/text=Sem+Imagem"
            
            processed_list.append({
                "timestamp": detection.get("timestamp", "N/A"),
                "public_image_url": image_url 
            })

    sorted_list = sorted(processed_list, key=lambda x: x['timestamp'], reverse=True)
    return render_template("index.html", detections=sorted_list)

@app.route("/imagens/<filename>")
def get_image(filename):
    """
    Esta rota serve as imagens da sua pasta local 'imagens_salvas'.
    """
    try:
        return send_from_directory(app.config['IMAGE_FOLDER'], filename)
    except FileNotFoundError:
        return "Imagem não encontrada", 404

@app.route("/detections") 
def get_detections():
    """ Rota de API (opcional) para buscar dados brutos """
    ref = db.reference("/face_detections")
    detections = ref.get()
    return detections if detections else {}


if __name__ == "__main__":
    app.run(debug=True)
import cv2
import pathlib
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db # <--- SÓ O DATABASE
import tkinter as tk
from PIL import Image, ImageTk
import os # <--- Importado para criar pastas


# Configuração do Firebase

# 1. Baixe sua chave no console do Firebase
# 2. Renomeie para "minha-chave-firebase.json" e coloque na mesma pasta
# 3. Cole a URL do seu Realtime Database abaixo
try:
    cred = credentials.Certificate("chave_privada.json") 
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://centro-conviver-default-rtdb.firebaseio.com/"
        
    })
except ValueError:
    # Evita erro se o app já foi inicializado
    pass

ref = db.reference("/face_detections")


# Configuração da Pasta Local 

# Nome da pasta onde as imagens serão salvas
LOCAL_STORAGE_PATH = "imagens_salvas" 
if not os.path.exists(LOCAL_STORAGE_PATH):
    os.makedirs(LOCAL_STORAGE_PATH) # Cria a pasta se ela não existir
    print(f"Pasta '{LOCAL_STORAGE_PATH}' criada.")


# Configuração do OpenCV

cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

# Variáveis de controle
face_detected_start = None
threshold_seconds = 5
confirmed_faces_count = 0
last_detection_time = "N/A"
last_confirmation_time = 0
cooldown_seconds = 10 


# Configuração da GUI (Tkinter)

window = tk.Tk()
window.title("Painel De Controle - Detecção Facial")
counter_label = tk.Label(window, text=f"Faces Confirmadas: {confirmed_faces_count}", font=("Arial", 14))
counter_label.pack()
time_label = tk.Label(window, text=f"Última Detecção: {last_detection_time}", font=("Arial", 14))
time_label.pack()
video_label = tk.Label(window)
video_label.pack()


# Função Principal (Update)

def update_frame():
    global face_detected_start, confirmed_faces_count, last_detection_time
    global last_confirmation_time 

    ret, frame = camera.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Lógica de detecção (5 segundos + cooldown)
        if len(faces) > 0:
            if face_detected_start is None:
                face_detected_start = time.time()
            else:
                elapsed = time.time() - face_detected_start
                
                # Verifica se passou dos 5s E se o cooldown de 10s já acabou
                if elapsed >= threshold_seconds and (time.time() - last_confirmation_time > cooldown_seconds):
                    
                    confirmed_faces_count += 1
                    last_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Face confirmada em {last_detection_time}!")

                    # --- Bloco de SALVAMENTO LOCAL (Modificado) ---
                    try:
                        # 1. Recorta a imagem
                        (x, y, w, h) = faces[0]
                        face_crop = frame[y:y+h, x:x+w]
                        
                        # 2. Define o nome do arquivo
                        filename = f"face_{last_detection_time.replace(':', '-').replace(' ', '_')}.jpg"
                        
                        # 3. Define o CAMINHO LOCAL COMPLETO 
                        local_path = os.path.join(LOCAL_STORAGE_PATH, filename)
                        
                        # 4. SALVA O ARQUIVO NO DISCO
                        cv2.imwrite(local_path, face_crop)
                        print(f"Imagem salva em: {local_path}")
                        
                        # 5. Salva o LOG no Realtime Database (APENAS o nome do arquivo)
                        ref.push({
                            "timestamp": last_detection_time,
                            "status": "confirmed",
                            "filename": filename # Salva só o nome do arquivo
                        })
                        
                    except Exception as e:
                        print(f"Erro ao salvar localmente ou no Firebase DB: {e}")
                    # --- Fim do Bloco de Salvamento ---

                    # Reseta os timers
                    face_detected_start = None
                    last_confirmation_time = time.time() 
                    
        else:
            face_detected_start = None # Reseta o timer se a face sumir

        # Atualiza os labels da GUI
        counter_label.config(text=f"Faces Confirmadas: {confirmed_faces_count}")
        time_label.config(text=f"Última Detecção: {last_detection_time}")

        # Converte o frame do OpenCV para a imagem do Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Agenda a próxima atualização do frame
    window.after(10, update_frame)


# Função de Limpeza (Cleanup)

def on_closing():
    """ Função chamada quando a janela do Tkinter é fechada. """
    print("Fechando... liberando câmera.")
    camera.release()
    cv2.destroyAllWindows()
    window.destroy()


# Início do Programa

print("Iniciando detector facial (Modo de salvamento local)...")
window.protocol("WM_DELETE_WINDOW", on_closing) 
update_frame()
window.mainloop()
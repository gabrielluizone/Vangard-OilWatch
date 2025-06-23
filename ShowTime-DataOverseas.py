# pip install opencv-python
# pip install ultralytics
# pip install requests
# pip install logging

import os
import cv2
from ultralytics import YOLO
import requests
import logging
import numpy as np

# Configura logging para suprimir mensagens de debug
logging.basicConfig(level=logging.WARNING)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Mostra seleção de modelo usando input()
print("\nSelecione o Modelo:")
print("1 - Pré-Treinado")
print("2 - Overseas-Vo5") 
print("3 - Overseas-To9")
print("Digite sua escolha (1, 2 ou 3): ", end="")

# Obtém entrada do usuário
escolha = input().strip()

# Define modelo baseado na seleção
if escolha == '1':
    nome_modelo = "yolov8n.pt"  # Usando YOLOv8n como modelo pré-treinado
elif escolha == '2':
    nome_modelo = "Vo5.pt"
elif escolha == '3':
    nome_modelo = "To9.pt"
else:
    print("Seleção inválida. Usando Vo5.pt como padrão")
    nome_modelo = "Vo5.pt"

# Obtém limiar de confiança do usuário
print("\nDigite o limiar de confiança (0-99): ", end="")
try:
    limiar_confianca = int(input().strip()) / 100
    limiar_confianca = max(0.0, min(0.99, limiar_confianca))  # Limita entre 0 e 0.99
except ValueError:
    print("Entrada inválida. Usando limiar padrão de 50%")
    limiar_confianca = .50

# Baixa modelo se não existir no diretório atual
if not os.path.exists(nome_modelo):
    if escolha == '1':
        url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    else:
        url = f"https://github.com/gabrielluizone/Vangard-OilWatch/raw/refs/heads/main/models/{nome_modelo}"
    
    with open(nome_modelo, 'wb') as f:
        for chunk in requests.get(url, stream=True).iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

# Carrega modelo YOLO com verbose=False para suprimir saída
modelo = YOLO(nome_modelo, verbose=False)

# Pergunta qual câmera usar
print("\nSelecione a câmera:")
print("0 - Câmera Principal")
print("1 - Webcam USB")
print("Digite sua escolha (0 ou 1): ", end="")
camera_index = int(input().strip())

# Abre câmera selecionada
captura = cv2.VideoCapture(camera_index)

# Obtém resolução máxima suportada pela câmera
largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configura janela com tamanho máximo da câmera
cv2.namedWindow("Detecção YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detecção YOLO", largura, altura)
print("\nPressione a tecla `q` para sair")

while True:
    # Lê frame da câmera
    ret, frame = captura.read()
    if not ret:
        break
    
    # Executa detecção YOLO com limiar de confiança
    resultados = modelo(frame, conf=limiar_confianca)
    
    # Exibe resultados
    frame_anotado = resultados[0].plot()
    
    cv2.imshow("Detecção YOLO", frame_anotado)
    
    # Interrompe loop se 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
captura.release()
cv2.destroyAllWindows()

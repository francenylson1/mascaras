import cv2
import mediapipe as mp
from adafruit_servokit import ServoKit
import math
import time

# --- Config PCA9685 ---
kit = ServoKit(channels=16)

# --- Máscaras e servos ---
mascaras = {
    "homem_de_ferro": {"vertical": 0, "horizontal": 1},
    "transformers": {"vertical": 2, "horizontal": 3},
    "homem_aranha": {"vertical": 4, "horizontal": 5},
    "minions": {"vertical": 6, "horizontal": 7},
    "hulk": {"vertical": 8, "horizontal": 9}
}

# --- Ângulos dos servos ---
posicoes = {
    "homem_de_ferro": {"vertical": {"cima":172,"centro":155,"baixo":124},
                        "horizontal":{"esquerda":55,"centro":100,"direita":141}},
    "transformers": {"vertical":{"cima":66,"centro":73,"baixo":124},
                     "horizontal":{"esquerda":85,"centro":37,"direita":0}},
    "homem_aranha": {"vertical":{"cima":110,"centro":90,"baixo":70},
                      "horizontal":{"esquerda":50,"centro":90,"direita":130}},
    "minions": {"vertical":{"cima":125,"centro":95,"baixo":65},
                "horizontal":{"esquerda":60,"centro":90,"direita":120}},
    "hulk": {"vertical":{"cima":135,"centro":100,"baixo":65},
             "horizontal":{"esquerda":60,"centro":90,"direita":120}}
}

# --- MediaPipe Config ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Funções auxiliares ---
def contar_dedos(hand_landmarks):
    dedos = []
    # Polegar
    dedos.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
    # Demais dedos
    for tip in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            dedos.append(1)
        else:
            dedos.append(0)
    return sum(dedos)

def palma_frente(hand_landmarks):
    # Polegar e punho para determinar se palma frente ou trás
    return hand_landmarks.landmark[0].z < hand_landmarks.landmark[9].z

# Variável para armazenar posições atuais dos servos
posicoes_atuais = {}

def inicializar_posicoes():
    """Inicializa as posições atuais dos servos"""
    for mascara in mascaras:
        for eixo in mascaras[mascara]:
            servo_id = mascaras[mascara][eixo]
            # Posição inicial no centro
            posicoes_atuais[servo_id] = posicoes[mascara][eixo]["centro"]
            kit.servo[servo_id].angle = posicoes_atuais[servo_id]
    time.sleep(1)  # Aguarda estabilização inicial

def mover_servo_suave(servo_id, angulo_destino, passo=2, delay=0.02):
    """Move o servo suavemente de 2 em 2 graus"""
    if servo_id not in posicoes_atuais:
        posicoes_atuais[servo_id] = kit.servo[servo_id].angle or 90
    
    angulo_atual = posicoes_atuais[servo_id]
    
    # Calcula a direção do movimento
    if angulo_atual < angulo_destino:
        # Movimento crescente
        for angulo in range(int(angulo_atual), int(angulo_destino) + 1, passo):
            kit.servo[servo_id].angle = angulo
            posicoes_atuais[servo_id] = angulo
            time.sleep(delay)
    elif angulo_atual > angulo_destino:
        # Movimento decrescente
        for angulo in range(int(angulo_atual), int(angulo_destino) - 1, -passo):
            kit.servo[servo_id].angle = angulo
            posicoes_atuais[servo_id] = angulo
            time.sleep(delay)
    
    # Garante que chegue exatamente no destino
    kit.servo[servo_id].angle = angulo_destino
    posicoes_atuais[servo_id] = angulo_destino

def mover_mascara(nome, eixo, direcao):
    servo_id = mascaras[nome][eixo]
    angulo_destino = posicoes[nome][eixo][direcao]
    mover_servo_suave(servo_id, angulo_destino)
    print(f"{nome} olhando {direcao} ({eixo})")

# --- Inicialização ---
print("Inicializando servos...")
inicializar_posicoes()
print("Servos inicializados!")

# --- Captura de vídeo ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_class in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos = contar_dedos(hand_landmarks)
            frente = palma_frente(hand_landmarks)
            lado = "direita" if frente else "esquerda"

            # Determinar qual mão
            hand_label = hand_class.classification[0].label  # "Left" ou "Right"

            # --- Mão Direita: horizontal ---
            if hand_label == "Right":
                if dedos == 1:
                    mover_mascara("homem_de_ferro", "horizontal", lado)
                elif dedos == 2:
                    mover_mascara("transformers", "horizontal", lado)
                elif dedos == 3:
                    mover_mascara("homem_aranha", "horizontal", lado)
                elif dedos == 4:
                    mover_mascara("minions", "horizontal", lado)
                elif dedos == 5:
                    mover_mascara("hulk", "horizontal", lado)

            # --- Mão Esquerda: vertical ---
            elif hand_label == "Left":
                if dedos == 1:
                    mover_mascara("homem_de_ferro", "vertical", "cima" if frente else "baixo")
                elif dedos == 2:
                    mover_mascara("transformers", "vertical", "cima" if frente else "baixo")
                elif dedos == 3:
                    mover_mascara("homem_aranha", "vertical", "cima" if frente else "baixo")
                elif dedos == 4:
                    mover_mascara("minions", "vertical", "cima" if frente else "baixo")
                elif dedos == 5:
                    mover_mascara("hulk", "vertical", "cima" if frente else "baixo")

    cv2.imshow("Controle SuperHeroi", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from adafruit_servokit import ServoKit
import math

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
    "homem_de_ferro": {"vertical": {"cima":120,"centro":90,"baixo":60},
                        "horizontal":{"esquerda":60,"centro":90,"direita":120}},
    "transformers": {"vertical":{"cima":130,"centro":100,"baixo":70},
                     "horizontal":{"esquerda":70,"centro":100,"direita":130}},
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

def mover_mascara(nome, eixo, direcao):
    servo = mascaras[nome][eixo]
    angulo = posicoes[nome][eixo][direcao]
    kit.servo[servo].angle = angulo
    print(f"{nome} olhando {direcao} ({eixo})")

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

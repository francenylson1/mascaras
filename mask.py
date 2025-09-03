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
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# --- Funções auxiliares ---
def contar_dedos(hand_landmarks, hand_label):
    dedos = []
    
    # Polegar - detecção simplificada baseada na distância horizontal
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    
    # Polegar aberto se a distância horizontal for significativa
    thumb_distance = abs(thumb_tip.x - thumb_mcp.x)
    dedos.append(1 if thumb_distance > 0.04 else 0)
    
    # Demais dedos - comparação vertical simples
    tip_ids = [8, 12, 16, 20]  # Indicador, médio, anelar, mindinho
    mcp_ids = [5, 9, 13, 17]   # Articulações da base (mais confiável)
    
    for tip, mcp in zip(tip_ids, mcp_ids):
        # Dedo levantado se a ponta está significativamente acima da base
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y - 0.02:
            dedos.append(1)
        else:
            dedos.append(0)
    
    return sum(dedos)

def palma_frente(hand_landmarks, hand_label):
    # Simplificado: sempre retorna True para focar apenas na contagem de dedos
    # Removemos a complexidade da orientação da palma
    return True

# Variáveis para armazenar posições atuais dos servos e estabilização
posicoes_atuais = {}
ultima_deteccao = {"Right": {"dedos": 0, "frente": False, "contador": 0}, 
                   "Left": {"dedos": 0, "frente": False, "contador": 0}}
MIN_FRAMES_ESTABILIZACAO = 3  # Número mínimo de frames para confirmar gesto

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

            # Determinar qual mão
            hand_label = hand_class.classification[0].label  # "Left" ou "Right"
            
            # Contar dedos e detectar orientação com maior precisão
            dedos = contar_dedos(hand_landmarks, hand_label)
            frente = palma_frente(hand_landmarks, hand_label)
            lado = "direita" if frente else "esquerda"
            
            # Sistema de estabilização - confirma gesto apenas após frames consecutivos
            gesto_estavel = False
            if (ultima_deteccao[hand_label]["dedos"] == dedos and 
                ultima_deteccao[hand_label]["frente"] == frente):
                ultima_deteccao[hand_label]["contador"] += 1
                if ultima_deteccao[hand_label]["contador"] >= MIN_FRAMES_ESTABILIZACAO:
                    gesto_estavel = True
            else:
                ultima_deteccao[hand_label]["dedos"] = dedos
                ultima_deteccao[hand_label]["frente"] = frente
                ultima_deteccao[hand_label]["contador"] = 1
            
            # Exibir informações de debug
            status = "ESTÁVEL" if gesto_estavel else f"DETECTANDO ({ultima_deteccao[hand_label]['contador']})"
            
            # Determinar ação baseada no gesto (qualquer mão)
            acao = ""
            if gesto_estavel:
                if dedos == 0:
                    acao = "→ DIREITA (MÃO FECHADA)"
                elif dedos == 5:
                    acao = "→ ESQUERDA (MÃO ABERTA)"
                else:
                    acao = f"→ SEM AÇÃO ({dedos} dedos)"
            else:
                acao = "→ DETECTANDO..."
            
            # Debug detalhado
            thumb_distance = abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x)
            cv2.putText(frame, f"{hand_label}: {dedos} dedos - {status}", 
                       (10, 30 if hand_label == "Right" else 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if gesto_estavel else (0, 255, 255), 2)
            cv2.putText(frame, f"Polegar dist: {thumb_distance:.3f} - {acao}", 
                       (10, 50 if hand_label == "Right" else 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Controle com qualquer mão
            if gesto_estavel:
                # Mão fechada (0 dedos) = girar todos para DIREITA
                if dedos == 0:
                    for mascara in mascaras:
                        mover_mascara(mascara, "horizontal", "direita")
                
                # Mão aberta (5 dedos) = girar todos para ESQUERDA
                elif dedos == 5:
                    for mascara in mascaras:
                        mover_mascara(mascara, "horizontal", "esquerda")

    cv2.imshow("Controle SuperHeroi", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

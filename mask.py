import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Simulação do ServoKit para desenvolvimento no desktop
try:
    from adafruit_servokit import ServoKit
    SERVOKIT_DISPONIVEL = True
except (ImportError, NotImplementedError):
    print("ServoKit não disponível - usando simulação")
    SERVOKIT_DISPONIVEL = False
    class ServoKit:
        def __init__(self, channels=16):
            self.servo = [type('servo', (), {'angle': 90})() for _ in range(channels)]

# Simulação do RPi.GPIO para desenvolvimento no desktop
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO_DISPONIVEL = True
except ImportError:
    print("RPi.GPIO não disponível - usando simulação")
    GPIO_DISPONIVEL = False
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, state): pass
        @staticmethod
        def cleanup(): pass

# --- Config PCA9685 ---
kit = ServoKit(channels=16)

# --- Config LEDs dos Olhos (GPIOs de teste) ---
leds_olhos = {
    "homem_de_ferro": {"olho_esquerdo": 18, "olho_direito": 19},
    "transformers": {"olho_esquerdo": 20, "olho_direito": 21},
    "homem_aranha": {"olho_esquerdo": 22, "olho_direito": 23},
    "minions": {"olho_esquerdo": 24, "olho_direito": 25},
    "hulk": {"olho_esquerdo": 26, "olho_direito": 27}
}

# Estado dos LEDs (True = aceso, False = apagado)
estado_leds = {mascara: False for mascara in leds_olhos.keys()}

# Configurar GPIOs dos LEDs
if GPIO_DISPONIVEL:
    for mascara, pinos in leds_olhos.items():
        GPIO.setup(pinos["olho_esquerdo"], GPIO.OUT)
        GPIO.setup(pinos["olho_direito"], GPIO.OUT)
        GPIO.output(pinos["olho_esquerdo"], GPIO.LOW)
        GPIO.output(pinos["olho_direito"], GPIO.LOW)

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
    
    # Polegar aberto se a distância horizontal for significativa (limiar reduzido)
    thumb_distance = abs(thumb_tip.x - thumb_mcp.x)
    dedos.append(1 if thumb_distance > 0.025 else 0)
    
    # Demais dedos - comparação vertical simples
    tip_ids = [8, 12, 16, 20]  # Indicador, médio, anelar, mindinho
    mcp_ids = [5, 9, 13, 17]   # Articulações da base (mais confiável)
    
    for tip, mcp in zip(tip_ids, mcp_ids):
        # Dedo levantado se a ponta está significativamente acima da base (limiar reduzido)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y - 0.015:
            dedos.append(1)
        else:
            dedos.append(0)
    
    return sum(dedos)

def detectar_gesto_rock(hand_landmarks, hand_label):
    """Detecta especificamente o gesto rock (indicador + mindinho estendidos)"""
    # Usar a mesma lógica da função contar_dedos para consistência
    dedos_individuais = []
    
    # Polegar - detecção simplificada baseada na distância horizontal
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    thumb_distance = abs(thumb_tip.x - thumb_mcp.x)
    polegar_estendido = thumb_distance > 0.025
    dedos_individuais.append(polegar_estendido)
    
    # Demais dedos - comparação vertical simples
    tip_ids = [8, 12, 16, 20]  # Indicador, médio, anelar, mindinho
    mcp_ids = [5, 9, 13, 17]   # Articulações da base
    
    for tip, mcp in zip(tip_ids, mcp_ids):
        # Dedo levantado se a ponta está significativamente acima da base
        dedo_estendido = hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y - 0.015
        dedos_individuais.append(dedo_estendido)
    
    # dedos_individuais = [polegar, indicador, medio, anelar, mindinho]
    polegar, indicador, medio, anelar, mindinho = dedos_individuais
    
    # Gesto rock: indicador E mindinho estendidos, outros fechados
    # Versão mais flexível: aceita se indicador e mindinho estão estendidos
    rock_basico = indicador and mindinho and not medio and not anelar
    
    # Versão ainda mais flexível: apenas indicador e mindinho estendidos (ignora polegar)
    rock_simples = indicador and mindinho
    
    # Debug contínuo quando há pelo menos 2 dedos
    total_dedos = sum(dedos_individuais)
    if total_dedos == 2:  # Exatamente 2 dedos estendidos
        print(f"\n=== ROCK DEBUG {hand_label} - {total_dedos} dedos ===")
        print(f"Polegar: {polegar}, Indicador: {indicador}, Médio: {medio}, Anelar: {anelar}, Mindinho: {mindinho}")
        print(f"Rock básico (sem médio/anelar): {rock_basico}")
        print(f"Rock simples (só indicador+mindinho): {rock_simples}")
        print(f"Thumb distance: {thumb_distance:.3f}")
    
    # Retorna True se detectar o padrão rock (começando com versão mais flexível)
    return rock_simples

def controlar_leds_olhos(acao):
    """Controla os LEDs dos olhos das máscaras"""
    global estado_leds
    
    if acao == "toggle":
        for mascara in estado_leds.keys():
            # Inverte o estado atual
            estado_leds[mascara] = not estado_leds[mascara]
            novo_estado = estado_leds[mascara]
            
            if GPIO_DISPONIVEL:
                # Controla os GPIOs reais
                GPIO.output(leds_olhos[mascara]["olho_esquerdo"], GPIO.HIGH if novo_estado else GPIO.LOW)
                GPIO.output(leds_olhos[mascara]["olho_direito"], GPIO.HIGH if novo_estado else GPIO.LOW)
            
            # Print para debug
            status = "ACESOS" if novo_estado else "APAGADOS"
            print(f"🔥 LEDs {mascara.upper()}: {status} (GPIOs {leds_olhos[mascara]['olho_esquerdo']}, {leds_olhos[mascara]['olho_direito']})")
        
        # Print geral
        status_geral = "ACESOS" if any(estado_leds.values()) else "APAGADOS"
        print(f"💡 TODOS OS LEDs DOS OLHOS: {status_geral}")

def palma_frente(hand_landmarks, hand_label):
    # Simplificado: sempre retorna True para focar apenas na contagem de dedos
    # Removemos a complexidade da orientação da palma
    return True

# Variáveis para armazenar posições atuais dos servos e estabilização
posicoes_atuais = {}
ultima_deteccao = {"Right": {"dedos": 0, "frente": False, "contador": 0, "rock": False, "contador_rock": 0}, 
                   "Left": {"dedos": 0, "frente": False, "contador": 0, "rock": False, "contador_rock": 0}}
MIN_FRAMES_ESTABILIZACAO = 3  # Número mínimo de frames para confirmar gesto
ultimo_toggle_leds = 0  # Controle para evitar múltiplas ativações dos LEDs

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
        posicoes_atuais[servo_id] = getattr(kit.servo[servo_id], 'angle', 90)
    
    angulo_atual = posicoes_atuais[servo_id]
    
    # Calcula a direção do movimento
    if angulo_atual < angulo_destino:
        # Movimento crescente
        for angulo in range(int(angulo_atual), int(angulo_destino) + 1, passo):
            if SERVOKIT_DISPONIVEL:
                kit.servo[servo_id].angle = angulo
            else:
                kit.servo[servo_id].angle = angulo
                print(f"Simulação: Servo {servo_id} -> {angulo}°")
            posicoes_atuais[servo_id] = angulo
            time.sleep(delay)
    elif angulo_atual > angulo_destino:
        # Movimento decrescente
        for angulo in range(int(angulo_atual), int(angulo_destino) - 1, -passo):
            if SERVOKIT_DISPONIVEL:
                kit.servo[servo_id].angle = angulo
            else:
                kit.servo[servo_id].angle = angulo
                print(f"Simulação: Servo {servo_id} -> {angulo}°")
            posicoes_atuais[servo_id] = angulo
            time.sleep(delay)
    
    # Garante que chegue exatamente no destino
    if SERVOKIT_DISPONIVEL:
        kit.servo[servo_id].angle = angulo_destino
    else:
        kit.servo[servo_id].angle = angulo_destino
        print(f"Simulação: Servo {servo_id} -> {angulo_destino}° (final)")
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
            
            # Detectar gesto rock
            rock_detectado = detectar_gesto_rock(hand_landmarks, hand_label)
            
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
            
            # Sistema de estabilização para gesto rock
            rock_estavel = False
            if ultima_deteccao[hand_label]["rock"] == rock_detectado:
                if rock_detectado:
                    ultima_deteccao[hand_label]["contador_rock"] += 1
                    if ultima_deteccao[hand_label]["contador_rock"] >= MIN_FRAMES_ESTABILIZACAO:
                        rock_estavel = True
                else:
                    ultima_deteccao[hand_label]["contador_rock"] = 0
            else:
                ultima_deteccao[hand_label]["rock"] = rock_detectado
                ultima_deteccao[hand_label]["contador_rock"] = 1 if rock_detectado else 0
            
            # Exibir informações de debug
            status = "ESTÁVEL" if gesto_estavel else f"DETECTANDO ({ultima_deteccao[hand_label]['contador']})"
            
            # Determinar ação baseada no gesto (qualquer mão)
            acao = ""
            if rock_estavel:
                acao = "🤟 ROCK - LEDs TOGGLE!"
            elif gesto_estavel:
                if dedos == 0:
                    acao = "→ DIREITA (MÃO FECHADA)"
                elif dedos >= 4:  # Aceita 4 ou 5 dedos como mão aberta
                    acao = f"→ ESQUERDA (MÃO ABERTA - {dedos} dedos)"
                else:
                    acao = f"→ SEM AÇÃO ({dedos} dedos)"
            else:
                if rock_detectado:
                    acao = f"🤟 DETECTANDO ROCK ({ultima_deteccao[hand_label]['contador_rock']})"
                else:
                    acao = "→ DETECTANDO..."
            
            # Debug detalhado
            thumb_distance = abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[2].x)
            rock_status = "ROCK ESTÁVEL" if rock_estavel else ("ROCK DETECTANDO" if rock_detectado else "")
            
            cv2.putText(frame, f"{hand_label}: {dedos} dedos - {status} {rock_status}", 
                       (10, 30 if hand_label == "Right" else 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if (gesto_estavel or rock_estavel) else (0, 255, 255), 2)
            cv2.putText(frame, f"Polegar dist: {thumb_distance:.3f} - {acao}", 
                       (10, 50 if hand_label == "Right" else 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Controle dos LEDs com gesto rock
            if rock_estavel and time.time() - ultimo_toggle_leds > 2:  # Evita múltiplas ativações
                controlar_leds_olhos("toggle")
                ultimo_toggle_leds = time.time()
            
            # Controle com qualquer mão
            if gesto_estavel:
                # Mão fechada (0 dedos) = girar todos para DIREITA
                if dedos == 0:
                    for mascara in mascaras:
                        mover_mascara(mascara, "horizontal", "direita")
                
                # Mão aberta (4 ou 5 dedos) = girar todos para ESQUERDA
                elif dedos >= 4:
                    for mascara in mascaras:
                        mover_mascara(mascara, "horizontal", "esquerda")

    cv2.imshow("Controle SuperHeroi", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Limpeza dos GPIOs
if GPIO_DISPONIVEL:
    GPIO.cleanup()
    print("GPIOs limpos com sucesso!")

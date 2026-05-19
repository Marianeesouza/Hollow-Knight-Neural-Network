import pygame

_GAMEPAD = None

def normalize_trigger(value):
    if value is None:
        return None
    if value < 0.0:
        return (value + 1) / 2
    return value

def discretize_axis(value, threshold=0.2):
    if value is None or abs(value) <= threshold:
        return 0.0
    return 1.0 if value > 0 else -1.0

def discretize_trigger(value, threshold=0.2):
    if value is None:
        return 0.0
    return 1.0 if value > threshold else 0.0

def capturar_snapshot_gamepad():
    """
    Captura e retorna o estado atual do primeiro gamepad conectado.
    Garante mapeamento consistente para controles padrão Xbox/XInput.
    """
    global _GAMEPAD
    
    if not pygame.get_init():
        pygame.init()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
        
    if _GAMEPAD is None:
        if pygame.joystick.get_count() == 0:
            return None
        _GAMEPAD = pygame.joystick.Joystick(0)
        try:
            _GAMEPAD.init()
        except Exception:
            _GAMEPAD = None
            return None

    # Atualiza o hardware
    pygame.event.pump()
    
    try:
        num_eixos = _GAMEPAD.get_numaxes()
        num_botoes = _GAMEPAD.get_numbuttons()

        # helpers seguros
        def safe_axis(idx, default=None):
            try:
                return _GAMEPAD.get_axis(idx) if idx < num_eixos else default
            except Exception:
                return default

        def safe_button(idx, default=0):
            try:
                return _GAMEPAD.get_button(idx) if idx < num_botoes else default
            except Exception:
                return default

        # --- AJUSTE DOS GATILHOS (Prevenção de erro de 0.5 parado) ---
        axis_4 = safe_axis(4)
        axis_5 = safe_axis(5)

        trigger_l = normalize_trigger(axis_4)
        trigger_r = normalize_trigger(axis_5)

        # Fallback para joysticks que expõem gatilhos como botões em vez de eixos.
        if trigger_l is None:
            trigger_l = float(safe_button(6)) if num_botoes > 6 else 0.0
        if trigger_r is None:
            trigger_r = float(safe_button(7)) if num_botoes > 7 else 0.0

        snapshot = {
            # Analógicos (Eixos Y invertidos: CIMA = 1.0, BAIXO = -1.0)
            "axis_left_x":   discretize_axis(safe_axis(0, 0.0)),
            "axis_left_y":   discretize_axis(safe_axis(1, 0.0) * -1),
            "axis_right_x":  discretize_axis(safe_axis(2, 0.0)) if num_eixos > 2 else 0.0,
            "axis_right_y":  discretize_axis(safe_axis(3, 0.0) * -1) if num_eixos > 3 else 0.0,

            # Gatilhos discretos (0.0 ou 1.0)
            "trigger_left":  discretize_trigger(trigger_l),
            "trigger_right": discretize_trigger(trigger_r),

            # Botões digitais principais (0 ou 1)
            "btn_a":         safe_button(0),
            "btn_b":         safe_button(1),
            "btn_x":         safe_button(2),

            # Sistema
            "btn_back":      safe_button(6),
            "btn_start":     safe_button(7),
        }
    except Exception:
        # Problema ao ler joystick nesse frame — considerar joystick desconectado
        try:
            if pygame.joystick.get_count() == 0:
                _GAMEPAD = None
        except Exception:
            pass
        return None
    
    return snapshot
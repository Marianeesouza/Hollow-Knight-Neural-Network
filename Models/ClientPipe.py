import time
import struct

from Models.DataHandler import DataHandler


class Pipe:
    def __init__(self, pipe_name='HK_RL_Pipe'):
        self.pipe_path = rf'\\.\pipe\{pipe_name}'
        self.pipe = None
        self.format_string = '<ffiiii????????f?f????fff?????ffiiiii'
        self.payload_size = struct.calcsize(self.format_string)

    def connect(self):
        print("Waiting for Hollow Knight to open...")
        while True:
            try:
                self.pipe = open(self.pipe_path, 'rb')
                print("✅ Successfully connected!")
                break
            except FileNotFoundError:
                time.sleep(1)

    def disconnect(self):
        if self.pipe:
            self.pipe.close()

    def read_state(self): #-> list:
        try:
            # Lê exatamente os 26 bytes do Pipe (A execução pausa aqui até o C# enviar)
            raw_bytes = self.pipe.read(self.payload_size)

            # Se o jogo fechar, o pipe quebra e retorna vazio.
            # Retornamos None em vez de 'continue' para não travar o Python.
            if not raw_bytes or len(raw_bytes) < self.payload_size:
                return None

            # Desempacota os bytes direto para variáveis no Python
            unpacked_data = struct.unpack(self.format_string, raw_bytes)

            # Monta o estado
            #"""
            state = {
                "px": unpacked_data[0],
                "py": unpacked_data[1],
                "hp": unpacked_data[2],
                "maxHp": unpacked_data[3],
                "soul": unpacked_data[4],
                "maxSoul": unpacked_data[5],
                #"dead": unpacked_data[4],
                "facingRight": unpacked_data[6],
                "facingLeft": unpacked_data[7],
                "onGround": unpacked_data[8],
                "falling": unpacked_data[9],
                "wallJumping": unpacked_data[10],
                "jumping": unpacked_data[11],
                "doubleJumping": unpacked_data[12],
                "dashing": unpacked_data[13],
                "dashCoolDown": unpacked_data[14],
                "shadowDashing": unpacked_data[15],
                "shadowDashCoolDown": unpacked_data[16],
                "canFocus": unpacked_data[17],
                "focusing": unpacked_data[18],
                "invulnerable": unpacked_data[19],
                "isAttacking": unpacked_data[20],
                "attackDuration": unpacked_data[21],
                "attackRecoveryTime": unpacked_data[22],
                "attackCooldownTime": unpacked_data[23],
                "attackForward": unpacked_data[24],
                "attackUp": unpacked_data[25],
                "attackDown": unpacked_data[26],
                "canCast": unpacked_data[27],
                "casting": unpacked_data[28],
                "bx": unpacked_data[29],
                "by": unpacked_data[30],
                "bossHp": unpacked_data[31],
                "bossMaxHp": unpacked_data[32],
                #"bossScene": unpacked_data[23],
                #"bossIsDead": unpacked_data[24],
                "bossState": unpacked_data[33],
                "bossStateAmount": unpacked_data[34],
                "bossScene": unpacked_data[35],
                # boss here...
            }#"""

            return state
            #return DataHandler.treat_data(unpacked_data)

        except Exception as e:
            print(f"Erro ao ler binário: {e}")
            return None
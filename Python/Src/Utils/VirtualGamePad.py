import vgamepad as vg

class VirtualGamePad:

    def __init__(self):

        self.gamepad = vg.VX360Gamepad()

        self.commands = {
            0: None,                                    # Idle
            1: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,     # Look up
            2: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,   # Look down
            3: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,   # move left
            4: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,  # move right
            5: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,           # jump
            6: None,                                    # dash
            7: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,           # attack
            8: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,           # cast / heal
        }

    def update_gamepad(self, active_actions):
        for i, button in self.commands.items():

            if i == 0:
                continue

            if i in active_actions:
                if i == 6:
                    self.update_trigger(1)
                else:
                    self.gamepad.press_button(button=button)
            else:
                if i == 6:
                    self.update_trigger(0)
                else:
                    self.gamepad.release_button(button=button)

        # Envia todos os estados de uma vez
        self.gamepad.update()

    def update_trigger(self, value):
        analog_value = 255 if value else 0
        self.gamepad.right_trigger(analog_value)

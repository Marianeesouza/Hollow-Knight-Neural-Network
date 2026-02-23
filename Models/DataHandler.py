import numpy as np

class DataHandler:

    @staticmethod
    def stored_size(data: np.ndarray) -> int:

        if data is None:
            return 0

        return data.size

    @staticmethod
    def show(data, index: int = None):
        if index is not None:
            print(f'\r{data[index]}', end="", flush=True)
        else:
            print(f'\r{data}', end="", flush=True)

    @staticmethod
    def treat_data(data: dict) -> np.ndarray:
        result = []

        if not data:
            return np.array(result, dtype=float)

        pos_player = np.array([data.get('px', 0.0), data.get('py', 0.0)])
        pos_boss = np.array([data.get('bx', 0.0), data.get('by', 0.0)])

        delta = pos_boss - pos_player

        delta_norm = 1 / (np.abs(delta / 10) + 1)

        delta_norm = delta_norm * np.sign(delta)

        result.append(float(delta_norm[0]))  # deltaX
        result.append(float(delta_norm[1]))  # deltaY


        for key, value in data.items():

            # if it is a boolean
            if isinstance(value, bool):
                result.append(1.0 if value else 0.0) # transform it in 0 or 1
                continue

            # one-hot for bossState
            elif key == 'bossState':
                one_hot = [0.0] * data['bossStateAmount']

                if value is not None and 0 <= int(value) < data['bossStateAmount']:
                    one_hot[int(value)] = 1.0

                # adding it to the list
                result.extend(one_hot)
                continue

            else:
                if value is None: # if the value is null
                    result.append(-1.0) # null flag (-1)
                    continue
                else:
                    if key in ['maxHp', 'maxSoul', 'maxBossHp', 'maxBossStateAmount', 'bx', 'by', 'bossScene']:
                        continue

                    match key:
                        case 'hp':
                            max_hp = data.get('maxHp', 1)
                            result.append(float(value / max_hp) if max_hp > 0 else 0.0)
                            continue
                        case 'soul':
                            max_soul = data.get('maxSoul', 1)
                            result.append(float(value / max_soul) if max_soul > 0 else 0.0)
                            continue
                        case 'bossHp':
                            max_boss_hp = data.get('maxBossHp', 1)
                            result.append(float(value / max_boss_hp) if max_boss_hp > 0 else 0.0)
                            continue
                    result.append(float(value))

        return np.array(result, dtype=float)
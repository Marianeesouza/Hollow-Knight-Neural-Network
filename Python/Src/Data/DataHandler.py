import numpy as np

MAX_POSITION_X = 37.74
MIN_POSITION_X = 15.27
MAX_POSITION_Y = 40
MIN_POSITION_Y = 28.41
MAX_PLAYER_VELOCITY_X = 20
MAX_PLAYER_VELOCITY_Y = 20.95
MAX_BOSS_VELOCITY_X = 29.37
MAX_BOSS_VELOCITY_Y = 41


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
        if not data:
            return np.array([], dtype=float)

        result = []

        # relative position (player to boss)
        pos_player = np.array([data.get('px', 0.0), data.get('py', 0.0)])
        pos_boss = np.array([data.get('bx', 0.0), data.get('by', 0.0)])

        pos_delta = pos_boss - pos_player

        range_x = MAX_POSITION_X - MIN_POSITION_X
        range_y = MAX_POSITION_Y - MIN_POSITION_Y

        # normalizing the position delta to be between -1 and 1, considering the maximum range of positions in the game
        pos_delta_norm = pos_delta / np.array([range_x, range_y])
        pos_delta_norm = np.clip(pos_delta_norm, -1, 1)

        #pos_delta_norm = 1 / (np.abs(pos_delta / 10) + 1)

        #pos_delta_norm = pos_delta_norm * np.sign(pos_delta)

        result.append(float(pos_delta_norm[0]))  # deltaX
        result.append(float(pos_delta_norm[1]))  # deltaY

        # relative velocity (player to boss)
        velocity_player = np.array([data.get('pvx', 0.0), data.get('pvy', 0.0)])
        velocity_boss = np.array([data.get('bvx', 0.0), data.get('bvy', 0.0)])

        velocity_delta = velocity_boss - velocity_player

        # normalizing the velocity delta considering the maximum range of velocities in the game
        result.append(float(velocity_delta[0]) / (MAX_BOSS_VELOCITY_X + MAX_PLAYER_VELOCITY_X)) # velocity deltaX
        result.append(float(velocity_delta[1]) / (MAX_BOSS_VELOCITY_Y + MAX_PLAYER_VELOCITY_Y)) # velocity deltaY

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
                    if key in ['maxHp', 'maxSoul', 'bossMaxHp', 'bossStateAmount', 'px', 'py', 'bx', 'by', 'bossScene', 'bvx', 'bvy', 'bossScene', 'bossState']:
                        continue

                    match key:
                        case 'pvx':
                            # normalizing the player velocity considering the maximum range of velocities in the game
                            result.append(float(value / MAX_PLAYER_VELOCITY_X))
                            continue
                        case 'pvy':
                            # normalizing the player velocity considering the maximum range of velocities in the game
                            result.append(float(value / MAX_PLAYER_VELOCITY_Y))
                            continue
                        #case 'bvx':
                            #result.append(float(value / MAX_BOSS_VELOCITY_X))
                            #continue
                        #case 'bvy':
                            #result.append(float(value / MAX_BOSS_VELOCITY_Y))
                            #continue
                        case 'hp':
                            max_hp = data.get('maxHp', 1)
                            result.append(float(value / max_hp) if max_hp > 0 else 0.0)
                            continue
                        case 'soul':
                            max_soul = data.get('maxSoul', 1)
                            result.append(float(value / max_soul) if max_soul > 0 else 0.0)
                            continue
                        case 'bossHp':
                            max_boss_hp = data.get('bossMaxHp', 1)
                            result.append(float(value / max_boss_hp) if max_boss_hp > 0 else 0.0)
                            continue
                    result.append(float(value))

        return np.array(result, dtype=float)
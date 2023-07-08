import numpy as np
import random
import os

# Funciones auxiliares que pueden resultar útiles para tu implementación
from utils import bfs_search, get_valid_moves

# Path actual de trabajo
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# Hiperparámetros de entrenamiento (jugar con ellos, estudiar que ocurre al cambiarlos)
CAT_MAX_EXPLORATION_RATE = 1
CAT_MIN_EXPLORATION_RATE = 0.0001
CAT_EXPLORATION_DECAY_RATE = 0.0001
CAT_LR = 0.1
CAT_DISCOUNT_RATE = 0.1

MOUSE_MAX_EXPLORATION_RATE = 1
MOUSE_MIN_EXPLORATION_RATE = 0.0001
MOUSE_EXPLORATION_DECAY_RATE = 0.0001
MOUSE_LR = 0.1
MOUSE_DISCOUNT_RATE = 0.1

class ReinforcedAgent:

    def __init__(self, position, table_name = None, alpha = 0.2, gamma = 0.9):

        # Posición inicial del agente
        self.pos = position
        self.gamma = gamma
        self.alpha = alpha

        # ===== CONSTRUCCIÓN DE LA Q-TABLE ===== #
        # Cargamos el mapa y buscamos las posiciones libres dentro de este
        free_positions = []
        lab_map = np.load(os.path.join(CURRENT_PATH, "game_map.npy"))
        for x in range(lab_map.shape[0]):
            for y in range(lab_map.shape[1]):
                if lab_map[x, y] == 0:
                    free_positions.append((x, y))

        # Diccionario que recibe una tupla del estado de juego de key y retorna el índice de su fila asociada en la Q-Table
        self.states_index = dict()
        index = 0
        for cat_pos in free_positions:
            for mouse_pos in free_positions:
                self.states_index[tuple([cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1]])] = index
                index += 1
            
        # Tasa de exploración del agente
        self.exploration_rate = 1
        
        # En caso de haber una Q-Table preexistente, utilizarla
        if table_name is None:
            self.q_table = np.zeros((index, 5))
        
        # En caso de no entregar una Q-Table, crear una llena de ceros
        else:
            self.q_table = np.load(os.path.join(CURRENT_PATH, "data", table_name))


    # Obtener la acción a ejecutar dado el estado del juego
    def get_action(self, lab_map, cat_pos, mouse_pos, noise = 0, train = False):

        # Si entrenamos, considerar si explorar o explotar
        if train:
            if random.random() < self.exploration_rate:
                return random.randint(0, 4)
        
        # Si no, considerar los movimientos como ruidosos, con una probabilidad de hacer uno aleatorio
        else:
            if random.random() < noise:
                return random.randint(0, 4)
        
        # Calculamos el estado actual del juego
        state = (cat_pos[0], cat_pos[1], mouse_pos[0], mouse_pos[1])


        # ===== COMPLETAR =====
        # Se debe retornar el movimiento que lleve a un mejor estado futuro, basándose en la Q-Table
        # move = 0
        # =====================

        # Obtener el índice del estado actual en la tabla Q
        state_index = self.states_index[state]

        # Obtener el movimiento que tiene el valor más alto en la tabla Q para el estado actual
        move = np.argmax(self.q_table[state_index])

        return move
    
    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        pass
    
    def update_policy(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        state = (old_cat_pos[0], old_cat_pos[1], old_mouse_pos[0], old_mouse_pos[1])
        new_state = (new_cat_pos[0], new_cat_pos[1], new_mouse_pos[0], new_mouse_pos[1])
        
        reward = self.get_reward(lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos)

        # ===== COMPLETAR =====
        # Se debe actualizar el valor asociado al par estado-acción en la Q-Table
        # recuerda que la acción jugada fue action en el estado state
        # self.q_table[self.states_index[state], action] = 0
        # =====================

        # Obtener el índice del estado actual en la tabla Q
        q_prime = np.max(self.q_table[self.states_index[new_state]])
        # Obtener el índice del estado actual en la tabla Q
        index = self.states_index[state]
        old_q_value = self.q_table[index, action]


        # if reward < 0:
        #     learned_value = self.gamma * q_prime - old_q_value
        # else:
        #     learned_value = reward - old_q_value

        learned_value = reward + self.gamma * q_prime - old_q_value

        #learned_value = r + self.gamma * q_prime - old_q_value
        self.q_table[index, action] += self.alpha * learned_value

        # print('valor aprendido', learned_value)
        # print('sumado', self.alpha * learned_value)

    
    def update_exploration(self, n_game):
        # Disminuir la tasa de exploración a medida que el agente juega más juegos
        pass

class RLCat(ReinforcedAgent):

    def __init__(self, position, table_path = None, alpha = CAT_LR, gamma = CAT_DISCOUNT_RATE):

        super().__init__(position, table_path, alpha = CAT_LR, gamma = CAT_DISCOUNT_RATE)

    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # ===== COMPLETAR =====
        # Se debe calcular el reward para la acción realizada por el agente
        # reward = 0
        # =====================

        distancia_old = len(bfs_search(lab_map, old_cat_pos, old_mouse_pos))
        distancia_new = len(bfs_search(lab_map, new_cat_pos, new_mouse_pos))

        # print("old_cat_pos: ", old_cat_pos[0], old_cat_pos[1])
        # print("new_cat_pos: ", new_cat_pos[0], new_cat_pos[1])
        # Si el gato captura al ratón, dar una recompensa positiva
        if new_cat_pos[0] == new_mouse_pos[0] and new_cat_pos[1] == new_mouse_pos[1]:
            reward = 30
        # Si el gato se mueve y se acerca al ratón, dar una recompensa positiva
        elif distancia_new < distancia_old:
            reward = 5
        # Si la distancia se mantiene, dar una recompensa neutra
        elif distancia_new == distancia_old:
            reward = 0
        # Si la distancia aumenta, dar una recompensa negativa
        elif distancia_new > distancia_old:
            reward = -5

        return reward
    
    def update_exploration(self, n_game):
        # ===== COMPLETAR =====
        # Se debe actualizar la tasa de exploración del agente
        # self.exploration_rate = 0
        # =====================

        # Disminuir la tasa de exploración a medida que el gato juega más juegos
        self.exploration_rate = CAT_MIN_EXPLORATION_RATE + (CAT_MAX_EXPLORATION_RATE - CAT_MIN_EXPLORATION_RATE) * np.exp(-CAT_EXPLORATION_DECAY_RATE * n_game)


        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableCat{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableCat{n_game}.npy")

    def update_exploration(self, n_game):
        # Disminuir la tasa de exploración a medida que el gato juega más juegos
        self.exploration_rate = CAT_MIN_EXPLORATION_RATE + (CAT_MAX_EXPLORATION_RATE - CAT_MIN_EXPLORATION_RATE) * np.exp(-CAT_EXPLORATION_DECAY_RATE * n_game)

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableCat{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableCat{n_game}.npy")
    
class RLMouse(ReinforcedAgent):
    def __init__(self, position, table_path = None, alpha = MOUSE_LR, gamma = MOUSE_DISCOUNT_RATE):

        super().__init__(position, table_path, alpha = MOUSE_LR, gamma = MOUSE_DISCOUNT_RATE)

    def get_reward(self, lab_map, action, old_cat_pos, new_cat_pos, old_mouse_pos, new_mouse_pos):
        # ===== COMPLETAR =====
        # Se debe calcular el reward para la acción realizada por el agente
        # reward = 0
        # =====================

        distancia_old = len(bfs_search(lab_map, old_cat_pos, old_mouse_pos))
        distancia_new = len(bfs_search(lab_map, new_cat_pos, new_mouse_pos))

        # Si el ratón es capturado por el gato, dar una recompensa negativa
        if new_cat_pos[0] == new_mouse_pos[0] and new_cat_pos[1] == new_mouse_pos[1]:
            reward = -30
        # Si el gato se mueve y se acerca al ratón, dar una recompensa negativa
        elif distancia_new < distancia_old:
            reward = -5
        # Si la distancia aumenta, dar una recompensa positiva
        elif distancia_new > distancia_old:
            reward = 5
        # Si la distancia se mantiene, dar una recompensa neutra
        elif distancia_new == distancia_old:
            reward = 2


        return reward
    
    def update_exploration(self, n_game):
        # ===== COMPLETAR =====
        # Se debe actualizar la tasa de exploración del agente
        # self.exploration_rate = 0
        # =====================

        # Disminuir la tasa de exploración a medida que el ratón juega más juegos
        self.exploration_rate = MOUSE_MIN_EXPLORATION_RATE + (MOUSE_MAX_EXPLORATION_RATE - MOUSE_MIN_EXPLORATION_RATE) * np.exp(-MOUSE_EXPLORATION_DECAY_RATE * n_game)

        # Cada 1000 partidas, aprovecharemos de guardar la tabla de desempeño del agente
        if n_game % 1000 == 0:
            np.save(os.path.join(CURRENT_PATH, "data", f"QTableMouse{n_game}.npy"), self.q_table)
            print(f"Epsilon: {self.exploration_rate} | Guardando QTable en agents/data/QTableMouse{n_game}.npy")

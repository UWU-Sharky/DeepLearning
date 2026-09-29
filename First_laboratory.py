import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CityDeliveryEnv:
    def __init__(self, grid_map):
        # Сохраняем исходную карту
        self.base_map = grid_map.copy()
        self.height, self.width = grid_map.shape
        
        # Находим координаты ключевых объектов:
        # Стартовые позиции агента (значение 2)
        self.start_positions = list(zip(*np.where(self.base_map == 2)))
        
        # Точки появления других транспортных средств (значение 3)
        self.spawn_positions = list(zip(*np.where(self.base_map == 3)))
        
        # Целевая область (значение 4)
        self.target_positions = list(zip(*np.where(self.base_map == 4)))
        
        # Карта чистого дорожного полотна (где 0, 2, 3, 4 - это проезжая часть)
        # 1 - стены/препятствия
        self.walkable_mask = (self.base_map != 1)
        
        # Действия агента: 0: Вверх, 1: Вниз, 2: Влево, 3: Вправо, 4: Стоп
        self.actions = {
            0: (-1, 0),  
            1: (1, 0),  
            2: (0, -1), 
            3: (0, 1),   
            4: (0, 0)
        }
        
        # Текущие динамические переменные
        self.agent_pos = None
        self.vehicles = []  # Список активных машин

    def start(self):
        self.reset(np.random.choice(len(self.start_positions)))



    def reset(self, start_idx):
        """Сброс состояния среды перед новым эпизодом."""

        # Выбираем начальную точку агента
        self.agent_pos = self.start_positions[start_idx]    
        # Очищаем список автомобилей
        self.vehicles = []
        
        return self._get_observation()

    def step(self, action: int):
        """
        Выполнение одного шага в среде.
        action: 0 - Вверх, 1 - Вниз, 2 - Влево, 3 - Вправо, 4 - Стоп
        """
        # 1. Вычисляем предполагаемую новую позицию агента
        dr, dc = self.actions[action]
        curr_r, curr_c = self.agent_pos
        next_r, next_c = curr_r + dr, curr_c + dc
        
        # 2. Проверяем, свободна ли ячейка (не стена)
        if 0 <= next_r < self.height and 0 <= next_c < self.width:
            if self.walkable_mask[next_r, next_c]:
                self.agent_pos = (next_r, next_c)  # Перемещаем агента
        
        # 3. Обновляем движение трафика
        self._update_vehicles()
        
        # 4. Проверяем условия завершения эпизода и рассчитываем награду
        reward = -1  # Стандартный штраф за каждый шаг (время)
        done = False
        info = {"status": "in_progress"}
        
        # Список всех текущих позиций машин
        vehicles_positions = [v["pos"] for v in self.vehicles]
        
        # Условие 1: Авария (столкновение с машиной)
        if self.agent_pos in vehicles_positions:
            reward = -100
            done = True
            info["status"] = "crash"
            
        # Условие 2: Достижение цели (ячейка со значением 4)
        elif self.base_map[self.agent_pos] == 4:
            reward = 100
            done = True
            info["status"] = "success"
            
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        """Возвращает текущее состояние среды."""
        return {
            "agent_pos": self.agent_pos,
            "vehicles_pos": [v["pos"] for v in self.vehicles]
        }
    
    def _update_vehicles(self, spawn_prob: float = 0.02):
        """Логика спавна и движения сторонних автомобилей."""
        
        # 1. Спавн новых машин в точках '3' с вероятностью (например, 0.02)
        for spawn_pos in self.spawn_positions:
            # Проверяем, нет ли уже машины в этой точке
            occupied = any(v["pos"] == spawn_pos for v in self.vehicles)
            if not occupied and np.random.rand() < spawn_prob:
                # Генерируем время жизни от 15 до 150 шагов
                lifetime = np.random.randint(15, 151)
                self.vehicles.append({
                    "pos": spawn_pos,
                    "last_dir": None,  # У новой машины еще нет направления
                    "lifetime": lifetime
                })
        
        updated_vehicles = []
        
        # 2. Перемещение существующих машин
        for vehicle in self.vehicles:
            vehicle["lifetime"] -= 1
            
            # Если время жизни истекло, машина убирается (припарковывается)
            if vehicle["lifetime"] <= 0:
                continue
                
            curr_r, curr_c = vehicle["pos"]
            
            # Находим все доступные соседние ячейки (где НЕ стена '1')
            valid_moves = []
            for action_id, (dr, dc) in self.actions.items():
                if action_id == 4: # Исключаем остановку для машин
                    continue
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if self.walkable_mask[nr, nc]:
                        valid_moves.append((action_id, (nr, nc)))
            
            # Если ехать некуда, машина остается на месте
            if not valid_moves:
                updated_vehicles.append(vehicle)
                continue
                
            last_dir = vehicle["last_dir"]
            chosen_move = None
            
            # Проверяем, доступно ли продолжение предыдущего направления
            same_dir_move = [m for m in valid_moves if m[0] == last_dir]
            
            # Выбор направления согласно правилам:
            if same_dir_move and np.random.rand() < 0.7:
                # 70% вероятность продолжить движение прямо
                chosen_move = same_dir_move[0]
            else:
                # Выбор случайного направления из доступных на развилке / из тупика
                rand_idx = np.random.choice(len(valid_moves))
                chosen_move = valid_moves[rand_idx]
                
            # Обновляем координаты и направление машины
            vehicle["last_dir"] = chosen_move[0]
            vehicle["pos"] = chosen_move[1]
            updated_vehicles.append(vehicle)
            
        self.vehicles = updated_vehicles

if __name__ == '__main__':
    with open('citymap.txt', 'r', encoding='utf-8') as f:
        lines = [list(line.strip()) for line in f if line.strip()]
    Enviroment = CityDeliveryEnv(np.array(lines, dtype='U1').astype(int))
    Enviroment.start()

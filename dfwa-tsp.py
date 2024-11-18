import numpy as np
import random
from readTSP import read_tsp_file

class DFWA_TSP:
    def __init__(self, distance_matrix, num_fireworks=5, nspk_factor=100, mutation_factor=3, min_amp=0.5, max_amp=30, max_iter=1000):
        self.distance_matrix = distance_matrix
        self.num_fireworks = num_fireworks
        self.nspk_factor = nspk_factor
        self.mutation_factor = mutation_factor
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.max_iter = max_iter
        self.num_cities = distance_matrix.shape[0]
        self.fireworks = []
        self.best_tour = None
        self.best_distance = float('inf')
        self.amp = (min_amp + max_amp) / 2

    def greedy_initialization(self):
        self.fireworks = []
        for _ in range(self.num_fireworks):
            tour = [random.randint(0, self.num_cities - 1)]
            while len(tour) < self.num_cities:
                last_city = tour[-1]
                next_city = min(set(range(self.num_cities)) - set(tour), key=lambda x: self.distance_matrix[last_city, x])
                tour.append(next_city)
            self.fireworks.append(tour)

    def calculate_distance(self, tour):
        return sum(self.distance_matrix[tour[i], tour[(i + 1) % self.num_cities]] for i in range(self.num_cities))

    def calculate_firework_number(self):
        longest_distance = max(self.calculate_distance(tour) for tour in self.fireworks)
        total_diff = sum(longest_distance - self.calculate_distance(tour) + 1e-8 for tour in self.fireworks)
        num_spks = []
        for tour in self.fireworks:
            tour_distance = self.calculate_distance(tour)
            strength = round(self.nspk_factor * (longest_distance - tour_distance + 1e-8) / total_diff)
            # print(strength)
            num_spks.append(max(1, strength))
        return num_spks

    def explode(self):
        sparks = []
        num_spks = self.calculate_firework_number()  
        
        # 几何平均振幅用于判断分界
        theta_threshold = (self.min_amp * self.max_amp) ** 0.5

        for firework, num_sparks in zip(self.fireworks, num_spks):
            firework_distance = self.calculate_distance(firework)  

            for _ in range(num_sparks):
                spark = firework[:]
                # 振幅控制选择操作
                if self.amp > theta_threshold:
                    # 使用 2-opt 操作（局部优化）
                    i, j = sorted(random.sample(range(self.num_cities), 2))
                    spark[i:j] = reversed(spark[i:j])
                else:
                    # 使用 3-opt 操作（全局扰动）
                    i, j, k = sorted(random.sample(range(self.num_cities), 3))
                    spark = spark[:i] + spark[j:k] + spark[i:j] + spark[k:]

                # 计算新解的路径长度
                spark_distance = self.calculate_distance(spark)

                # 接受火花的概率计算
                if spark_distance < firework_distance:
                    sparks.append(spark)  # 更优解直接接受
                else:
                    # 按概率接受次优解
                    acceptance_probability = np.exp(-spark_distance / firework_distance * self.amp)
                    if random.random() < acceptance_probability:
                        sparks.append(spark)
        return sparks
    
    def update_amplitude(self, current_best_distance, previous_best_distance):
        """动态调整振幅"""
        if current_best_distance >= previous_best_distance:
            self.amp = max(self.amp / 1.1, self.min_amp)
        else:
            self.amp = min(self.amp * 1.1, self.max_amp)

    def mutate(self, firework): # 这里mutation_factor是指定的，没有计算，可改进
        """基于插入的变异方法"""
        mutations = []
        firework_distance = self.calculate_distance(firework)

        for _ in range(self.mutation_factor):
            mutation = firework[:]
            # 随机选择一个城市
            city_idx = random.randint(0, self.num_cities - 1)
            city = mutation.pop(city_idx)

            # 找到所有非相邻的边
            non_adjacent_edges = []
            for i in range(len(mutation)):
                a, b = mutation[i], mutation[(i + 1) % len(mutation)]
                if city not in (a, b):  # 排除相邻边
                    non_adjacent_edges.append((i, a, b))

            # 随机排列非相邻边
            random.shuffle(non_adjacent_edges)

            accepted = False
            for edge_idx, a, b in non_adjacent_edges:
                # 尝试将 city 插入到边 [a, b] 之间
                new_mutation = mutation[:edge_idx + 1] + [city] + mutation[edge_idx + 1:]
                new_distance = self.calculate_distance(new_mutation)

                if new_distance < firework_distance:
                    # 如果路径长度减少，直接接受
                    mutations.append(new_mutation)
                    accepted = True
                    break
                else:
                    # 按概率接受次优解
                    acceptance_probability = np.exp(-new_distance / firework_distance * self.amp)
                    if random.random() < acceptance_probability:
                        mutations.append(new_mutation)
                        accepted = True
                        break

            if not accepted:
                # 如果所有插入尝试都未接受，则随机插入到某条非相邻边
                random_edge = random.choice(non_adjacent_edges)
                edge_idx = random_edge[0]
                mutation = mutation[:edge_idx + 1] + [city] + mutation[edge_idx + 1:]
                mutations.append(mutation)

        return mutations

    def select(self, sparks):
        """基于概率的选择机制"""
        candidates = self.fireworks + sparks  # 合并烟花和火花
        distances = [self.calculate_distance(tour) for tour in candidates]  # 计算所有候选解的路径长度
        best_index = np.argmin(distances)  # 找到最优解的索引
        best_tour = candidates[best_index]  # 最优解
        best_distance = distances[best_index]

        # 确保最优解保留
        next_generation = [best_tour]
        next_distances = [best_distance]

        # 剔除最优解的候选列表
        remaining_candidates = [candidates[i] for i in range(len(candidates)) if i != best_index]
        remaining_distances = [distances[i] for i in range(len(distances)) if i != best_index]

        # 计算选择概率
        epsilon = 1e-8  # 防止分母为 0
        probabilities = []
        for dist in remaining_distances:
            # probabilities.append(1 / ((dist - best_distance) ** 2 + epsilon))
            probabilities.append(np.exp(-((dist - best_distance) / best_distance) ** 2))
        total_probability = sum(probabilities)
        probabilities = [p / total_probability for p in probabilities]  # 归一化
        # print(probabilities) # 出现概率分布极端导致卡住的情况
        # 按概率选择其余烟花
        while len(next_generation) < self.num_fireworks:
            selected_index = np.random.choice(len(remaining_candidates), p=probabilities)
            selected_tour = remaining_candidates[selected_index]
            if selected_tour not in next_generation:  # 避免重复选择
                next_generation.append(selected_tour)
                next_distances.append(remaining_distances[selected_index])

        # 更新烟花和最优解
        self.fireworks = next_generation
        self.best_tour = best_tour
        self.best_distance = best_distance


    
    def optimize(self):
        self.greedy_initialization()
        previous_best_distance = float('inf')

        for iteration in range(self.max_iter):
            sparks = []
            for firework in self.fireworks:
                sparks += self.explode()          # 爆炸火花
                sparks += self.mutate(firework)  # 插入变异火花

            self.select(sparks)  # 选择下一代烟花
            self.update_amplitude(self.best_distance, previous_best_distance)
            previous_best_distance = self.best_distance

            # 打印中间结果
            if iteration % 10 == 0:  # 每 10 次迭代打印一次
                print(f"Iteration {iteration}: Best Distance = {self.best_distance}")

        return self.best_tour, self.best_distance
    
# def read_tsp_file(filename):
#     with open(filename, 'r') as f:
#         lines = f.readlines()
    
#     dimension = None
#     edge_weight_type = None
#     edge_weight_format = None
#     edge_weight_section = False
#     edge_weights = []
    
#     for line in lines:
#         line = line.strip()
#         if line.startswith('NAME'):
#             name = line.split(':')[1].strip()
#         elif line.startswith('TYPE'):
#             problem_type = line.split(':')[1].strip()
#         elif line.startswith('COMMENT'):
#             comment = line.split(':')[1].strip()
#         elif line.startswith('DIMENSION'):
#             dimension = int(line.split(':')[1])
#             distance_matrix = np.zeros((dimension, dimension))
#         elif line.startswith('EDGE_WEIGHT_TYPE'):
#             edge_weight_type = line.split(':')[1].strip()
#         elif line.startswith('EDGE_WEIGHT_FORMAT'):
#             edge_weight_format = line.split(':')[1].strip()
#         elif line.startswith('EDGE_WEIGHT_SECTION'):
#             edge_weight_section = True
#             continue
#         elif line == 'EOF':
#             break
#         elif edge_weight_section:
#             numbers = list(map(int, line.split()))
#             edge_weights.extend(numbers)
    
#     # 根据 EDGE_WEIGHT_FORMAT 解析数据
#     if edge_weight_format == 'LOWER_DIAG_ROW':
#         idx = 0
#         for i in range(dimension):
#             for j in range(i + 1):
#                 distance_matrix[i][j] = edge_weights[idx]
#                 distance_matrix[j][i] = edge_weights[idx]  # 对称填充上三角部分
#                 idx += 1
#     else:
#         raise ValueError(f'Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}')
    
#     return distance_matrix

# 使用示例
filename = 'data/rat575.tsp'  # 请确保文件路径正确
data = read_tsp_file(filename)
distance_matrix = data['distance_matrix']

# 测试
dfwa_tsp = DFWA_TSP(distance_matrix, num_fireworks=15, nspk_factor=1000, mutation_factor=20, min_amp=5, max_amp=250, max_iter=1000)

# 运行优化
best_tour, best_distance = dfwa_tsp.optimize()

print("最佳路径:", best_tour)
print("最佳路径长度:", best_distance)



import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 초기 개체군 생성 함수
def initialize(num_blocks, num_population):
    population = []
    for i in range(num_population):
        sequence = np.random.choice(num_blocks, num_blocks, replace=False)
        population.append(sequence)
    return population

# 적합도 계산 함수
def calculate_fitness(blocks, sequence):
    if isinstance(sequence, list):
        sequence = np.array(sequence)

    num_of_blocks = blocks.shape[0]
    num_of_process = blocks.shape[1]
    temp = np.zeros((num_of_blocks + 1, num_of_process + 1))
    for i in range(1, num_of_blocks + 1):
        for j in range(1, num_of_process + 1):
            if temp[i - 1, j] > temp[i, j - 1]:
                temp[i, j] = temp[i - 1, j] + blocks[sequence[i - 1], j - 1]
            else:
                temp[i, j] = temp[i, j - 1] + blocks[sequence[i - 1], j - 1]
    C_max = temp[num_of_blocks, num_of_process]

    return C_max

# 선택 함수 (selection): 엘리트와 룰렛 선택 방식
def selection(data, population, _num_elite, _num_roulette, _lower_bound = None):
    fitness = []
    for i in range(len(population)):
        fitness.append(calculate_fitness(data, population[i]))

    # Fitness 값을 기준으로 정렬된 index를 얻음 (작을수록 좋은 fit)
    rank = np.argsort(fitness)

    # 상위 엘리트 개체 10개 선택
    elite = [copy.deepcopy(population[i]) for i in rank[:_num_elite]]

    # 적합도 값에 따라 확률을 계산
    if _lower_bound is not None:
        inverse_fitness = [1/(f - _lower_bound) for f in fitness]
    else:
        inverse_fitness = [1/f for f in fitness]

    # Roulette wheel selection을 위해 역수를 합계로 나누어 확률 계산
    selection_probs = [p/sum(inverse_fitness) for p in inverse_fitness]

    # 5개 최상위 적합도를 갖는 5개 개체의 적합도와 염색체 출력
    print("[Top 5 Elites]")
    for i in range(5):
        print(
            f"Elite {i + 1}: Fitness = {round(fitness[rank[i]], 3)}, Chromosome = {[round(ch, 2) for ch in population[rank[i]][:10]]}... , Probability = {round(selection_probs[rank[i]], 5)}")

    # Roulette wheel selection을 통해 추가로 10개 선택
    selected = np.random.choice(len(population), _num_roulette, p=selection_probs, replace=True)

    # Elite 개체와 선택된 개체를 결합하여 새로운 population 구성
    parents = elite + [copy.deepcopy(population[s]) for s in selected]

    return parents, min(fitness), np.copy(population[rank[0]])


# 돌연변이 함수 정의

def swap_mutation(chromosome):
    chrom = copy.deepcopy(chromosome)
    index = np.random.choice(len(chromosome), size=2, replace=False)  # 서로 다른 두 index 선택
    left = min(index)
    right = max(index)
    chrom[left] = chromosome[right]
    chrom[right] = chromosome[left]
    return chrom.copy()

def swap_neighbor_mutation(chromosome):
    chrom = copy.deepcopy(chromosome)
    index = np.random.choice(len(chromosome)-1)  # 100개의 block이 있다면, 0~98 범위에서 하나의 index 선정
    chrom[index] = chromosome[index+1]
    chrom[index+1] = chromosome[index]
    return chrom.copy()

def PMXcrossover(p1, p2, length=3):
    child1 = p1.tolist().copy()
    child2 = p2.tolist().copy()

    indices = np.random.choice(len(p1)-length) # 100개의 블록이 있다면, 0~94의 범위에서 추출
    slice_1 = p1[indices:indices+length].tolist().copy()
    slice_2 = p2[indices:indices+length].tolist().copy()
    pool = copy.deepcopy(slice_1)
    for s in slice_2:
        if s not in pool:
            pool.append(s)
    map1 = list()
    map2 = list()
    for elem in child1:
        if elem in pool:
            map1.append(elem)
    for elem in child2:
        if elem in pool:
            map2.append(elem)
    for idx, c in enumerate(child1):
        if c in pool:
            child1[idx] = map2.pop(0)
    for idx, c in enumerate(child2):
        if c in pool:
            child2[idx] = map1.pop(0)
    return np.array(child1), np.array(child2)


def reproduction(parents, _num_population, _num_elite, _p_crossover=0.9, _p_mutation=0.1, _crossover_length = 3):
    new_population = list()  # 초기 부모 리스트에 Elite 10과 선택된 10 포함

    # Elite 개체는 new_population에 그대로 남김
    elite = copy.deepcopy(parents[:_num_elite])
    new_population.extend(elite)

    while len(new_population) < _num_population:
        # 부모 중에서 두 개체 무작위 선택
        parents_indices = np.random.choice(len(parents), 2, replace=False)
        p1 = copy.deepcopy(parents[parents_indices[0]])
        p2 = copy.deepcopy(parents[parents_indices[1]])

        # Crossover 수행 여부
        if np.random.rand() < _p_crossover:
            child1, child2 = PMXcrossover(p1, p2, length=_crossover_length)
        else:
            child1, child2 = p1, p2  # crossover가 없으면 원본 유지

        # Mutation 수행 여부
        if np.random.rand() < _p_mutation:
            child1 = swap_neighbor_mutation(child1)
        if np.random.rand() < _p_mutation:
            child2 = swap_mutation(child2)

        # New population에 추가
        new_population.extend([child1, child2])

    # 만약 population 크기가 초과한 경우 자르기
    return new_population[:_num_population]

# 유전 알고리즘 실행 함수
def run_GA(_num_blocks, _num_population=10, _num_elite=20, _num_generation=1000, _p_crossover=0.9, _p_mutation=0.7):
    _num_roulette = _num_population - _num_elite
    generations = 0

    population = initialize(_num_blocks, _num_population) # 초기 개체군 생성
    top_fitness = 0.0
    top_fitness_record = []
    top_individual = None

    while generations < _num_generation:
    # while top_fitness <= -1.0:
        generations += 1
        print('-' * 15 + ' Generation ' + str(generations) + ' ' + '-' * 15)
        parents, top_fitness, top_individual = selection(data, population,
                                                         _num_elite=_num_elite,
                                                         _num_roulette=_num_roulette,
                                                         _lower_bound = 5493)
        population = reproduction(parents, _num_population, _num_elite=2,
                                  _p_crossover=0.7, _p_mutation=0.9, _crossover_length = 3)
        top_fitness_record.append(top_fitness)
        print('Top Individual:', [round(t, 2) for t in top_individual[:10]],"...", round(top_fitness,3))

    return population, np.copy(top_individual), top_fitness_record

if __name__ == "__main__":

    # 데이터 세트 정의
    data = pd.read_csv('Taillard_1.csv', header=None)
    data = data.to_numpy()

    seed = 0
    # 랜덤 시드 설정 (재현성을 위해)
    np.random.seed(seed)

    # 알고리즘 실행
    population, top_individual, top_fitness_record = run_GA(_num_blocks=100,
                                                            _num_population=100,
                                                            _num_elite=20,
                                                            _num_generation=500,
                                                            _p_crossover=1.0,
                                                            _p_mutation=1.0)

    # 결과 표시
    print("-" * 15, "Result", "-" * 15)
    for idx, p in enumerate(population):
        f = calculate_fitness(data, p)
        print(
            f"Individual {idx + 1}: Fitness = {round(f, 3)}, Chromosome = {[round(ch, 2) for ch in p]}")

    # 결과 그래프 생성
    plt.figure()
    plt.plot(top_fitness_record, c='black', label='Fitness')
    plt.title('Fitness(seed={0}, swap neighbor mutation, length=5)'.format(seed))
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.show()
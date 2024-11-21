from environment.Process import *
from environment.Source import Source
from environment.Sink import Sink
from environment.Part import *
from environment.Buffer import *
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from visualization.Gantt import *
from visualization.GUI import *
from cfg_local import Configure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import copy
from MIO_PFSP import get_MIO_individual, get_std_individual
from run_GA_solution import run_simulation
from GA import initialize, swap_neighbor_mutation, swap_mutation, PMXcrossover, reproduction

def calculate_fitness(_data, _individual):
    makespan = run_simulation(_data,
                              3,
                              seq = _individual,
                              show_gantt=False,
                              record_wip=False,
                              save_wip=False)
    return makespan


# 선택 함수 (selection): 엘리트와 룰렛 선택 방식
def selection(_data, population, _num_elite, _num_roulette, _lower_bound=None):
    fitness = []
    for i in range(len(population)):
        fitness.append(calculate_fitness(_data, population[i]))

    # Fitness 값을 기준으로 정렬된 index를 얻음 (작을수록 좋은 fit)
    rank = np.argsort(fitness)

    # 상위 엘리트 개체 10개 선택
    elite = [copy.deepcopy(population[i]) for i in rank[:_num_elite]]

    # 적합도 값에 따라 확률을 계산
    if _lower_bound is not None:
        inverse_fitness = [1 / (f - _lower_bound) for f in fitness]
    else:
        inverse_fitness = [1 / f for f in fitness]

    # Roulette wheel selection을 위해 역수를 합계로 나누어 확률 계산
    selection_probs = [p / sum(inverse_fitness) for p in inverse_fitness]

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


# 유전 알고리즘 실행 함수
def run_GA(_data, _num_blocks, _num_population=10, _num_elite=20,
           _num_generation=200, _p_crossover=0.9, _p_mutation=0.7, _mio=None):
    _num_roulette = _num_population - _num_elite
    generations = 0
    population = initialize(_num_blocks, _num_population)  # 초기 개체군 생성
    # mio = get_MIO_individual(_data)
    # population[0] = copy.deepcopy(np.array(mio))
    top_fitness = 0.0
    top_fitness_record = []
    top_individual = None

    while generations < _num_generation:
        # while top_fitness <= -1.0:
        if _mio is not None:
            population[-1] = np.array(copy.deepcopy(_mio))
        generations += 1
        print('-' * 15 + ' Generation ' + str(generations) + ' ' + '-' * 15)
        parents, top_fitness, top_individual = selection(_data, population,
                                                         _num_elite=_num_elite,
                                                         _num_roulette=_num_roulette,
                                                         _lower_bound=5500)
        population = reproduction(parents, _num_population, _num_elite=2,
                                  _p_crossover=0.7, _p_mutation=0.9, _crossover_length=3)
        top_fitness_record.append(top_fitness)
        print('Top Individual:', [round(t, 2) for t in top_individual[:10]], "...", round(top_fitness, 3))

    return population, np.copy(top_individual), top_fitness_record


if __name__ == '__main__':
    import time
    import csv
    from datetime import datetime

    start_time = time.time()

    now = datetime.now()
    subfix = now.strftime('%Y-%m-%d-%H-%M-%S')
    np.random.seed(2)
    num_generation = 100

    mio = get_MIO_individual('data\\data_Taillard.json')
    std = get_std_individual('data\\data_Taillard.json')
    # 알고리즘 실행
    mio_population, mio_top_individual, mio_top_fitness_record = run_GA('data\\data_Taillard.json',
                                                                        _num_blocks=100,
                                                                        _num_population=100,
                                                                        _num_elite=5,
                                                                        _num_generation=num_generation,
                                                                        _p_crossover=0.9,
                                                                        _p_mutation=0.9,
                                                                        _mio=mio)

    basic_population, basic_top_individual, basic_top_fitness_record = run_GA('data\\data_Taillard.json',
                                                                              _num_blocks=100,
                                                                              _num_population=100,
                                                                              _num_elite=5,
                                                                              _num_generation=num_generation,
                                                                              _p_crossover=0.9,
                                                                              _p_mutation=0.9)

    std_population, std_top_individual, std_top_fitness_record = run_GA('data\\data_Taillard.json',
                                                                              _num_blocks=100,
                                                                              _num_population=100,
                                                                              _num_elite=5,
                                                                              _num_generation=num_generation,
                                                                              _p_crossover=0.9,
                                                                              _p_mutation=0.9,
                                                                        _mio=std)
    finish_time = time.time()

    makespan = run_simulation('data\\data_Taillard.json', 2,
                                                      mio, show_gantt=False,
                              record_wip=False,
                              save_wip=False)
    mio_makespan = run_simulation('data\\data_Taillard.json', 2,
                                                      mio_top_individual, show_gantt=False,
                              record_wip=False,
                              save_wip=False)
    basic_makespan = run_simulation('data\\data_Taillard.json', 2,
                                                      basic_top_individual,show_gantt=False,
                              record_wip=False,
                              save_wip=False)
    std_makespan = run_simulation('data\\data_Taillard.json', 2,
                                                      std_top_individual, show_gantt=False,
                              record_wip=False,
                              save_wip=False)
    print("Time:", finish_time - start_time)
    print("MIO Makespan:",makespan)
    print("MIO GA Best Makespan:",mio_makespan)
    print("Basic GA Best Makespan:",basic_makespan)
    print("STD GA Best Makespan:",std_makespan)

    with open('GA_result(Basic)_{0}.csv'.format(subfix), 'w', newline='') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        # 데이터를 작성
        write.writerow(["Basic"] + basic_top_individual.tolist())
        write.writerow(["MIO"] + mio)
        write.writerow(["Rank"] + mio_top_individual.tolist())
        write.writerow(["Variance"] + std_top_individual.tolist())

    # 실제 데이터 사용 시 파일 읽기 등을 통해 `results` 데이터프레임을 채워주세요.
    # results = pd.read_csv('simulation_results.csv')
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 24

    plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams.update({'font.size': 14})
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(12, 6))

    plt.plot(basic_top_fitness_record, marker='o', linestyle='-', color='red',
             label='Basic GA')
    plt.plot(mio_top_fitness_record, marker='o', linestyle='-', color='blue',
             label='Rank Initialization')
    plt.plot(std_top_fitness_record, marker='o', linestyle='-', color='green',
             label='Variance Initialization')

    # 축 설정 및 레이블
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.title('Summary of Evolutionary Process')
    plt.legend(loc='upper right', ncol=1)
    plt.grid(True)
    plt.xticks(fontsize=18)
    # 그래프 표시
    # plt.show()
    plt.savefig('GA_result(Basic)_{0}.png'.format(subfix), dpi=300)
    print()

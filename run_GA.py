
from visualization.Gantt import *
import matplotlib.pyplot as plt
import copy
from MIO_PFSP import get_MIO_individual, get_std_individual
from run_GA_solution import run_PFSP_simulation, run_simulation
from GA import initialize, swap_neighbor_mutation, swap_mutation, PMXcrossover, reproduction
import os
import json


def calculate_fitness(filename, _individual):
    # makespan = run_PFSP_simulation(_data, _individual)
    makespan = run_simulation(filename, _individual)
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
    # print("[Top 5 Elites]")
    # for i in range(5):
    #     print(
    #         f"Elite {i + 1}: Fitness = {round(fitness[rank[i]], 3)}, Chromosome = {[round(ch, 2) for ch in population[rank[i]][:10]]}... , Probability = {round(selection_probs[rank[i]], 5)}")

    # Roulette wheel selection을 통해 추가로 10개 선택
    selected = np.random.choice(len(population), _num_roulette, p=selection_probs, replace=True)

    # Elite 개체와 선택된 개체를 결합하여 새로운 population 구성
    parents = elite + [copy.deepcopy(population[s]) for s in selected]

    return parents, min(fitness), np.copy(population[rank[0]])


# 유전 알고리즘 실행 함수
def run_GA(filename, _num_blocks, _num_population=10, _num_elite=20,
           _num_generation=200, _p_crossover=0.9, _p_mutation=0.7, _replace=None):
    _num_roulette = _num_population - _num_elite
    generations = 0
    population = initialize(_num_blocks, _num_population)  # 초기 개체군 생성
    top_fitness_record = []
    top_individual_record = []

    # _data = pd.read_csv(filename, header=None)
    # _data = _data.to_numpy()
    # if _data.shape[0] != _num_blocks:
    #     _data = _data.transpose()
    while generations < _num_generation:
        # while top_fitness <= -1.0:
        generations += 1
        print('-' * 15 + ' Generation ' + str(generations) + ' ' + '-' * 15)
        # parents, top_fitness, top_individual = selection(_data, population,
        #                                                  _num_elite=_num_elite,
        #                                                  _num_roulette=_num_roulette,
        #                                                  _lower_bound=5000)
        parents, top_fitness, top_individual = selection(filename, population,
                                                         _num_elite=_num_elite,
                                                         _num_roulette=_num_roulette,
                                                         _lower_bound=5000)
        if _replace is not None:
            parents[-1] = np.array(copy.deepcopy(_replace))
        population = reproduction(parents, _num_population, _num_elite=_num_elite,
                                  _p_crossover=_p_crossover, _p_mutation=_p_mutation, _crossover_length=3)
        top_fitness_record.append(top_fitness)
        top_individual_record.append(np.copy(top_individual))
        print('Top Individual:', [round(t, 2) for t in top_individual[:10]], "...", round(top_fitness, 3))

    return population, top_individual_record, top_fitness_record


if __name__ == '__main__':
    import time
    import csv
    from datetime import datetime

    now = datetime.now()
    subfix = now.strftime('%Y-%m-%d-%H-%M-%S')
    for i in range(5):
        # Create folder name using keyword and subfix

        data_name = 'Taillard_1'
        keyword = f"Validate_{data_name}"
        folder_name = f"{keyword}_{subfix}"
        data = data_name + '.csv'
        # keyword = "Convergence"
        num_blocks = 100
        num_population = 100
        num_generation = 500
        num_elite = 5
        p_crossover = 0.9
        p_mutation = 0.9
        random_seed = i
        np.random.seed(random_seed)


        config_dict = dict()
        config_dict['keyword'] = keyword
        config_dict['timecode'] = subfix
        config_dict['data_name'] = data_name
        config_dict['num_blocks'] = num_blocks
        config_dict['num_population'] = num_population
        config_dict['num_generation'] = num_generation
        config_dict['num_elite'] = num_elite
        config_dict['p_crossover'] = p_crossover
        config_dict['p_mutation'] = p_mutation
        config_dict['random_seed'] = random_seed
        # Create folder
        if not os.path.exists(os.path.join('result', folder_name)):
            os.makedirs(os.path.join('result', folder_name))
        if not os.path.exists(os.path.join('result', folder_name+f'\\seed{str(i)}')):
            os.makedirs(os.path.join('result', folder_name+f'\\seed{str(i)}'))
        # Define the path where the JSON will be saved
        config_json_path = os.path.join(folder_name, 'config.json')


        rank, rank_list = get_MIO_individual(f'data\\data_{data_name}.json')
        std, std_list = get_std_individual(f'data\\data_{data_name}.json')
        # 알고리즘 실행

        rank_start_time = time.time()
        rank_population, rank_top_individual_record, rank_top_fitness_record = run_GA('data\\'+data,
                                                                            _num_blocks=num_blocks,
                                                                            _num_population=num_population,
                                                                            _num_elite=num_elite,
                                                                            _num_generation=num_generation,
                                                                            _p_crossover=p_crossover,
                                                                            _p_mutation=p_mutation,
                                                                            _replace=rank)
        rank_finish_time = time.time()
        rank_time = rank_finish_time - rank_start_time
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Rank_individual.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in rank_top_individual_record:
                writer.writerow(array)
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Rank_makespan.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in rank_top_fitness_record:
                writer.writerow([array])

        basic_start_time = time.time()
        basic_population, basic_top_individual_record, basic_top_fitness_record = run_GA('data\\'+data,
                                                                            _num_blocks=num_blocks,
                                                                            _num_population=num_population,
                                                                            _num_elite=num_elite,
                                                                            _num_generation=num_generation,
                                                                            _p_crossover=p_crossover,
                                                                            _p_mutation=p_mutation)
        basic_finish_time = time.time()
        basic_time = basic_finish_time - basic_start_time
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Basic_individual.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in basic_top_individual_record:
                writer.writerow(array)
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Basic_makespan.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in basic_top_fitness_record:
                writer.writerow([array])

        std_start_time = time.time()
        std_population, std_top_individual_record, std_top_fitness_record = run_GA('data\\'+data,
                                                                                    _num_blocks=num_blocks,
                                                                                    _num_population=num_population,
                                                                                    _num_elite=num_elite,
                                                                                    _num_generation=num_generation,
                                                                                    _p_crossover=p_crossover,
                                                                                    _p_mutation=p_mutation,
                                                                                    _replace=std)
        std_finish_time = time.time()
        std_time = std_finish_time - std_start_time
        finish_time = time.time()
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Variance_individual.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in std_top_individual_record:
                writer.writerow(array)
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\Variance_makespan.csv", mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write each numpy array as a row in the CSV
            for array in std_top_fitness_record:
                writer.writerow([array])

        config_dict['Rank_time'] = rank_time
        config_dict['Rank_makespan'] = rank_top_fitness_record[-1]
        config_dict['Basic_time'] = basic_time
        config_dict['Basic_makespan'] = basic_top_fitness_record[-1]
        config_dict['Variance_time'] = std_time
        config_dict['Variance_makespan'] = std_top_fitness_record[-1]

        print("MIO GA Best Makespan:",rank_top_fitness_record[-1])
        print("Basic GA Best Makespan:",basic_top_fitness_record[-1])
        print("STD GA Best Makespan:",std_top_fitness_record[-1])


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

        plt.plot(basic_top_fitness_record, marker='o', linestyle='-', color='red', markersize=4,
                 label='Basic GA')
        plt.plot(rank_top_fitness_record, marker='o', linestyle='-', color='blue', markersize=4,
                 label='Rank Initialization')
        plt.plot(std_top_fitness_record, marker='o', linestyle='-', color='green', markersize=4,
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
        plt.savefig('result\\{0}\\seed{1}\\GA_result(Basic)_{2}_{3}.png'.format(folder_name, str(i), data_name, random_seed), dpi=300)
        print()


        # Save the dictionary as a JSON file
        with open(f"result\\{keyword}_{subfix}\\seed{str(i)}\\config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)

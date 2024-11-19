import json
import sys
import math

def get_rank_of_value(a, b):
    # 원본 리스트를 정렬하고, 중복 값을 제거
    sorted_unique_a = sorted(set(a))

    # 각 원소의 순위를 계산 (작은 값이 낮은 순위)
    rankings = {value: rank for rank, value in enumerate(sorted_unique_a, start=1)}

    # b의 순위를 반환
    return rankings.get(b, None)  # b가 없으면 None 반환

def get_MIO_individual(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # 각 공정 인덱스별 값을 저장할 리스트 초기화
    process_times_by_index = {}

    # 데이터를 순회하면서 processing_time[0]의 값들을 모은다.
    for job_key, job_data in data.items():
        processing_times = job_data["processing_time"][0]
        for index, time in enumerate(processing_times):
            if index not in process_times_by_index:
                process_times_by_index[index] = []
            process_times_by_index[index].append(time)

    # # 결과 출력
    # for index, times in process_times_by_index.items():
    #     print(f"Processing time at index {index}: {times}")

    for job_key, job_data in data.items():
        target = job_data["processing_time"][0]
        job_data["MIO Point"] = 0.0
        for idx, t in enumerate(target):
            rank =  get_rank_of_value(process_times_by_index[idx], t)

            job_data["MIO Point"] += rank
            # print(job_key,"의 {0}번째 공정의 시간이 {1}중에 {2}순위였으므로 MIO point가{3}만큼 증가하여{4}가 되었습니다.".format(
            #     idx, process_times_by_index[idx], rank, rank, job_data["MIO Point"]
            # ))
    # MIO Point 값에 따라 정렬
    sorted_jobs = sorted(data.items(), key=lambda x: (x[1]["MIO Point"], int(x[0].split('_')[-1])))
    sorted_keys = []
    # 정렬된 결과 출력
    for job_key, job_data in sorted_jobs:
        print(f"{job_key}: MIO Point = {job_data['MIO Point']}")
        sorted_keys.append(int(job_key.split('_')[-1]))

    return sorted_keys
def calculate_standard_deviation(data):
    # 평균 계산
    mean = sum(data) / len(data)

    # 각 데이터와 평균의 차이를 제곱하여 합산
    variance = sum((x - mean) ** 2 for x in data) / len(data)

    # 표준편차 계산
    std_dev = math.sqrt(variance)
    return std_dev
def get_std_individual(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    # 각 공정 인덱스별 값을 저장할 리스트 초기화
    process_times_by_index = {}

    # 데이터를 순회하면서 processing_time[0]의 값들을 모은다.
    for job_key, job_data in data.items():
        processing_times = job_data["processing_time"][0]
        std = calculate_standard_deviation(processing_times)

        for index, time in enumerate(processing_times):
            if index not in process_times_by_index:
                process_times_by_index[index] = []
            process_times_by_index[index].append(time)

    # # 결과 출력
    # for index, times in process_times_by_index.items():
    #     print(f"Processing time at index {index}: {times}")

    for job_key, job_data in data.items():
        target = job_data["processing_time"][0]
        job_data["MIO Point"] = 0.0
        for idx, t in enumerate(target):
            rank =  get_rank_of_value(process_times_by_index[idx], t)

            job_data["MIO Point"] += rank
            # print(job_key,"의 {0}번째 공정의 시간이 {1}중에 {2}순위였으므로 MIO point가{3}만큼 증가하여{4}가 되었습니다.".format(
            #     idx, process_times_by_index[idx], rank, rank, job_data["MIO Point"]
            # ))
    # MIO Point 값에 따라 정렬
    sorted_jobs = sorted(data.items(), key=lambda x: (x[1]["MIO Point"], int(x[0].split('_')[-1])))
    sorted_keys = []
    # 정렬된 결과 출력
    for job_key, job_data in sorted_jobs:
        print(f"{job_key}: MIO Point = {job_data['MIO Point']}")
        sorted_keys.append(int(job_key.split('_')[-1]))

    return sorted_keys


if __name__ == "__main__":
    from run_GA_solution import run_simulation
    mio = get_MIO_individual("data\\data_GA.json")
    print(mio)

    makespan = run_simulation('data\\data_GA.json',
                              2,mio,
                              False,
                              False,
                              False)
    print(makespan)
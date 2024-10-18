"""
Data Hierarchy:

1. Job
Jobs are defined by their required processes and order.
Job information is stored as a list, with elements corresponding to jobs
(e.g., job_list[0] = ['Process1', 'Process4', 'Process5']).

2. Process (Operation)

Processes specify the type of work, compatible resources, and processing times.
Each operation is uniquely identified by combining job and process, denoted as 'Operation'
(e.g., Op1_3 for the third process of Job 1).

3. Operation

Processing times for each compatible machine are specified for each operation.
Example: Op3_1 has a processing time of 3 on Machine 1 and 7 on Machine 7.
"""
from environment.Process import *
from environment.Source import Source
from environment.Sink import Sink
from environment.Part import *
from environment.Buffer import *
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from visualization.Gantt import *
from cfg_local import Configure
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    """ 모델 준비 """
    # 1. simpy 환경 생성
    env = simpy.Environment()
    # 2. 시뮬레이션 환경설정 객체 설정(경로, 실험 이름, job 개수 등 반복적으로 호출되는 것들)
    cfg = Configure()
    # 3. monitor 객체 생성
    monitor = Monitor(cfg.filepath)
    # 4. model 구성
    model = dict()
    for i in range(8):
        model['M' + str(i)] = Machine(env, i, 'M' + str(i))
    for i in range(5):
        model['FS_' + str(i)] = Process(cfg, env, 'FS_' + str(i), model, monitor, None)

    model['PMS'] = Process(cfg, env, 'PMS', model, monitor, None)
    model['Buffer'] = Buffer(cfg, env, 'Buffer', model, monitor)
    """ 변하지 않는 값 정의 """
    jobtype = JobType(idx=0, name='Part')

    work_fs = WorkType(idx=0, name='FS')
    work_pms = WorkType(idx=1, name='PMS')
    for i in range(5):
        o = OperationType(idx=i, name='FS_' + str(i),
                          process=model['FS_' + str(i)],
                          m_list=[model['M' + str(i)]])
        work_fs.add_operation_type(o)

    o_pms = OperationType(idx=0, name='PMS_',
                          process=model['PMS'],
                          m_list=[model['M5'], model['M6'], model['M7']])
    work_pms.add_operation_type(o_pms)

    jobtype.add_work_type(work_fs)
    jobtype.add_work_type(work_pms)

    """ 변하는 값 (processing time 등) 정의 """
    # 4-3. Source 객체 생성
    model['Source'] = Source(cfg, env, 'Source', model, monitor, job_type=jobtype, IAT=0, num_parts=10)
    # 4-4. sink 생성
    model['Sink'] = Sink(cfg, env, monitor)
    # model['Buffer'] = Buffer(cfg, env, monitor)

    # 5. 시뮬레이션 실행
    env.run(5000)
    # 6. 후처리를 위한 이벤트 로그 저장
    monitor.save_event()

    # In case of the situation where termination of the simulation greatly affects the machine utilization time,
    # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines

    machine_log = read_machine_log(cfg.filepath)
    # 7. 간트차트 출력
    gantt = Gantt(cfg, machine_log, len(machine_log), printmode=True, writemode=False)
    # gui = GUI(gantt)
    print()
    #
    # total_simulation_time = model['Sink'].last_arrival  # 끝나는 시간
    # # utilization_rate = (total_utilization_time / total_simulation_time) * 100
    # # 기계 통계 출력
    # print_machine_statistics(model, total_simulation_time)
    print()

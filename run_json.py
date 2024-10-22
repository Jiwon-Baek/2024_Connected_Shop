
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
    import json

    with open('data.json', 'r') as f:
        data = json.load(f)

    data = data['0']

    num_blocks = 20
    # num_blocks = len(data)
    num_shops = len(data['Job_0']['Work'])
    num_machines_list = data['Job_0']['num_machine'] # [ [1,1,1,1,1],[3] ]
    num_machines_by_shop = [sum(s) for s in data['Job_0']['num_machine']] # [5, 3]
    num_machines = sum(num_machines_by_shop)

    """ 모델 준비 """
    env = simpy.Environment()
    cfg = Configure()
    monitor = Monitor(cfg.filepath)
    model = dict()

    l=0
    for i, lst in enumerate(num_machines_list):
        # [1,1,1,1,1]
        shop_prefix = data['Job_0']['Work'][i] # 'FS'
        for j, p in enumerate(lst): # p = 1
            m_list = []
            for k in range(p):
                m_name = 'M' + str(l)
                model[m_name] = Machine(env, l, m_name)
                m_list.append(model[m_name])
                l+=1
            p_name = shop_prefix if len(lst) == 1 else shop_prefix + str(j)
            model[p_name] = Process(cfg, env, p_name, model, monitor, None,
                                                  _machine_list=m_list)

    model['Buffer'] = Buffer(cfg, env, 'Buffer', model, monitor)


    """ 변하지 않는 값 정의 """
    jobtype = JobType(idx=0, name='Part', preset=data)
    work_fs = WorkType(idx=0, name='FS')
    work_pms = WorkType(idx=1, name='PMS')
    for i in range(5):
        o = OperationType(idx=i, name='FS' + str(i),
                          process=model['FS' + str(i)],
                          m_list=[model['M' + str(i)]])
        work_fs.add_operation_type(o)

    o_pms = OperationType(idx=0, name='PMS',
                          process=model['PMS'],
                          m_list=[model['M5'], model['M6'], model['M7']])
    work_pms.add_operation_type(o_pms)

    jobtype.add_work_type(work_fs)
    jobtype.add_work_type(work_pms)

    """ 변하는 값 (processing time 등) 정의 """
    # 4-3. Source 객체 생성
    model['Source'] = Source(cfg, env, 'Source', model, monitor, job_type=jobtype, IAT=0, num_parts=num_blocks)
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
    unity_log = generate_unity_log(cfg.filepath, num_blocks)
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

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

def count_WIP(env, WIP_list, process):
    yield env.timeout(0.001)

    while len(WIP_list) < 1e5:
        WIP_list.append(process.WIP)
        yield env.timeout(1)

def run_PFSP_simulation(filename, sequence):
    blocks = pd.read_csv(filename, header=None)
    blocks = blocks.to_numpy()
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

def run_simulation(filepath, num_PM, seq=None,
                   source_capacity = float('inf'), buffer_capacity=float('inf'),
                   show_gantt=False, record_wip=False, save_wip=False):
    with open(filepath, 'r') as f:
        data = json.load(f)

    num_blocks = len(seq)
    num_shops = len(data['Job_0']['Work'])
    num_machines_list = data['Job_0']['num_machine'] # [ [1,1,1,1,1],[3] ]
    num_machines_by_shop = [sum(s) for s in data['Job_0']['num_machine']] # [5, 3]
    num_machines = sum(num_machines_by_shop)

    """ 모델 준비 """
    env = simpy.Environment()
    cfg = Configure()
    monitor = Monitor(cfg.filepath)
    model = dict()

    l = 0
    for i, lst in enumerate(num_machines_list):
        shop_prefix = data['Job_0']['Work'][i]  # 'FS'
        for j, p in enumerate(lst):  # p = 1
            m_list = []
            for k in range(p):
                m_name = 'M' + str(l)
                model[m_name] = Machine(env, l, m_name)
                m_list.append(model[m_name])
                l += 1
            p_name = shop_prefix if len(lst) == 1 else shop_prefix + str(j)
            model[p_name] = Process(cfg, env, p_name, model, monitor, None,
                                    _machine_list=m_list)

    model['Buffer'] = Buffer(cfg, env, 'Buffer', model, monitor, _capacity=buffer_capacity)

    """ 변하지 않는 값 정의 """
    jobtype = JobType(idx=0, name='Part', preset=data)
    work_fs = WorkType(idx=0, name='FS')
    work_pms = WorkType(idx=1, name='PMS')
    for i in range(num_machines_by_shop[0]):
        o = OperationType(idx=i, name='FS' + str(i),
                          process=model['FS' + str(i)],
                          m_list=[model['M' + str(i)]])
        work_fs.add_operation_type(o)

    if num_PM == 1:
        m_list = [model['M5']]
    elif num_PM == 2:
        m_list = [model['M5'], model['M6']]
    elif num_PM == 3:
        m_list = [model['M5'], model['M6'], model['M7']]
    elif num_PM == 4:
        m_list = [model['M5'], model['M6'], model['M7'], model['M8']]
    elif num_PM == 5:
        m_list = [model['M5'], model['M6'], model['M7'], model['M8'], model['M9']]
    else:
        m_list = None

    o_pms = OperationType(idx=0, name='PMS',
                          process=model['PMS'],
                          m_list=m_list)
    work_pms.add_operation_type(o_pms)
    jobtype.add_work_type(work_fs)
    jobtype.add_work_type(work_pms)

    """ 변하는 값 (processing time 등) 정의 """

    # 4-3. Source 객체 생성
    if seq is not None:
        model['Source'] = Source(cfg, env, 'Source', model, monitor,
                                 job_type=jobtype, IAT=0, num_parts=num_blocks,_seq=seq,
                                 capacity = source_capacity)
    else:
        model['Source'] = Source(cfg, env, 'Source', model, monitor, job_type=jobtype, IAT=0, num_parts=num_blocks)
    # 4-4. sink 생성
    model['Sink'] = Sink(cfg, env, monitor)

    if record_wip:
        WIP_source = list()
        WIP_buffer = list()
        env.process(count_WIP(env, WIP_source, model['Source']))
        env.process(count_WIP(env, WIP_buffer, model['Buffer']))

    # 5. 시뮬레이션 실행
    env.run(1e5)
    # 6. 후처리를 위한 이벤트 로그 저장
    # monitor.save_event()

    makespan = model['Sink'].last_arrival

    if show_gantt:
        machine_log = read_machine_log(cfg.filepath)
        # unity_log = generate_unity_log(cfg.filepath, num_blocks)
        # 7. 간트차트 출력
        gantt = Gantt(cfg, machine_log, len(machine_log), printmode=True, writemode=False)
        gui = GUI(gantt)

    if record_wip:
        if save_wip:
            plt.figure()
            plt.plot(WIP_source[:int(makespan)], c='blue', label='Source')
            plt.plot(WIP_buffer[:int(makespan)], c='red', label='Buffer')
            plt.legend()
            plt.ylim([0,100])
            plt.xlabel('Time')
            plt.ylabel('# of WIP')
            plt.title('Stock level / machine = (5,%d)' % num_PM)
            plt.savefig(filepath.split('.')[0] + '_' + str(num_PM) + '_' + str(source_capacity) + '_' + str(buffer_capacity) + '_WIP.png')
            plt.clf()
            plt.close()
            if isinstance(makespan, float):
                makespan = int(makespan)


        return makespan, copy.deepcopy(WIP_source[:int(makespan)]), copy.deepcopy(WIP_buffer[:int(makespan)])
    else:
        return makespan

if __name__ == '__main__':
    seq = np.random.choice(100, 100, replace=False)
    # makespan = run_simulation('data\\data_GA.json',
    #                           2,seq,
    #                           False,
    #                           False,
    #                           False)

    optimal = [58, 90, 42, 31, 50, 74, 39, 38, 29, 62, 40, 15, 86, 63, 13, 44, 76, 75, 49, 34, 92, 70, 71, 53, 78, 21, 9, 16, 85, 46, 81, 96, 33, 24, 4, 41, 25, 18, 28, 67, 45, 35, 83, 17, 43, 19, 3, 69, 12, 2, 64, 79, 73, 37, 7, 48, 97, 52, 47, 20, 60, 82, 65, 94, 51, 87, 27, 72, 26, 23, 11, 8, 100, 98, 77, 54, 6, 10, 56, 93, 91, 57, 22, 55, 59, 99, 80, 68, 95, 88, 89, 36, 66, 61, 32, 14, 30, 5, 1, 84]
    optimal = [job - 1 for job in optimal]

    for n in range(1,4):
        for s in range(10,31,10):
            makespan = run_simulation('data\\data_GA.json',
                                      n,
                                      seq,
                                      s,
                                      float('inf'),
                                      False,
                                      True,
                                      True)
            print("Machine:",n,"\tSource:",s,"\tMakespan:",makespan[0])
    print()

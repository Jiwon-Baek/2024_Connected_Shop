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

    while len(WIP_list) < 5e5:
        WIP_list.append(process.WIP)
        yield env.timeout(1)

def run_simulation(filepath, num_PM, show_gantt=False, save_wip=False):
    with open(filepath, 'r') as f:
        data = json.load(f)

    data = data['0']

    num_blocks = 4
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
    model['Source'] = Source(cfg, env, 'Source', model, monitor, job_type=jobtype, IAT=0, num_parts=num_blocks,
                             seq=[0,1,2,3])
    # 4-4. sink 생성
    model['Sink'] = Sink(cfg, env, monitor)

    WIP_source = list()
    WIP_buffer = list()
    env.process(count_WIP(env, WIP_source, model['Source']))
    env.process(count_WIP(env, WIP_buffer, model['Buffer']))

    # 5. 시뮬레이션 실행
    env.run(1e6)
    # 6. 후처리를 위한 이벤트 로그 저장
    monitor.save_event()

    makespan = model['Sink'].last_arrival

    if show_gantt:
        machine_log = read_machine_log(cfg.filepath)
        # unity_log = generate_unity_log(cfg.filepath, num_blocks)
        # 7. 간트차트 출력
        gantt = Gantt(cfg, machine_log, len(machine_log), printmode=True, writemode=False)
        gui = GUI(gantt)

    if save_wip:
        plt.figure()
        plt.plot(WIP_source[:makespan], c='blue', label='Source')
        plt.plot(WIP_buffer[:makespan], c='red', label='Buffer')
        plt.legend()
        plt.ylim([0,40])
        plt.xlabel('Time')
        plt.ylabel('# of WIP')
        plt.title('Stock level / machine = (5,%d)' % num_PM)
        plt.savefig(filepath.split('.')[0] + '_' + str(num_PM) + '_WIP.png')
        plt.clf()
        plt.close()
    return makespan, copy.deepcopy(WIP_source[:makespan]), copy.deepcopy(WIP_buffer[:makespan])


if __name__ == '__main__':

    # 결과 저장을 위한 데이터프레임 초기화
    results = pd.DataFrame(
        columns=['lambda', 'pt_mean', 'pt_sigma', 'num_machine', 'makespan', 'mean_wip_source', 'mean_wip_buffer'])
    for lmbda in [50, 60, 70, 80, 90]:
        for pt_mean in [140, 160, 180, 200, 220, 240]:
            for pt_sigma in [10, 20, 40, 80]:
                # for pt_mean in [140, 160, 180, 200, 220, 240]:
        #     for pt_sigma in [10, 20, 40, 80]:
            # for lmbda in [50, 60, 70, 80, 90]:
    #     for pt_mean in [140, 150, 160, 170, 180, 190]:
    #         for pt_sigma in [5, 10, 15, 20, 25, 30]:
                print('-'*30)
                for num_PM in range(1,6):
                    makespan, wip_source, wip_buffer = run_simulation('data\\Taillard_{0}_{1}_{2}.json'.format(lmbda, pt_mean, pt_sigma), num_PM, False, True)
                    mean_wip_source = np.array(wip_source).mean().round(2)
                    mean_wip_buffer = np.array(wip_buffer).mean().round(2)
                    print("Makespan: ", makespan, "\tWIP(Source): ", mean_wip_source, "\tWIP(Buffer): ", mean_wip_buffer)

                    # 새로운 행을 데이터프레임으로 생성
                    new_row = pd.DataFrame([{
                        'lambda': lmbda,
                        'pt_mean': pt_mean,
                        'pt_sigma': pt_sigma,
                        'num_machine': num_PM,
                        'makespan': makespan,
                        'mean_wip_source': mean_wip_source,
                        'mean_wip_buffer': mean_wip_buffer
                    }])

                    # 기존 데이터프레임과 새 행을 합침
                    results = pd.concat([results, new_row], ignore_index=True)

    # 결과를 CSV 파일로 저장
    results.to_csv('simulation_results.csv', index=False)

    print("실험이 완료되었고 결과가 simulation_results.csv에 저장되었습니다.")
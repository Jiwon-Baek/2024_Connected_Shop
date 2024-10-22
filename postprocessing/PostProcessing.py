import pandas as pd
from collections import OrderedDict, defaultdict

# TODO : read_machine_log로 이름 변경
def read_machine_log(_filepath):
    """
    기계 로그 파일을 읽어와 작업의 시작 및 완료 이벤트를 필터링하고, 각 기계별로 작업의 시작과 완료 시간을 계산하여 새 CSV 파일로 저장.

    ### Args:
        - `_filepath (str)`: 기계 로그를 포함한 CSV 파일의 경로.

    ### Returns:
        - `pd.DataFrame`: 기계별로 작업의 시작과 완료 시간 및 처리 시간을 포함하는 데이터프레임.
    """
    df = pd.read_csv(_filepath)
    df = df.drop(df.columns[0], axis=1)
    # Filter 'Started' and 'Finished' events
    df_started = df[df['Event'] == 'Started'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)
    df_finished = df[df['Event'] == 'Finished'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)

    machine_list = df['Machine'].unique()
    machine_start = []
    machine_finish = []
    for i in range(len(machine_list)):
        machine_start.append(df_started[(df_started['Machine'] == machine_list[i])])
        machine_finish.append(df_finished[(df_finished['Machine'] == machine_list[i])])

        machine_start[i].reset_index(drop=True, inplace=True)
        machine_finish[i].reset_index(drop=True, inplace=True)
    data = []

    for i in range(len(machine_list)):
        for j in range(len(machine_finish[i])):
            temp = {'Machine': machine_list[i],
                    'Job': machine_start[i].loc[j, 'Part'],
                    'Start': int(machine_start[i].iloc[j, 0]),
                    'Finish': int(machine_finish[i].iloc[j, 0]),
                    'Delta': int(machine_finish[i].iloc[j, 0] - machine_start[i].iloc[j, 0])}
            data.append(temp)

    data = pd.DataFrame(data)
    data = data.sort_values(by=['Start'])
    data.reset_index(drop=True, inplace=True)
    data.to_csv(_filepath.split('.')[0]+'_machine_log.csv')




    return data

def generate_unity_log(_filepath, num_blocks=10):
    data = pd.read_csv(_filepath)
    data = data.drop(data.columns[0], axis=1)
    with open(_filepath.split('.')[0]+'_Unity_log.csv', 'w') as f:
        # column명 작성
        f.write('Job,Machine,Move,Start,Finish\n')
    process = ['FS0', 'FS1', 'FS2', 'FS3', 'FS4', 'Buffer', 'PMS']
    # process = ['Source','FS0', 'FS1', 'FS2', 'FS3', 'FS4', 'Buffer', 'PMS']
    process_map = {
        # 'Source':'S',
                   'M0': 0,
                   'M1': 1,
                   'M2': 2,
                   'M3': 3,
                   'M4': 4,
                   'Buffer': 5,
                   'M5': 6,
                   'M6': 7,
                   'M7': 8}

    for i in range(num_blocks):
        for p in process:
            print(p)
            j = data[(data.Part == 'Part_' + str(i)) & (data.Process == p)]
            machine = j[j.Event == 'Started']['Machine'].values[0]
            mapped = process_map[machine]
            move = j[j.Event == 'Started']['Time'].values[0]
            start = move + 0.5
            finish = j[j.Event == 'Finished']['Time'].values[0]

            with open(_filepath.split('.')[0]+'_Unity_log.csv', 'a') as f:
                f.write('%s,%s,%.2f,%.2f,%.2f\n' % (str(i), mapped, move, start, finish))

if __name__ == '__main__':
    from visualization.Gantt import Gantt
    from cfg_local import Configure

    cfg = Configure()
    machine_log = read_machine_log('../result/2024-10-22-18-13-13.csv')
    generate_unity_log('../result/2024-10-22-18-13-13.csv', 100)
    # 7. 간트차트 출력
    gantt = Gantt(cfg, machine_log, len(machine_log), printmode=True, writemode=False)
    # gui = GUI(gantt)
    print()
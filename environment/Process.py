import simpy
import random
random.seed(42)
import sys
# 파이썬 라이브러리가 설치되어 있는 디렉터리를 확인할 수 있다.
import os
import numpy as np


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# sys.path: 파이썬 라이브러리가 설치되어 있는 디렉터리를 확인할 수 있다.
# __file__: 현재 실행중인 스크립트의 경로를 담고 있는 특수 변수.
# abspath: 절대 경로를 반환 / dirname: 주어진 경로의 디렉토리 이름 반환. / dirname 3번 사용해서 세단계 상위 디렉토리 경로 반환
# sys.path:에 새로운 경로 추가: 이로써 해당 경로에 있는 파이썬 모듈 import 할 수 있게 됨.

from environment.Monitor import *

# from config import *
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


class Process(object):
    def __init__(self, _cfg, _env, _name, _model, _monitor, _transportation_times, _machine_list = None):
        """
        프로세스 객체 초기화 메서드
        :: 주어진 매개변수들을 사용하여 프로세스 객체를 초기화함. 시뮬레이션 환경, 프로세스 이름, 모델, 모니터 객체, 설정 파일 및 운송 시간 데이터를 설정

        ### Args:
            - `_cfg (object)`: 설정 데이터 및 구성 정보를 포함하는 객체.
            - `_env (object)`: 시뮬레이션 환경 객체.
            - `_name (str)`: 프로세스의 이름.
            - `_model (object)`: 모델 데이터 객체.
            - `_monitor (object)`: 이벤트 기록을 위한 모니터 객체.
            - `_transportation_times (list)`: 운송 시간 데이터를 포함하는 리스트.
        """
        # 프로세스 객체의 초기화. 필수 입력 데이터로 시뮬레이션 환경, 공정 이름, 모델, 모니터 객체, 설정파일.
        # input data
        self.env = _env
        self.name = _name  # 해당 프로세스의 이름
        self.model = _model
        # 모델 데이터
        self.monitor = _monitor
        # 이벤트 기록을 위한 모니터 객체
        self.cfg = _cfg
        # 설정 데이터, 구성 정보를 담고 있음.
        self.tpt = _transportation_times
        self.in_buffer = simpy.FilterStore(_env, capacity=float('inf')) # 입력 버퍼
        self.out_buffer = simpy.FilterStore(_env, capacity=float('inf')) # 출력 버퍼
        # simpy.FilterStore: 필터링 기능을 제공하는 저장소, 특정 조건 만족하는 항목만 처리. / inf: 버퍼크기 무한

        self.work_waiting = [self.env.event() for i in range(self.cfg.num_job)]

        if _machine_list is not None:
            self.machine_list = _machine_list
            self.availability = simpy.FilterStore(_env, capacity=len(self.machine_list))
        else:
            self.machine_list, self.machine_available = None, None

        _env.process(self.run())
        _env.process(self.to_next_process())

    # Class Description
    # Process.run() : 작업 하나를 골라서 env.process()에 work를 추가시킴
    # Process.work() : machine의 사용권을 얻고, timeout이 일어나는 부분
    # Process.to_next_process() : 다음 process로 전달

    def work(self, part, machine, pt):
        """
        특정 부품에 대한 작업을 수행하는 메서드
        :: 선택된 기계에서 부품을 처리하는 로직을 담고 있으며, 작업 시작과 완료 시 모니터에 이벤트를 기록
        :: 작업을 시작하기 전에 선행 조건이 충족되었는지 확인하고, 기계의 사용 가능 상태를 체크한 후 작업을 수행
        :: 작업이 완료되면 기계의 상태를 업데이트하고 부품을 출력 버퍼로 이동시킴

        ### Args:
            - `part (object)`: 현재 작업 중인 부품 객체. 이 객체는 `op` 리스트를 통해 작업 관련 정보를 가지고 있음
            - `machine (object)`: 부품을 처리할 기계 객체. 기계의 상태와 큐를 관리
            - `pt (int)`: 공정에 소요되는 시간 (단위: 시간). 작업을 수행하는 데 필요한 시간을 나타냄

        ### Yields:
            - `None`: 이 메서드는 시뮬레이션의 이벤트를 생성하여 다른 작업이 이 메서드의 완료를 기다리도록 함
            - `yield`를 사용하여 시뮬레이션의 이벤트를 처리함
        """
        # 밑에 run 함수에서 불러옴.
        # 특정 부품에 대한 실제 작업을 수행. 선택된 기계에서 부품을 처리하는 로직을 담고 있으며, 작업시작과 완료 시 모니터에 이벤트 기록.
        # 작업 선행조건 만족시까지 대기 후, 기계 사용가능 여부 확인하고 작업 시작, 완료되면 사용된 기계 상태 업데이트 후 부품을 출력 버퍼로 이동.
        # 1. Check if former operations are all finished & requirements are satisfied
        operation = part.op_list[part.current_work][part.step[part.current_work]]

        # 작업이 큐에 들어가는 시간 기록
        operation.queue_entry_time = self.env.now
        yield operation.requirements
        # print("operation %s 의 모든 선행조건이 완료되었습니다." % operation.name)
        # 선행조건이 충족되길 기다림.
        yield machine.availability.put('using')
        print("%d\t%s 가 machine %s 의 사용권 획득" % (self.env.now, operation.name, machine.name))

        # 아직 원래 process임
        if part.process is not None: # Sink 에서 출발할 때에는 part.process 가 None 일 수 있음
            print("%d\t기존에 part %s의 process는 %s 입니다." % (self.env.now,part.name, part.process.name))
            yield part.process.availability.get()
            print("%d\t이전 process인 %s의 사용권 반환" % (self.env.now,part.process.name))
            if part.process.name == 'Buffer':
                print('break')
        part.set_process(self)
        print("%d\t%s 의 process를 %s 로 바꿨습니다" % (self.env.now,part.name, part.process.name))

        # 작업이 큐에서 나오는 시간 기록 및 대기 시간 계산
        wait_end_time = self.env.now
        machine.operation_count += 1  # 추가: 처리한 operation 수 증가
        wait_time = wait_end_time - operation.queue_entry_time
        # print(f"{machine.name}의 {machine.operation_count}번째 wait time은 {wait_time}이다.")
        machine.total_waiting_time += wait_time

        # 2. Update machine status
        # 다른 class object들에게 알려주기 위해 상태와 가장 빠른 종료시간을 기록
        # TODO : work()를 발생시킬 때, 해당 machine의 내부 변수에
        #  이만큼의 operation이 대기중이라는 사실을 기록해서 다른 class에서도 참조하도록 해야 하지 않을까?

        machine.status = 'Working'
        # 기계 상태 업데이트
        machine.turn_idle = self.env.now + pt
        # 작업 완료 예정시간 계산 / 이제 일을 시작하니까 지금시점부터 pt 더하기
        machine.queue.remove(operation)
        # 기계 큐에서 작업 제거 / 지금 일을 하니까.

        # 3. Proceed & Record through console
        self.monitor.record(self.env.now, self.name, machine=machine.name,
                            part_name=part.name, event="Started") # 작업 시작 기록

        if self.cfg.CONSOLE_MODE:
            monitor_console(self.env.now, part, self.cfg.OBJECT, "Started on")

        yield self.env.timeout(pt)
        print("%d\t%s 가 machine %s 에서 %d 만큼 작업되어 현재 시간이 %d가 됨"
              % (self.env.now,part.name, machine.name, pt, self.env.now))
        self.monitor.record(self.env.now, self.name, machine=machine.name,
                            part_name=part.name, event="Finished") # 작업 완료 기록
        if self.cfg.CONSOLE_MODE:
            monitor_console(self.env.now, part, self.cfg.OBJECT, "Finished on")

        machine.util_time += pt
        machine.workingtime_log.append(pt)
        machine.waiting_time += self.env.now - part.start_waiting_time

        # 4. Send(route) to the out_part queue for routing and update the machine availability
        yield self.out_buffer.put(part)
        yield machine.availability.get()
        print("%d\t%s가 machine %s의 사용권 반환" % (self.env.now,part.name, machine.name))

        # 기계 사용 상태 해체
        machine.status = 'Idle'

    def run(self):

        while True:
            ############### 1. Job의 결정
            # TODO : call agent for selecting a part
            part = yield self.in_buffer.get()
            part.start_waiting_time = self.env.now

            ############### 2. Machine의 결정
            # TODO : call agent for selecting a machine
            operation = part.op_list[part.current_work][part.step[part.current_work]]

            if len(operation.machine_available) > 1:  # 만약 여러 machine에서 작업 가능한 operation이라면
                machine, pt = self.heuristic_FJSP(operation)
            else:  # 만약 단일 기계에서만 작업 가능한 operation이라면
                machine, pt = self.heuristic_JSSP(operation)

            # 결정된 사항을 기록해 둠
            part.machine = machine
            operation.machine_determined = machine
            operation.process_time_determined = pt
            machine.queue.append(operation)
            # 선택된 기계의 큐에 작업 추가.
            eti = machine.expected_turn_idle(self.env.now)
            # print('%d \t%s have %d operations in queue... turning idle at %d... \tfinish working at %d' %
            #     (self.env.now, machine.name, len(machine.queue), machine.turn_idle, eti))

            ############### 3. work() 인스턴스 생성 / 작업이 실제로 기계에서 작업시작.
            self.env.process(self.work(part, machine, pt))


    # region dispatching
    def heuristic_LIT(self, operation):
        """
        가장 적은 대기 시간(Least Idle Time)을 가진 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 대기 시간이 가장 짧은 기계를 선택.
        :: 대기 시간이 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택함

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함하고 있음

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체이며, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간
        """
        machine_list = operation.machine_available
        least_idle_time = min(len(m.queue) for m in machine_list)
        candidates = [m for m in machine_list if len(m.queue) == least_idle_time]
        least_idle_machine = random.choice(candidates)
        process_time = operation.process_time[machine_list.index(least_idle_machine)]
        return least_idle_machine, process_time

    def heuristic_LUT(self, operation):
        """
        가장 적은 사용 시간을 가진 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 가장 적은 총 사용 시간을 기록한 기계를 선택. 
        :: 사용 시간이 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택함

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함함

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간
        """
        machine_list = operation.machine_available
        least_util_time = min(m.util_time for m in machine_list)
        candidates = [m for m in machine_list if m.util_time == least_util_time]
        least_util_machine = random.choice(candidates)
        process_time = operation.process_time[machine_list.index(least_util_machine)]
        return least_util_machine, process_time

    def heuristic_SPT(self, operation):
        """
        가장 짧은 처리 시간을 가진 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 가장 짧은 처리 시간을 가진 기계를 선택. 
        :: 처리 시간이 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택함

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간
        """
        machine_list = operation.machine_available
        min_process_time = min(operation.process_time)
        candidates = [m for i, m in enumerate(machine_list) if operation.process_time[i] == min_process_time]
        selected_machine = random.choice(candidates)
        return selected_machine, min_process_time

    def heuristic_LPT(self, operation):
        """
        가장 긴 처리 시간을 가진 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 가장 긴 처리 시간을 가진 기계를 선택.
        :: 처리 시간이 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택.

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환 
            - `Machine`은 선택된 기계 객체, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간.
        """
        machine_list = operation.machine_available
        max_process_time = max(operation.process_time)
        candidates = [m for i, m in enumerate(machine_list) if operation.process_time[i] == max_process_time]
        selected_machine = random.choice(candidates)
        return selected_machine, max_process_time

    def heuristic_MOR(self, operation):
        """
        대기 작업이 가장 많은 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 현재 대기 중인 작업 수가 가장 많은 기계를 선택. 
        :: 대기 작업 수가 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택.

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함.

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간.
        """
        machine_list = operation.machine_available
        max_operations_remaining = max(len(m.queue) for m in machine_list)
        candidates = [m for m in machine_list if len(m.queue) == max_operations_remaining]
        most_ops_remaining_machine = random.choice(candidates)
        process_time = operation.process_time[machine_list.index(most_ops_remaining_machine)]
        return most_ops_remaining_machine, process_time

    def heuristic_LOR(self, operation):
        """
        대기 작업이 가장 적은 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 현재 대기 중인 작업 수가 가장 적은 기계를 선택. 
        :: 대기 작업 수가 같은 기계가 여러 개일 경우, 그 중 하나를 랜덤으로 선택.

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함하고 있습니다.

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간.
        """
        machine_list = operation.machine_available
        least_operations_remaining = min(len(m.queue) for m in machine_list)
        candidates = [m for m in machine_list if len(m.queue) == least_operations_remaining]
        least_ops_remaining_machine = random.choice(candidates)
        process_time = operation.process_time[machine_list.index(least_ops_remaining_machine)]
        return least_ops_remaining_machine, process_time

    # def heuristic_LIT(self, operation):
    #     machine_list = operation.machine_available
    #     least_idle_machine = min(machine_list, key=lambda m: m.expected_turn_idle())
    #     process_time = operation.process_time[machine_list.index(least_idle_machine)]
    #     return least_idle_machine, process_time
    #
    # # 2. LUT (Least Utilization Time)
    # def heuristic_LUT(self, operation):
    #     machine_list = operation.machine_available
    #     least_util_machine = min(machine_list, key=lambda m: m.util_time)
    #     process_time = operation.process_time[machine_list.index(least_util_machine)]
    #     return least_util_machine, process_time
    #
    # # 3. MOR (Most Operations Remaining): 머신 큐에 작업 가장 많이 있는 머신 우선.
    # def heuristic_MOR(self, operation):
    #     machine_list = operation.machine_available
    #     most_ops_remaining_machine = max(machine_list, key=lambda m: len(m.queue))
    #     process_time = operation.process_time[machine_list.index(most_ops_remaining_machine)]
    #     return most_ops_remaining_machine, process_time
    #
    # # 4. MWR (Most Waiting Resources)
    # def heuristic_MWR(self, operation):
    #     machine_list = operation.machine_available
    #     most_waiting_resources_machine = max(machine_list, key=lambda m: len(m.availability.items))
    #     process_time = operation.process_time[machine_list.index(most_waiting_resources_machine)]
    #     return most_waiting_resources_machine, process_time
    #
    # def heuristic_LOR(self, operation):
    #     machine_list = operation.machine_available
    #     least_ops_remaining_machine = min(machine_list, key=lambda m: len(m.queue))
    #     process_time = operation.process_time[machine_list.index(least_ops_remaining_machine)]
    #     return least_ops_remaining_machine, process_time
    #
    # def heuristic_LWR(self, operation):
    #     machine_list = operation.machine_available
    #     least_waiting_resources_machine = min(machine_list, key=lambda m: len(m.availability.items))
    #     process_time = operation.process_time[machine_list.index(least_waiting_resources_machine)]
    #     return least_waiting_resources_machine, process_time
    #
    # def heuristic_FJSP_SPT(self, operation):
    #     machine_list = operation.machine_available
    #     pt_list = operation.process_time
    #     pt_list_min_index = pt_list.index(min(pt_list))
    #     machine = machine_list[pt_list_min_index]
    #
    #     return machine, min(pt_list)
    #
    # def heuristic_FJSP_LPT(self, operation):
    #     machine_list = operation.machine_available
    #     pt_list = operation.process_time
    #     pt_list_max_index = pt_list.index(max(pt_list))
    #     machine = machine_list[pt_list_max_index]
    #
    #     return machine, max(pt_list)


    def heuristic_FJSP(self, operation):
        """
        기계의 남은 가동 시간이 가장 짧은 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업(`operation`)을 수행할 수 있는 기계 목록에서 현재 남아있는 가동 시간이 가장 짧은 기계를 선택. 
        :: 기계가 유휴 상태일 경우 즉시 선택, 유휴 상태가 아닌 기계 중에서는 남은 가동 시간이 가장 짧은 기계를 선택.

        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 가능한 기계 목록과 각 기계에 대한 공정 시간을 포함.

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체이며, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간.
        """
        # compatible machine list
        machine_list = operation.machine_available
        # 사용 가능한 기계 목록 가져옴.
        # compatible machine들의 현황을 파악해서 가장 idle한 machine을 지정

        # Remark : 그런데 학습에 따라서 현재 대기시간이 많이 남은 machine이더라도 그게 좋다고 판단되면 선택할 수 있어야 하지 않나?
        # TODO : 현재는 고의로 idle하게 machine을 남겨두는 것이 불가능함(non-delay) -> 수정 필요

        remaining_time = []
        # 기계 대기시간을 평가하여 가장 빠르게 사용가능한 기계 목록 가져옴.
        for m in machine_list:
            if m.status == 'Idle':
                remaining_time.append(0)
                # 기계 쉬는 중이면 즉시 선택
            else:
                remaining_time.append(m.expected_turn_idle(self.env.now) - self.env.now)
                # 그렇지 않으면 남은 가동시간 계산

                # remaining_time.append(m.turn_idle - self.env.now)
        # index of the machine with the least remaining time
        least_remaining = np.argmin(remaining_time)
        # 남은 가동 시간이 가장 짧은 기계를 선택. "index"
        # TODO : 만약에 argmin인 값이 둘 이상일 때 (예를 들면 idle한 machine이 둘 이상일 때) 어떤 것을 선택할지 결정해야 함

        machine = machine_list[least_remaining]

        # process time on the certain machine이 list로 주어진 경우 vs. 단일 값으로 주어진 경우
        if isinstance(operation.process_time, list):
            process_time = operation.process_time[least_remaining]
        elif isinstance(operation.process_time, np.ndarray):
            process_time = operation.process_time[least_remaining]
        else:
            process_time = operation.process_time
        return machine, process_time
    # endregion

    def heuristic_JSSP(self, operation):
        """
        주어진 작업(`operation`)이 수행될 수 있는 단일 기계를 선택하는 휴리스틱 함수.
        :: 주어진 작업에 대해 사용할 수 있는 기계가 하나만 존재할 경우, 그 기계를 선택하고 공정 시간을 반환. 기계 선택 및 공정 시간 결정은 작업에 명시된 대로 설정됨.
        
        ### Args:
            - `operation (Operation)`: 작업을 정의하는 `Operation` 객체. 이 객체는 사용 가능한 단일 기계와 그에 대한 공정 시간을 포함.

        ### Returns:
            - `(Machine, float)`: 선택된 기계와 해당 기계에서의 작업 시간을 반환. 
            - `Machine`은 선택된 기계 객체이며, `float`는 그 기계에서 작업을 완료하는 데 필요한 시간.
        """
        machine = operation.machine_available[0]
        process_time = operation.process_time[0]

        # record the dispatching result
        operation.machine_determined = machine
        operation.process_time_determined = process_time
        return machine, process_time

    def to_next_process(self):
        """
        부품이 공정을 완료한 후 다음 공정으로 이동시키는 프로세스.
        :: 부품이 공정을 완료하면 출력 버퍼에서 부품을 가져와 다음 공정으로 이동시킴. 
        만약 마지막 공정이 완료되었으면 부품을 최종 목적지로 이동시킴. 부품의 현재 위치를 업데이트하고, 다음 공정의 입력 버퍼로 부품을 전달.

        ### Yields:
            - `_type_`: `None`을 반환. 부품이 다음 공정으로 이동하는 과정에서 발생하는 이벤트를 생성.
        """
        while True:
            part = yield self.out_buffer.get()
            # 출력 버퍼에서 부품 가져옴.
            # print('Part Arrived:', part.name)
            if part.step[part.current_work] != len(part.op_list[part.current_work]) - 1:  # for operation 0,1,2,3 -> part.step = 1,2,3,4
                # 모든 공정 완료되지 않았다면 다음 공정으로 진행
                part.step[part.current_work] += 1
                part.op_list[part.current_work][part.step[part.current_work]].requirements.succeed()
                next_process = part.op_list[part.current_work][part.step[part.current_work]].process  # i.e. model['Process0']
                # 다음 공정의 요구사항을 충족시키면 다음 공정을 가져옴.

                # 다음 공정의 완료여부 확인 필요 -> 빈자리 날때까지 대기
                yield next_process.availability.put('using')
                # print('Next process', next_process.name , 'Opened!')
                # The machine is not assigned yet (to be determined further)
                yield next_process.in_buffer.put(part)

                # print(part.machine.name,'turned ready successfully')
                # 다음 프로세스의 입력 버퍼로 부품 이동.
                part.loc = next_process.name
                # 부품의 현재 위치 업데이트
            else:

                # 현재 work의 마지막 공정이 끝남

                # 1. 만약 마지막 Work 가 아니라면
                if part.current_work != len(part.step) - 1:
                    # part.step[part.current_work] = 0
                    buffer = part.model['Buffer']
                    # next_process = part.op_list[part.current_work][part.step[part.current_work]].process  # i.e. model['Process0']
                    yield buffer.buffer.put(part)
                    yield buffer.availability.put('using')
                    # 다음 프로세스의 입력 버퍼로 부품 이동.
                    part.loc = buffer.name

                # 2. 만약 마지막 work 였다면
                else:
                    self.model['Sink'].put(part)
                # 모든 공정 완료시 최종 목적지로 부품 이동.


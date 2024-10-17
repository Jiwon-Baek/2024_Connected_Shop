from environment.Monitor import *
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

# from config import CONSOLE_MODE
"""
콘솔 모드의 주된 목적은 개발자나 사용자가 프로그램의 내부 동작 상태를 보다 쉽게 모니터링하고, 
문제를 진단하며, 시스템의 성능을 평가할 수 있게 하는 것. 
"""



# region Sink
# 제조 시뮬레이션 종단 처리 역할을 하는 클래스. 최종 도착지. 완성된 부품들이 처리되고 기록되는 곳.
class Sink(object):
    """
        Sink 클래스를 정의. 
        이 클래스는 시뮬레이션의 마지막 단계에서 부품이 도착하고 완료된 부품의 수를 기록하는 역할을 함.

    ### Args:
        - `cfg (object)`: 설정 및 구성 정보를 담고 있는 객체. 주로 시뮬레이션의 설정을 포함함.
        - `env (simpy.Environment)`: 시뮬레이션 환경을 나타내는 SimPy 환경 객체.
        - `monitor (object)`: 시뮬레이션 이벤트를 기록하거나 모니터링하는 객체.
    """
    def __init__(self, cfg, env, monitor):
        self.env = env
        # 이는 로그 또는 모니터링 목적으로 사용 가능
        self.name = 'Sink'
        self.monitor = monitor
        self.cfg = cfg

        # Sink를 통해 끝마친 Part의 갯수 / SINK를 통과한 부품의 총 수를 기록.
        self.parts_rec = 0
        # 마지막 Part가 도착한 시간 / 마지막으로 Sink에 도착한 부품의 시간을 기록.
        self.last_arrival = 0.0
        # self.finished = simpy.Event()


    # put function
    def put(self, part):
        """
            부품이 Sink에 도착할 때 호출됨. 
            도착한 부품 수를 증가시키고, 관련 정보를 기록.

        ### Args:
            - `part (object)`: Sink로 전송된 부품 객체. 부품의 이름이나 기타 정보를 포함할 수 있음.

        ### Returns:
            - `None`
        """
        # 이 메서드는 부품이 sink로 전송될때 호출. 부품의 도착을 처리하고 관련 정보 기록.
        self.parts_rec += 1
        # 도착한 부품 수 증가
        self.last_arrival = self.env.now
        # 마지막 도착 시간 업데이트
        if self.cfg.CONSOLE_MODE :
            # 콘솔모드가 활성화되어 있을 경우, 콘솔에 부품 완성을 알리는 메시지를 출력
            monitor_console(self.env.now, part, command="Completed on")
        # 모니터 객체를 사용하여 부품 완성 이벤트를 기록.
        self.monitor.record(self.env.now, self.name, machine=None,
                            part_name=part.name, event="Completed")
        # 모든 예정된 작업이 완료되었는지 확인하고 모든 작업이 완료되면 최종 도착시간 업데이트
        if self.parts_rec == self.cfg.num_job:
            self.last_arrival = self.env.now
            # 마지막 작업이 완료된 시간을 업데이트

# endregion

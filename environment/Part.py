import numpy as np
from adodbapi.process_connect_string import process


# region JobType


class JobType:
    def __init__(self, idx=None, name=None):
        """
        Given to Source to create jobs

        :param jobtype:
        :param process_order:
        :param machine_order:
        :param setup_time:
        :param processing_time:
        """
        self.idx = idx  # int or any symbols
        self.name = name  # int or any symbols
        self.work_list = []  # contains WorkType object

    def add_work_type(self, _work_type):
        self.work_list.append(_work_type)


# endregion

# region WorkType
class WorkType(object):
    def __init__(self, idx=None, name=None):
        self.idx = idx
        self.name = name
        self.op_list = []

    def add_operation_type(self, _operation_type):
        self.op_list.append(_operation_type)


# endregion

# region OperationType
class OperationType(object):
    def __init__(self, idx=None, name=None, process=None, pt_list=None, m_list=None):
        self.idx = idx
        self.name = name
        self.process = process
        self.machine_list = m_list
        # self.pt_list = pt_list if pt_list is not None else self.randomize_pt_list()
        self.pt_list = pt_list

    # def randomize_pt_list(self):
    #     if self.process.name[0] is 'F':
    #         return np.random.randint(low=5, high=10, size=len(self.machine_list))
    #     elif self.process.name[0] is 'P':
    #         return np.random.randint(low=20, high=30, size=len(self.machine_list))




# endregion


# region Operation
class Operation(object):
    # 실제 공정이 아니라 작업(job)내의 각 공정 단계를 나타내는 멤버 변수를 포함하는 클래스
    """
    This class does not act as a process.
    Instead, this is just a member variable of a job that contains process info.
    This class is only used when generating a job sequence.
    The Process class is to be generated further.
    """

    def __init__(self, model, env, idx, part_name, op_type, process_type=None, machine_list=None,
                 process_time=None, requirements=None, preemption=None):
        """
        공정 객체 초기화 메서드
        :: 주어진 매개변수들을 사용하여 공정 객체 초기화

        ### Args:
            - `model (object)`: 공정 및 기계 객체들을 포함하는 모델
            - `env (object)`: 현재 환경 또는 시뮬레이션 객체
            - `id (int)`: 공정의 고유 ID
            - `part_name (str)`: 공정이 수행되는 부품의 이름
            - `process_type (str)`: 공정의 타입을 나타내는 문자열
            - `machine_list (list)`: 공정을 수행할 수 있는 기계들의 목록
            - `process_time (int or list)`: 공정에 필요한 시간, 정수 또는 리스트로 제공될 수 있음
            - `requirements (list, optional)`: 공정 시작 전에 충족되어야 하는 조건들을 나타냄. 기본값은 None
            - `operation_type (str, optional)`: 공정의 타입을 나타내는 문자열. 기본값은 None
            - `preemption (bool, optional)`: 선점 여부를 나타내는 불리언 값. 기본값은 None
        """
        self.model = model
        self.idx = idx  # Integer
        self.op_type = op_type
        self.process = op_type.process
        # self.process = self.model[process_type] if  isinstance(process_type, str) else process_type  # Convert String to Process Object
        # 문자열로 주어진 공정 타입을 해당 모델의 공정 객체로 변환
        self.process_time = op_type.pt_list  # Integer, or given as List
        # 공정에 필요한 시간, 정수 또는 리스트로 제공될 수 있음.
        self.part_name = part_name  # Inherited
        # 상속된 부품 이름
        self.name = part_name + '_Op' + str(idx)
        # 공정의 고유 이름, 부품 이름과 id를 결합하여 생성.

        self.machine_available = op_type.machine_list  # list of strings
        # 공정을 수행할 수 있는 기계 목록
        self.preemption = preemption

        # # In the simplest Job Shop problem, process type is often coincide with the machine type itself.
        # # 가장 간단한 Job Shop 문제에서는 프로세스 유형이 기계 유형 자체와 일치하는 경우가 많습니다.
        # if isinstance(machine_list, list):  # 인스턴스와 타입이 같은지 확인하는 코드
        #     self.machine_available = [self.model[machine_list[i]] for i in range(len(machine_list))]
        #     # machine list가 여러 기계를 포함하는 리스트일 경우, 각 기계에 대해 모델에서 해당 기계 객체를 찾아 머신어베일러블 리브스테 추가.
        # else:
        #     self.machine_available = [self.model[machine_list]]
        #     # 단일 기계만을 나타내는 경우, 이 기계를 모델에서 찾아 list에 단일 원소로 추가. 이 경우, process를 수행할 수 있는 기계는 오직 하나.

        # 결정된 특정 기계, 결정된 기계의 인덱스, 결정된 공정 시간
        self.machine_determined = None
        self.machine_determined_idx = None
        self.process_time_determined = None

        # 공정의 선행 조건이나 요구사항 처리.
        if requirements is None:
            # 공정 시작 전에 충족되어야 하는 조건들을 나타냄.
            self.requirements = env.event()  # preceding event
            # requirements가 논인 경우, 기본적으로 새로운 이벤트 생성하고 저장.
            if idx == 0:
                self.requirements.succeed()
                # 만약 첫번째 공정(id==0)이면 이 이벤트는 자동으로 성공 상태가 되어 즉시 시작가능.
        else:
            # if there are more requirements, more env.event() can be added up
            # you can handle events using Simpy.AllOf() or Simpy.AnyOf()
            self.requirements = [env.event() for i in range(5)]  # This is an arbitrary value
            # 여러 개의 이벤트를 생성하여 저장.

        self.queue_entry_time = None  # queue_entry_time 변수를 추가하여 초기화

    def randomize_pt_list(self):
        if self.process.name[0] == 'F':
            self.process_time = np.random.randint(low=5, high=10, size=len(self.machine_available))
        elif self.process.name[0] == 'P':
            self.process_time =  np.random.randint(low=20, high=30, size=len(self.machine_available))



# endregion
class Work(object):
    def __init__(self, _model, _env, _id, _part_name, _work_type):
        self.model = _model
        self.env = _env
        self.id = _id
        self.part_name = _part_name
        self.work_type = _work_type


# region Job
class Job(object):
    # 반복적으로 생성되는 작업을 의미, 각 작업은 여러 공정(operation)을 포함할 수 있음.
    """
    A job is to be repeatedly generated in a source.
    (Job1_1, Job1_2, Job1_3, ..., Job1_100,
    Job2_1, Job2_2, Job2_3, ..., Job2_100,
    ...                         Job10_100)

    Member Variable : job_type, id
    """

    def __init__(self, model, env, job_type, idx, release_date=None, due_date=None):
        self.model = model
        self.env = env
        self.job_type = job_type
        self.idx = idx
        self.due_date = due_date
        self.release_date = release_date

        # 작업 유형 객체, 각 작업 유형은 고유한 공정 수, 공정 순서, 기계 순서 등을 정의할 수 있음.
        if len(self.job_type.work_list) > 1:
            self.num_process = [len(w.op_list) for w in self.job_type.work_list]  # given as tuple
        else:
            self.num_process = job_type.num_process

        self.name = job_type.name + '_' + str(idx)
        self.current_work = -1
        if len(self.job_type.work_list) > 1:
            self.step = [-1 for w in range(len(self.job_type.work_list))]
        else:
            self.step = -1
        # 현재 공정 단계를 추적하는 인덱스

        self.loc = None  # current location
        # 작업의 현재 위치

        self.work_list = []
        self.op_list = [list() for i in range(len(self.job_type.work_list))]
        # 작업에 포함된 공정(operation)객체의 리스트.

        self.generate_work()
        self.generate_operation()
        # 작업 유형에 정의된 공정 수와 정보를 바탕으로 작업에 필요한 'operation' 객체를 생성하고 초기화.

    def generate_work(self):
        for j, _work_type in enumerate(self.job_type.work_list):
            w = Work(self.model, self.env, _id=j, _part_name=self.name, _work_type=_work_type)
            self.work_list.append(w)

    def generate_operation(self):
        """
        작업에 필요한 'operation' 객체를 생성하고 초기화하는 메서드

        작업 유형에 정의된 공정 수와 정보를 바탕으로 'operation' 객체를 생성하여 초기화하고, 이를 `op` 리스트에 추가합니다.
        """
        # 작업 유형에 정의된 공정 수와 정보를 바탕으로 작업에 필요한 'operation' 객체를 생성하고 초기화.
        for i, w_type in enumerate(self.job_type.work_list):
            for j, op_type in enumerate(w_type.op_list):
                o = Operation(self.model, self.env, idx=j, part_name=self.name, op_type=op_type)
                o.randomize_pt_list()
                self.op_list[i].append(o)
            # op라는 리스트에 model, id, part_name, machine_list 등을 저장한다.

# endregion Job

"""
Argumentparser를 쓰지 않고 실험 변수를 전달하기 위해 만든 클래스
"""
import os
from datetime import datetime

class Configure:
    def __init__(self, num_job=100, num_machine=5):
        self.env = 'OE'
        self.use_vessl = 0
        self.load_model = False
        self.model_path = None
        self.num_job = num_job
        self.num_machine = num_machine
        # self.IAT = float("inf")
        self.SIMUL_TIME = 1000

        # Process Variables
        self.DISPATCH_MODE = 'FIFO'  # FIFO, Manual

        # Monitor Variables
        self.OBJECT = 'Machine'
        self.CONSOLE_MODE = True

        # Visualization Variables
        self.TITLE = "FJSP data structure"
        self.ylabel = 'Job'
        self.xlabel = 'Time'

        current_working_directory = os.getcwd()

        # Directory Configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
        folder_name = 'result'  # Define the folder name
        self.save_path = os.path.join(script_dir, folder_name)  # Construct the full path to the folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        now = datetime.now()
        self.filename = now.strftime('%Y-%m-%d-%H-%M-%S')   # strftime: datetime 객체의 메서드
        self.filepath = os.path.join(self.save_path, self.filename + '.csv')

        self.num_operation = self.num_job * self.num_machine
        self.num_show = self.num_operation + 1
        self.show_interval_time = 100
        self.finished_pause_time = 1000


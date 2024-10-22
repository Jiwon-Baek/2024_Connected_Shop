import simpy

class Shop:
    def __init__(self, env, cfg, monitor, name, data):
        self.env = env
        self.cfg = cfg
        self.name = name
        self.data = data
        self.monitor = monitor
        self.model = dict()

        self.model['Source'] = Source(cfg, env, 'Source', self.model, monitor, job_type=data.jobtype, IAT=0, num_parts=10)
        self.model['Sink'] = Sink(cfg, env, monitor)

    def set_layout(self, num_process):
        for i in range(num_process):
            self.model['M' + str(i)] = Machine(self.env, i, 'M' + str(i))
            self.model['P' + str(i)] = Process(self.cfg, self.env, 'P' + str(i), self.model, self.monitor, None)



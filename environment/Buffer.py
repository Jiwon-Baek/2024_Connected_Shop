import simpy

class Buffer(object):
    def __init__(self, _cfg, _env, _name, _model, _monitor):
        self.env = _env
        # _ 언더바는 임시 또는 지역 변수로 사용하거나 접근제한을 나타냄(비공개, 내부용)
        self.cfg = _cfg
        self.name = _name  # 해당 Buffer의 이름
        self.model = _model
        self.monitor = _monitor

        self.buffer = simpy.Store(_env, capacity=100)  # 10 is an arbitrary number

        # _env.process(self.to_next_process())
        _env.process(self.routing())
    def routing(self):
        while True:
            # 1. Get a part from the list of generated parts
            part = yield self.buffer.get()

            print(part.name, "entered Buffer at ", self.env.now)
            self.env.process(self.to_next_process(part))

    def to_next_process(self, part):

        print(part.name, "started its routing at ", self.env.now)
        part.current_work += 1
        part.step[part.current_work] += 1
        self.monitor.record(self.env.now, self.name, machine=None,
                            part_name=part.name,
                            event="Routing Start")
        next_op = part.op_list[part.current_work][part.step[part.current_work]]
        next_process = next_op.process  # i.e. model['시운전']
        # print('First Process of ', part.name, ' is:', next_process.name)
        if self.cfg.CONSOLE_MODE:
            print(part.name, "is going to be put in ", next_process.name)
        yield
        yield next_process.in_buffer.put(part)
        print(part.name, "left Buffer at ", self.env.now)
        part.loc = next_process.name

        # 4. Record
        self.monitor.record(self.env.now, self.name, machine=None,
                            part_name=part.name,
                            event="Routing Finished")
        self.monitor.record(self.env.now, next_process.name, machine=None,
                            part_name=part.name, event="Part transferred from Source")
    pass
import simpy

class Buffer(object):
    def __init__(self, _cfg, _env, _name, _model, _monitor):
        self.env = _env
        # _ 언더바는 임시 또는 지역 변수로 사용하거나 접근제한을 나타냄(비공개, 내부용)
        self.cfg = _cfg
        self.name = _name  # 해당 Buffer의 이름
        self.model = _model
        self.monitor = _monitor
        self.availability = simpy.FilterStore(_env, capacity=float('inf'))
        self.buffer = simpy.Store(_env, capacity=float('inf'))  # 10 is an arbitrary number

        # _env.process(self.to_next_process())
        _env.process(self.routing())
    def routing(self):
        while True:
            # 1. Get a part from the list of generated parts
            part = yield self.buffer.get()

            # part.set_process(self)

            print(part.name, "entered Buffer at ", self.env.now)
            self.env.process(self.to_next_process(part))

    def to_next_process(self, part):


        print("%d\t%s가 Buffer에 도착했습니다." % (self.env.now,part.name))
        print("%d\t현재 표시되어 있는 %s의 process 는 %s 입니다." % (self.env.now,part.name, part.process.name))
        yield part.process.availability.get()
        print("%d\t이전 process인 %s의 사용권 반환" % (self.env.now,part.process.name))
        part.set_process(self)
        print("%d\t%s 의 process를 %s 로 바꿨습니다" % (self.env.now,part.name, part.process.name))
        # print(part.name, "started its routing at ", self.env.now)
        part.current_work += 1
        part.step[part.current_work] += 1
        self.monitor.record(self.env.now, self.name, machine=self.name,
                            part_name=part.name,
                            event="Started")
        next_op = part.op_list[part.current_work][part.step[part.current_work]]
        next_process = next_op.process  # i.e. model['시운전']
        if self.cfg.CONSOLE_MODE:
            print(part.name, "is going to be put in ", next_process.name)

        # 다음 공정의 완료여부 확인 필요 -> 빈자리 날때까지 대기
        yield next_process.availability.put('using')
        yield next_process.in_buffer.put(part)
        # self.availability.get()
        print(part.name, "left Buffer at ", self.env.now)
        part.loc = next_process.name

        # 4. Record
        self.monitor.record(self.env.now, self.name, machine=self.name,
                            part_name=part.name,
                            event="Finished")
        # self.monitor.record(self.env.now, next_process.name, machine=self.name,
        #                     part_name=part.name, event="Part transferred from Buffer")
    pass
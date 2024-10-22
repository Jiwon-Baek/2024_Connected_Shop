import json
import numpy as np

if __name__ == "__main__":
    info = dict()
    for i in range(10):
        info[str(i)] = dict()
        time = 0.0
        iat = 0.0
        for j in range(100):
            if j != 0:
                iat = np.random.exponential(15)
                time += iat

            pt_1 = [np.random.randint(low=10, high=30, size=1).tolist() for _ in range(5)]
            pt_2 = [np.random.randint(low=70, high=90, size=1).repeat(3).tolist()]
            d = {
                'Name':'Job_'+str(j),
                'Created':time,
                'IAT':iat,

                'Work':['FS','PMS'],
                'num_process':[5,1],
                'type_machine':['Unrelated', 'Identical'],

                'num_machine': [[1, 1, 1, 1, 1], [3]],
                'processing_time':[pt_1,pt_2],
            }


            info[str(i)]['Job_'+str(j)]=d

    # _data.json 파일에 딕셔너리 형태로 info 저장
    with open("data.json", 'w') as f:
        json.dump(info, f, indent='\t')
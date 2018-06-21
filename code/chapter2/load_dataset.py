
import sys
from sklearn import datasets
import numpy as np
import pandas as pd

def combineDataTarget(data,target,target_explain):
    targe_new = np.expand_dims(target,1)

    target_names = np.array([target_explain[i] for i in target])
    target_names_new = np.expand_dims(target_names,1)
    ret = np.concatenate((data,targe_new,target_names_new),1)
    return ret


class DataLoader(object):

    def load(self, name):
        if name == "iris":
            raw =  datasets.load_iris()
            print raw['target_names']
            c_data = combineDataTarget(raw['data']
                                       ,raw['target']
                                       ,raw['target_names'])
            columns = raw['feature_names']
            columns += ['flow_type_id']
            columns += ['flower_type']
            df = pd.DataFrame(c_data,columns=columns)
            return df
        else:
            raise Exception("Unsupport dataset name: "+name)

class DataCSVSaver(object):
    def save(self, name, df):
        df.to_csv(name,index=False)

def main(args):
    name = args
    dl = DataLoader()
    ds = DataCSVSaver()
    df = dl.load("iris")
    ds.save("./data/iris.csv",df)


if __name__ == "__main__":
    main(sys.argv)
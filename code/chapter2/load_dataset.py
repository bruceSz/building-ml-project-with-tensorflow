
import sys
from sklearn import datasets


class DataLoader(object):

    def load(self, name):
        if name == "iris":
            return datasets.load_iris()
        else:
            raise Exception("Unsupport dataset name: "+name)

class DataCSVSaver(object):
    def save(self, name, df):
        df.to_csv(name,index=False)

def main(args):
    name = args
    dl = DataLoader()
    da = dl.load("iris")
    print da['target_names']
    print da['feature_names']
    print 


if __name__ == "__main__":
    main(sys.argv)
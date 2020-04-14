from random import Random
import string
import unidecode

MAX_LEN = 24
class NameDataset:
    def __init__(self, train_file, test_file, ratio):
        self.datas = []
        self.train_file = train_file
        self.test_file = test_file
        self.ratio = ratio

    def add_data_country(self, loc_file, id, max_row=-1):
        with open(loc_file) as f:
            fdatas = f.read().strip().split("\n")
            print(len(fdatas))
            Random(1337).shuffle(fdatas)
            datas = []
            for fdata in fdatas[:max_row]:
                data = unidecode.unidecode(fdata) # clear unicode char to ascii
                data = [data]
                data.append(id)
                datas.append(data)
            self.datas.extend(datas)

    def write(self):
        Random(1337).shuffle(self.datas)
        n = int(round(len(self.datas)*self.ratio))
        print(n)
        test_data = self.datas[:n]
        train_data = self.datas[n:]
        with open(self.train_file, "w+") as f:
            f.write("nama,country\n")
            for data in train_data:
                f.write(",".join(map(str, data)) + "\n")

        with open(self.test_file, "w+") as f:
            f.write("nama,country\n")
            for data in test_data:
                f.write(",".join(map(str, data)) + "\n")

dataset = NameDataset("train.csv", "evaluation.csv", 0.2)
dataset.add_data_country("russian_name_dataset.txt", 0, 9800)
dataset.add_data_country("chinese_name_dataset.txt", 1, 9800)
dataset.add_data_country("arabic_name_dataset.txt", 2, 9800)
dataset.add_data_country("german_name_dataset.txt", 3, 9800)
dataset.add_data_country("korean_name_dataset.txt", 4, 9800)
dataset.write()

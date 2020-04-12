from random import Random

MAX_LEN = 24
class NameDataset:
    def __init__(self, train_file, test_file, ratio):
        self.datas = []
        self.train_file = train_file
        self.test_file = test_file
        self.ratio = ratio

    def padzero(self, s, n):
        return s.ljust(n, "\x00")

    def str2int(self, s):
        return list(map(ord, s))

    def make_name(self, s, pad):
        return self.str2int(self.padzero(s, pad))

    def add_data_country(self, loc_file, id, max_row=-1):
        with open(loc_file) as f:
            fdatas = f.read().strip().split("\n")
            print(len(fdatas))
            Random(1337).shuffle(fdatas)
            datas = []
            for fdata in fdatas[:max_row]:
                data = self.make_name(fdata[:MAX_LEN], MAX_LEN)
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
            for data in train_data:
                f.write(",".join(map(str, data)) + "\n")

        with open(self.test_file, "w+") as f:
            for data in test_data:
                f.write(",".join(map(str, data)) + "\n")

dataset = NameDataset("train.csv", "evaluation.csv", 0.2)
dataset.add_data_country("russian_name_dataset.txt", 0, 9800)
dataset.add_data_country("chinese_name_dataset.txt", 1, 9800)
dataset.add_data_country("arabic_name_dataset.txt", 2, 9800)
dataset.write()
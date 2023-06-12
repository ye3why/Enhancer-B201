def flatten_list(li):
    return sum(([x] if not isinstance(x, list) and not isinstance(x, tuple) else flatten_list(x) for x in li), [])


class Writer():
    def __init__(self, filename=None, isprint=False):
        self.filename = filename
        self.isprint = isprint
        if self.filename:
            self.f = open(filename, 'w')

    def __del__(self):
        self.close()

    def close(self):
        if self.filename:
            self.f.close()

    def print_and_write(self, content):
        if self.isprint:
            print(content)
        if self.filename:
            print(content, file=self.f)
            # self.f.write(content)

    def print_blank_line(self, number=1):
        self.print_and_write('\n' * (number - 1)) # print(end='\n')


def dict_debatch(datainfo):
    res = []
    for idx in range(datainfo[list(datainfo.keys())[0]].shape[0]):
        tmp = {}
        for k, v in datainfo.items():
            tmp[k] = v[idx]
        res.append(tmp)
    return res

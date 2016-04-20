class Reverse:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.index == 0:
            raise StopIteration
        else:
            self.index = self.index - 1
            return self.data[self.index]

def test(s):
    rev = Reverse(s)
    for char in rev:
        print char
        
test('abcde')
if __name__ == '__main__':
    s = "fspam"
    test(s)
    
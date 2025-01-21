import dis

def example():
    x = 1
    y = 2
    return x + y
def example2():
    class MyExamle:
        def __init__(self, name):
            self.name = name
        def sum(self, a, b):
            return a + b
    me = MyExamle("Jaina")
    me.sum(1, 2)
    an = MyExamle("car")
    an.sum(2, 2)
    # return me.name

# View the bytecode operations
dis.dis(example)
print("="*30)
dis.dis(example2)

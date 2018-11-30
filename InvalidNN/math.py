from InvalidNN import core


class Add(core.CompressNode):
    def __init__(self, name=None, scope=None):
        super().__init__(name, scope)

    def func(self, *args):
        return sum(*args)

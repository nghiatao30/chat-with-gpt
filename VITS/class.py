class test:

    def __init__(self, test2):
        self.test_2 = test2

    def call(self,a,b):

        c = self.test_2(a,b)
        return c

class test2:

    def forward(self,a,b):
        return a+b
        
test2 = test2()
testing = test(test2)
c = testing.call(1,2)
print(c)
# https://gist.github.com/ulope/1935894
'''
super(B, self).x() is more safe,
super(self.__class__, self).x() can not be used iteratively
'''

class A(object):
    def x(self):
        print ("A.x")

class B(A):
    def x(self):
        print ("B.x")
        super(B, self).x()

class C(B):
    def x(self):
        print ("C.x")
        super(C, self).x()
        
class D(C):
    def x(self):
        print ("D.x")
        super(self.__class__, self).x()
        
class E(D):
    def x(self):
        print ("E.x")
        super(self.__class__, self).x()


class X(object):
    def x(self):
        print ("X.x")

class Y(X):
    def x(self):
        print ("Y.x")
        super(self.__class__, self).x()  # <- this is WRONG don't do it!

class Z(Y):
    def x(self):
        print ("Z.x")
        super(self.__class__, self).x()  # <- this is WRONG don't do it!

if __name__ == '__main__':
    C().x()
    D().x()
#    E().x()
#    Z().x() # will cause 'RuntimeError: maximum recursion depth exceeded'
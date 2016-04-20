class Base:
    def __init__(self):
        print 'Base contructor'
        
class A(Base):
    def __init__(self):
        #print "A constructor"
        #super(A, self).__init__()
        Base.__init__(self)

if __name__ == '__main__':
    a = A()
    
# the weird thing is that, if Base is not Base(object), super() will not work
# Fuck!
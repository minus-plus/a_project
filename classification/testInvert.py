

def arrayInvert(array):
  """
  Inverts a matrix stored as a list of lists.
  """
  result = [[] for i in array]
  for outer in array:
    for inner in range(len(outer)):
      result[inner].append(outer[inner])
  return result

def testInvert():
    count = 1
    l = []
    for i in range(0,3):
        l2 = []
        for j in range(0, 3):
            l2.append(count)
            count = count + 1
        l.append(l2)
    print l
    l3 = arrayInvert(l)
    print l3

if __name__ == "__main__":
    testInvert()
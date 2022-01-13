class Person:
  def __init__(self, name = 'lihua'):
    self.name = name

  def __str__(self):
    return '{}'.format(self.name)

p1 = Person()

p2 = Person('zhangsan')

s = str(p1)


print(s)
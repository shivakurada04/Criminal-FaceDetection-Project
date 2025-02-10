class Person:
    def __init__(self):
        pass
    def talk(self,Name):
        self.Name=Name
        print(f'{Name} is talking ')
obj=Person()
obj.talk('Sai')
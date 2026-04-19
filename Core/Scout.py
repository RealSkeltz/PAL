from Shared.Classes.Pal import Pal
from Shared.Prompts import scout_identity

class Scout(Pal):
    def __init__(self):
        super().__init__(scout_identity)

    def run(self):
        return super().run()

    def see(self):
        return super().see()
    
    def listen(self):
        return super().listen()
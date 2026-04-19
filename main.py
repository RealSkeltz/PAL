import threading
from Core.Scout import Scout
from Shared.Classes.Pal import Pal

scout = Scout()
scout.run()

threading.Event().wait()  # keep main thread alive
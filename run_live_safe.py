import sys
import os

# aggiunge la cartella src al percorso
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# esegue il tuo script originale
from app.run_live import *

print("\n✅ Run completato")
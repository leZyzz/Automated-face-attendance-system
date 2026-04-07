import json
import os 
import numpy as np
def load_data(db_file):
        if not os.path.exists(db_file): return {}
        try:
            with open(db_file, 'r') as f:
                raw = json.load(f)
                return {k: {"id": v["id"], "embeddings": [np.array(e) for e in v["embeddings"]]} for k, v in raw.items()}
        except: return {}
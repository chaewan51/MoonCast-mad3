import json, os
import torchaudio
from pathlib import Path

m = json.loads(Path("voices/manifest.json").read_text())
# pick first 5 american
amer = [v for v in m if v.get("accent","").lower().strip()=="american"][:5]
print("american voices:", len([v for v in m if v.get("accent","").lower().strip()=="american"]))
for v in amer:
    p = Path(v["path"])
    print("\nPATH:", p, "exists:", p.exists())
    if p.exists():
        try:
            info = torchaudio.info(str(p))
            print("INFO:", info)
        except Exception as e:
            print("torchaudio.info FAILED:", repr(e))
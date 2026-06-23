"""
Ouvre toutes les figures generees (08_results/figures/*.png) avec la visionneuse
par defaut du systeme. Lancer apres run_all.py.

Usage : python 07_code/view_figures.py
"""
import os
import sys
import glob
import subprocess

import config as C

figs = sorted(glob.glob(str(C.FIGURES / "*.png")))
if not figs:
    print("Aucune figure trouvee. Lance d'abord : python 07_code/run_all.py")
    sys.exit(1)

print(f"{len(figs)} figures dans {C.FIGURES} :")
for f in figs:
    print("   ", os.path.basename(f))

if sys.platform == "darwin":            # macOS
    subprocess.run(["open", *figs])
elif sys.platform.startswith("linux"):
    for f in figs:
        subprocess.run(["xdg-open", f])
elif sys.platform.startswith("win"):
    for f in figs:
        os.startfile(f)                 # type: ignore[attr-defined]
else:
    print("Ouvre-les manuellement depuis", C.FIGURES)

print("Ouverture demandee.")

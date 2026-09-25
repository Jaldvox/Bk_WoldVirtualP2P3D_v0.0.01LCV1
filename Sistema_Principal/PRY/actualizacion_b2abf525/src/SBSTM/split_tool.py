import pathlib
SRC = pathlib.Path(r"C:\Users\User\Desktop\WoldVirtualP2P3D_v0.0.01LCV1\VersionDe_sesion\CONTRRF\TLLDCD\src")

# 1. SBSTM/__init__.py -> accesores.py (ultimas 83 lineas)
p = SRC / "SBSTM" / "__init__.py"
L = p.read_text(encoding="utf-8").splitlines(keepends=True)
tail = L[420:]
acc = SRC / "SBSTM" / "accesores.py"
header = '"""Accesores SBSTM extraidos de init: RPLC, voz, IAFREE, STYLOS."""\nfrom __future__ import annotations\n'
acc.write_text(header + "".join(tail), encoding="utf-8")
p.write_text("".join(L[:420]), encoding="utf-8")
print("sbstm_init", len(L[:420]), "accesores", len(tail) + 2)

# 2. BKSVCB.py -> bksvcb_net.py (desde linea 345, idx 344)
b = SRC / "STM_BKCH" / "BKSVCB.py"
BL = b.read_text(encoding="utf-8").splitlines(keepends=True)
net = SRC / "STM_BKCH" / "bksvcb_net.py"
net_header = '"""Servidor HTTP y daemon BKSVCB extraidos para limite 450."""\nfrom __future__ import annotations\n'
net.write_text(net_header + "".join(BL[344:]), encoding="utf-8")
b.write_text("".join(BL[:344]), encoding="utf-8")
print("bksvcb", len(BL[:344]), "net", len(BL[344:]) + 2)

# 3. RFEN1_RN_4.py recorte a <=450 (quita 2 lineas en blanco dobles)
r = SRC / "LC_STM" / "red_neuronal" / "RF_EN" / "RFEN1_RN_4.py"
RL = r.read_text(encoding="utf-8").splitlines(keepends=True)
print("rfen_antes", len(RL))
out = []
blanks = 0
for i, line in enumerate(RL):
    if line.strip() == "":
        blanks += 1
        if blanks > 1 and len(out) + (len(RL) - i) > 450:
            continue
    else:
        blanks = 0
    out.append(line)
while len(out) > 450:
    for i in range(len(out) - 1, -1, -1):
        if out[i].strip() == "":
            del out[i]
            break
    else:
        break
r.write_text("".join(out), encoding="utf-8")
print("rfen_despues", len(out))

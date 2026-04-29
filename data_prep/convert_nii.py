import os, subprocess, glob
from concurrent.futures import ProcessPoolExecutor

SRC_PAIR = "/workspace/DaTSCAN-pair"
SRC_ORIG = "/workspace/DaTSCAN/PPMI"
DST = "/workspace/DaTSCAN-nii"

def find_dcm_dir(subj, iid):
    # prefer folder containing .dcm; fall back to any match
    candidates = glob.glob(os.path.join(SRC_ORIG, subj, "Reconstructed_DaTSCAN", "*", iid))
    for c in candidates:
        if glob.glob(os.path.join(c, "*.dcm")):
            return c
    return None

def convert(args):
    visit, name = args
    out_dir = os.path.join(DST, visit)
    out_nii = os.path.join(out_dir, name + ".nii.gz")
    if os.path.exists(out_nii):
        return (visit, name, "skip")

    src = os.path.join(SRC_PAIR, visit, name)
    if not glob.glob(os.path.join(src, "*.dcm")):
        subj, iid = name.split("_", 1)
        fixed = find_dcm_dir(subj, iid)
        if fixed is None:
            return (visit, name, "NO_DCM")
        src = fixed

    r = subprocess.run(
        ["dcm2niix", "-z", "y", "-o", out_dir, "-f", name, src],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not os.path.exists(out_nii):
        return (visit, name, f"FAIL rc={r.returncode}: {r.stderr[:200]}")
    return (visit, name, "ok")

jobs = []
for v in ("SC", "V04"):
    os.makedirs(os.path.join(DST, v), exist_ok=True)
    for n in sorted(os.listdir(os.path.join(SRC_PAIR, v))):
        jobs.append((v, n))

print(f"Total jobs: {len(jobs)}")

ok = skip = fail = 0
fails = []
with ProcessPoolExecutor(max_workers=8) as ex:
    for i, res in enumerate(ex.map(convert, jobs), 1):
        v, n, status = res
        if status == "ok": ok += 1
        elif status == "skip": skip += 1
        else:
            fail += 1
            fails.append(res)
        if i % 200 == 0:
            print(f"  {i}/{len(jobs)} ok={ok} skip={skip} fail={fail}", flush=True)

print(f"Done. ok={ok} skip={skip} fail={fail}")
for f in fails[:10]:
    print(" FAIL:", f)

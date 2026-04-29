import csv, os, glob
from collections import defaultdict

CSV = "/workspace/PPMI/ppmi-data_3_26_2026.csv"
SRC = "/workspace/DaTSCAN/PPMI"
DST = "/workspace/DaTSCAN-pair3"

with open(CSV) as f:
    rows = list(csv.DictReader(f))

dat = [r for r in rows
       if "DaTSCAN" in r["Description"] and r["Group"] != "Control"]

subj_visit = defaultdict(set)
for r in dat:
    subj_visit[r["Subject"]].add(r["Visit"])

both = {s for s, v in subj_visit.items() if "SC" in v and "V04" in v}
print(f"PD subjects with both SC and V04: {len(both)}")

counts = {"SC": 0, "V04": 0}
missing = []

for visit in ("SC", "V04"):
    out = os.path.join(DST, visit)
    os.makedirs(out, exist_ok=True)
    recs = [r for r in dat if r["Visit"] == visit and r["Subject"] in both]
    for r in recs:
        subj = r["Subject"]
        iid = r["Image Data ID"]
        matches = glob.glob(os.path.join(SRC, subj, "Reconstructed_DaTSCAN", "*", iid))
        if not matches:
            missing.append((subj, visit, iid))
            continue
        link = os.path.join(out, f"{subj}_{iid}")
        if not os.path.exists(link) and not os.path.islink(link):
            os.symlink(matches[0], link)
        counts[visit] += 1

print("Linked:", counts)
print("Missing:", len(missing))

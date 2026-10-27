import os, glob, copy, gc, random, pickle, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
import torchvision, torchvision.transforms as T, timm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

for _a in sys.argv[1:]:
    _k, _, _v = _a.partition("=")
    if _k in ("SEED", "PRESET", "SAMPLE", "TOPN", "MODELS") and _v: os.environ[_k] = _v
ROOT = "/kaggle/input/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset"
MAG, BATCH = "400", 16
SEED = int(os.environ.get("SEED", 42))
MODELS = os.environ.get("MODELS", "xception,googlenet,densenet201,resnet18,inception_v3").split(",")
TOPN = int(os.environ.get("TOPN", "0"))
EP_HEAD, EP_FT, PATIENCE = 6, 25, 8
TOPK = 1500
POP, GEN = 100, 30
T0 = time.time()
PRESET = os.environ.get("PRESET", "full")
if PRESET == "fast": EP_HEAD, EP_FT, PATIENCE, TOPK, POP, GEN = 4, 12, 4, 1000, 50, 15
SAMPLE = os.environ.get("SAMPLE", "0") == "1"
TAG = "sample_" if SAMPLE else ""
if SAMPLE: EP_HEAD, EP_FT, TOPK, POP, GEN = 1, 1, 100, 4, 2
CACHE = f"{TAG}bb_cache_{SEED}.pkl"
RESULTS = f"{TAG}results_{len(MODELS)}models{'_top' + str(TOPN) if TOPN else ''}.csv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BN = nn.modules.batchnorm._BatchNorm

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
print(f"CONFIG | SEED={SEED} PRESET={PRESET} SAMPLE={SAMPLE} TOPN={TOPN} MODELS={MODELS} -> {RESULTS}")

paths, y, pid, seen = [], [], [], set()
for p in sorted(glob.glob(os.path.join(ROOT, "**", "*.png"), recursive=True)):
    n = os.path.basename(p); parts = n.split("-")
    if len(parts) < 4 or parts[-2] != MAG or n in seen: continue
    seen.add(n); paths.append(p); y.append(1 if n.split("_")[1] == "M" else 0); pid.append("-".join(parts[1:-2]))
paths, y, pid = np.array(paths), np.array(y), np.array(pid)
print(f"{MAG}X images: {len(paths)} | malignant {y.sum()} | benign {(y==0).sum()} | patients {len(set(pid))}")
print("Official 400X: 1820 total | 588 benign | 1232 malignant  <- must match, else stop")

i_tr, i_tmp = train_test_split(np.arange(len(y)), test_size=0.40, stratify=y, random_state=SEED)
i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
print(f"train {len(i_tr)} | val {len(i_va)} | test {len(i_te)}")
print("patients shared train/test:", len(set(pid[i_tr]) & set(pid[i_te])), "(image-level split -> optimistic)")
ytr, yva, yte = y[i_tr], y[i_va], y[i_te]

class DS(torch.utils.data.Dataset):
    def __init__(s, idx, tf): s.idx, s.tf = idx, tf
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]
        return s.tf(Image.open(paths[j]).convert("RGB")), torch.tensor(float(y[j]))

def loader(idx, size, ms, train):
    aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
           T.ColorJitter(.15, .15, .1, .02)] if train else []
    tf = T.Compose([T.Resize((size, size))] + aug + [T.ToTensor(), T.Normalize(*ms)])
    return torch.utils.data.DataLoader(DS(idx, tf), batch_size=BATCH, shuffle=train, num_workers=2, pin_memory=True)

IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
def build(name):
    tv = torchvision.models
    if name == "resnet18":
        m = tv.resnet18(weights="DEFAULT"); d = m.fc.in_features; m.fc = nn.Identity(); return m, d, 224, IMN
    if name == "googlenet":
        m = tv.googlenet(weights="DEFAULT"); m.aux_logits = False; m.aux1 = m.aux2 = None
        d = m.fc.in_features; m.fc = nn.Identity(); return m, d, 224, IMN
    if name == "inception_v3":
        m = tv.inception_v3(weights="DEFAULT"); m.aux_logits = False; m.AuxLogits = None
        d = m.fc.in_features; m.fc = nn.Identity(); return m, d, 299, IMN
    if name == "densenet201":
        m = tv.densenet201(weights="DEFAULT"); d = m.classifier.in_features; m.classifier = nn.Identity(); return m, d, 224, IMN
    if name == "xception":
        try: m = timm.create_model("legacy_xception", pretrained=True, num_classes=0)
        except Exception: m = timm.create_model("xception", pretrained=True, num_classes=0)
        return m, m.num_features, 299, ((.5, .5, .5), (.5, .5, .5))
    raise ValueError(name)

def run_epoch(net, ld, opt, scaler):
    for x, t in ld:
        x, t = x.to(DEV), t.to(DEV)
        with torch.autocast("cuda", enabled=DEV == "cuda"): out = net(x).squeeze(1)
        loss = Fn.binary_cross_entropy_with_logits(out.float(), t * 0.95 + 0.025)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

@torch.no_grad()
def predict(net, ld, feats=False):
    net.eval(); P, F, L = [], [], []
    for x, t in ld:
        x = x.to(DEV)
        with torch.autocast("cuda", enabled=DEV == "cuda"):
            f = net[0](x); P.append(torch.sigmoid(net[2](net[1](f)).squeeze(1)).float().cpu()); F.append(f.float().cpu()); L.append(t)
    P, L = torch.cat(P).numpy(), torch.cat(L).numpy()
    return (P, torch.cat(F).numpy(), L) if feats else (P, L)

def val_stats(net, ld):
    P, L = predict(net, ld)
    return ((P > .5) == L).mean(), float(-np.mean(L * np.log(P + 1e-7) + (1 - L) * np.log(1 - P + 1e-7)))

def train_stage(net, ltr, lva, params, epochs, backbone_eval, tag):
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.3, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best, best_sd, bad = (-1, 0), None, 0
    for ep in range(epochs):
        net.train()
        for mod in net[0].modules():
            if isinstance(mod, BN): mod.eval()
        if backbone_eval: net[0].eval()
        run_epoch(net, ltr, opt, scaler)
        acc, loss = val_stats(net, lva); sched.step(acc)
        print(f"  [{tag}] ep {ep+1:02d} val_acc {acc:.4f} val_loss {loss:.4f}")
        if (acc, -loss) > best: best, best_sd, bad = (acc, -loss), copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(best_sd); return best[0]

def run_backbone(name):
    print(f"\n===== {name} =====")
    bb, d, size, ms = build(name)
    net = nn.Sequential(bb, nn.Dropout(0.4), nn.Linear(d, 1)).to(DEV)
    ltr, lva, lte = loader(i_tr, size, ms, True), loader(i_va, size, ms, False), loader(i_te, size, ms, False)
    for p in bb.parameters(): p.requires_grad = False
    train_stage(net, ltr, lva, list(net[2].parameters()), EP_HEAD, True, "head")
    for p in bb.parameters(): p.requires_grad = True
    for mod in bb.modules():
        if isinstance(mod, BN):
            for p in mod.parameters(): p.requires_grad = False
    train_stage(net, ltr, lva, [{"params": [p for p in bb.parameters() if p.requires_grad], "lr": 1e-5},
                                {"params": net[2].parameters(), "lr": 1e-4}], EP_FT, False, "finetune")
    r = {}
    for k, idx in (("tr", i_tr), ("va", i_va), ("te", i_te)):
        P, F, _ = predict(net, loader(idx, size, ms, False), feats=True); r["p" + k], r["F" + k] = P, F
    print(f"{name} | val acc {((r['pva']>.5)==yva).mean():.4f} | test acc {((r['pte']>.5)==yte).mean():.4f}")
    print(f"{name} done | elapsed {(time.time()-T0)/60:.1f} min")
    del net, bb; gc.collect(); torch.cuda.empty_cache(); return r

res = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
if res: print("loaded cache:", CACHE, "->", list(res))
for n in MODELS:
    if n not in res:
        res[n] = run_backbone(n); pickle.dump(res, open(CACHE, "wb"))
if TOPN:
    va_acc = {n: float(((res[n]["pva"] > .5) == yva).mean()) for n in MODELS}
    MODELS = sorted(MODELS, key=lambda n: -va_acc[n])[:TOPN]
    print("val acc of all:", {n: round(v, 4) for n, v in va_acc.items()})
    print(f"selected top-{TOPN} by validation:", MODELS)

print("\n--- base models (test) ---")
for n in MODELS: print(f"{n:14s} acc {accuracy_score(yte, res[n]['pte'] > .5):.4f}")
avg = np.mean([res[n]["pte"] for n in MODELS], 0)
print(f"{'prob-avg ens':14s} acc {accuracy_score(yte, avg > .5):.4f}")

Ftr, Fva, Fte = [np.hstack([res[n]["F" + k] for n in MODELS]) for k in ("tr", "va", "te")]
sc = StandardScaler().fit(Ftr); Ftr, Fva, Fte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)

def info_value(F, lab, bins=10):
    iv = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        q = np.unique(np.quantile(F[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        b = np.digitize(F[:, j], q)
        g = np.bincount(b[lab == 0], minlength=len(q) + 1) + 0.5
        m = np.bincount(b[lab == 1], minlength=len(q) + 1) + 0.5
        pg, pm = g / g.sum(), m / m.sum()
        iv[j] = np.sum((pg - pm) * np.log(pg / pm))
    return iv

IVall = info_value(Ftr, ytr)
top = np.argsort(-IVall)[:TOPK]
Ftr, Fva, Fte, IV = Ftr[:, top], Fva[:, top], Fte[:, top], IVall[top]
print(f"\nconcat dims {len(IVall)} -> IV top-{TOPK}")

class FS(ElementwiseProblem):
    def __init__(self): super().__init__(n_var=Ftr.shape[1], n_obj=3, xl=0, xu=1, vtype=bool)
    def _evaluate(self, xv, out, *a, **k):
        m = xv.astype(bool)
        if m.sum() == 0: out["F"] = [1.0, 0.0, 1.0]; return
        clf = LogisticRegression(max_iter=200, C=0.1).fit(Ftr[:, m], ytr)
        out["F"] = [m.mean(), -IV[m].sum() / IV.sum(), 1 - clf.score(Fva[:, m], yva)]

r = minimize(FS(), NSGA2(pop_size=POP, sampling=BinaryRandomSampling(), crossover=TwoPointCrossover(prob=0.9),
                         mutation=BitflipMutation(prob=0.1, prob_var=0.01), eliminate_duplicates=True),
             ("n_gen", GEN), seed=SEED, verbose=True)
X, Fp = np.atleast_2d(r.X), np.atleast_2d(r.F)
print(f"NSGA-II done | elapsed {(time.time()-T0)/60:.1f} min")
mask = X[np.lexsort((Fp[:, 0], Fp[:, 2]))[0]].astype(bool)
print(f"Pareto size {len(Fp)} | selected {mask.sum()}/{len(mask)}")
Str, Sva, Ste = [torch.tensor(a[:, mask], dtype=torch.float32, device=DEV) for a in (Ftr, Fva, Fte)]

class GaussianMF(nn.Module):
    def __init__(s, n, k=2):
        super().__init__(); s.c = nn.Parameter(torch.empty(n, k).uniform_(-1.5, 1.5)); s.sig = nn.Parameter(torch.ones(n, k))
    def forward(s, x): return torch.exp(-(x.unsqueeze(-1) - s.c) ** 2 / (2 * s.sig ** 2 + 1e-3)).flatten(1)

class Fusion(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.fz = nn.Sequential(GaussianMF(n), nn.Linear(2 * n, 32), nn.Sigmoid())
        s.dn = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 64), nn.ReLU(),
                             nn.Dropout(.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.3))
        s.fu = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.fu(torch.cat([s.fz(x), s.dn(x)], 1)).squeeze(1)

seed_all(SEED)
head = Fusion(Str.shape[1]).to(DEV); opt = torch.optim.Adam(head.parameters(), 1e-3)
Ttr = torch.tensor(ytr, dtype=torch.float32, device=DEV)
best, best_sd, bad = -1, None, 0
for ep in range(150):
    head.train(); perm = torch.randperm(len(Str), device=DEV)
    for i in range(0, len(perm), 32):
        b = perm[i:i + 32]; loss = Fn.binary_cross_entropy_with_logits(head(Str[b]), Ttr[b] * .95 + .025)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad(): va = ((torch.sigmoid(head(Sva)).cpu().numpy() > .5) == yva).mean()
    if va > best: best, best_sd, bad = va, copy.deepcopy(head.state_dict()), 0
    else:
        bad += 1
        if bad >= 20: break
head.load_state_dict(best_sd); head.eval()
with torch.no_grad(): pred = (torch.sigmoid(head(Ste)).cpu().numpy() > .5).astype(int)

p, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
print(f"\nIMAGE-LEVEL split | {MAG}X | test n={len(yte)} | best val acc (head) {best:.4f}")
print("--- base models ---")
for n in MODELS: print(f"{n:14s} acc {accuracy_score(yte, res[n]['pte'] > .5):.4f}")
print(f"{'prob-avg ens':14s} acc {accuracy_score(yte, avg > .5):.4f}")
print(f"{'NSGA-II+fusion':14s} acc {accuracy_score(yte, pred):.4f} | P {p:.4f} R {rc:.4f} F1 {f1:.4f}")
print(confusion_matrix(yte, pred))

import csv
with open(RESULTS, "a", newline="") as fh:
    csv.writer(fh).writerow([SEED, accuracy_score(yte, avg > .5), accuracy_score(yte, pred), p, rc, f1])
rows = np.loadtxt(RESULTS, delimiter=",", ndmin=2)
sd = rows[:, 2].std(ddof=1) * 100 if len(rows) > 1 else 0.0
print(f"\nseeds logged: {len(rows)} | NSGA-II+fusion acc {rows[:,2].mean()*100:.2f} +/- {sd:.2f} | prob-avg ens {rows[:,1].mean()*100:.2f}")
print("ResQ-50 paper, 400X: 98.49 +/- 0.65 | BCHI-CovNet 98.87 | Swin-V2 98.88")
print(f"seed {SEED} | PRESET={PRESET} | total {(time.time()-T0)/60:.1f} min")

import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
df = pd.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "train.parquet")
X = df.drop(columns=["seq_ix","step_in_seq","need_prediction"]).values.astype(np.float64).reshape(-1,1000,32)
print("shape", X.shape, "nan", np.isnan(X).sum())
print("global mean abs, std per feat (min/med/max):", np.abs(X).mean().round(3), np.round([X.std((0,1)).min(), np.median(X.std((0,1))), X.std((0,1)).max()],3))
# need_prediction layout
npred = df.groupby("seq_ix").need_prediction.sum().unique(); print("need_prediction per seq", npred)
# targets: steps 100..998 predicted from previous step
def pooled(pred, true):
    return np.mean([r2_score(true[:,i], pred[:,i]) for i in range(32)]), [r2_score(true[:,i], pred[:,i]) for i in range(32)]
tgt = X[:,100:999,:].reshape(-1,32)
prev = X[:,99:998,:].reshape(-1,32)
m, per = pooled(prev, tgt); print("persistence R2", round(m,4), "feat min/max", round(min(per),3), round(max(per),3))
# mean-reversion toward global mean: shrink factor
mu = X.reshape(-1,32).mean(0)
for a in (0.9,0.95,0.98):
    print("shrink",a, round(pooled(mu+a*(prev-mu), tgt)[0],4))
# lag-1 autocorr of levels vs diffs (within sequence)
def ac(A, lag=1):
    a=A[:,:-lag,:]; b=A[:,lag:,:]
    a=a-a.mean((0,1)); b=b-b.mean((0,1))
    return (a*b).sum((0,1))/np.sqrt((a**2).sum((0,1))*(b**2).sum((0,1)))
D = np.diff(X,axis=1)
print("lag1 AC levels (min/med/max)", np.round([ac(X).min(), np.median(ac(X)), ac(X).max()],3))
print("lag1 AC diffs  (min/med/max)", np.round([ac(D).min(), np.median(ac(D)), ac(D).max()],3))
print("lag1 AC diffs per feature:", np.round(ac(D),2).tolist())
print("lag 2,5,10 AC diffs median:", [round(float(np.median(ac(D,l))),3) for l in (2,5,10)])
# variance across time: stationarity
for lo,hi in ((0,100),(100,500),(500,999)):
    print("std steps",lo,hi, round(float(X[:,lo:hi].std()),3), " mean", round(float(X[:,lo:hi].mean()),3))
# cross-sectional: per-sequence mean variation vs within-seq std
print("between-seq std of seq means / within-seq std (median):", round(float(np.median(X.mean(1).std(0)/X.std(1).mean(0))),3))
# feature correlation
C = np.corrcoef(X.reshape(-1,32).T); off = C[~np.eye(32,dtype=bool)]
print("feature corr |off-diag| mean", round(np.abs(off).mean(),3), "max", round(np.abs(off).max(),3))
ev = np.linalg.eigvalsh(np.cov(X.reshape(-1,32).T))[::-1]; print("PCA var explained top1/3/10:", np.round(np.cumsum(ev)/ev.sum(),3)[[0,2,9]])
# ridge on last k steps, split by sequence 80/20
rng = np.random.RandomState(29); idx = rng.permutation(517); tr, va = idx[:413], idx[413:]
def feats(Xs,k):
    n=Xs.shape[0]; rows=[]; ys=[]
    for t in range(100,999):
        rows.append(Xs[:,t-k:t,:].reshape(n,-1)); ys.append(Xs[:,t,:])
    return np.concatenate(rows), np.concatenate(ys)
for k in (1,5,20):
    Ftr,ytr = feats(X[tr],k); Fva,yva = feats(X[va],k)
    for al in (1.0,):
        r = Ridge(alpha=al).fit(Ftr,ytr); p=r.predict(Fva)
        pers = Fva[:,-32:]
        print(f"ridge k={k} val R2", round(pooled(p,yva)[0],4), " persistence on same val", round(pooled(pers,yva)[0],4))

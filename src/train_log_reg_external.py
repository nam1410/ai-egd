import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupKFold

data = np.load('dinov2_features.npz')
X, y, patients = data['X'], data['y'], data['patients']

groups = patients
gkf = GroupKFold(n_splits=5)
all_preds=[]; all_true=[]

for fold,(tr,va) in enumerate(gkf.split(X,y,groups)):
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[va])[:,1]
    preds = (probs>=0.5).astype(int)
    all_preds.append(preds); all_true.append(y[va])
    print(f"Fold {fold} AUC={roc_auc_score(y[va], probs):.3f}")

Yt = np.concatenate(all_true); Yp = np.concatenate(all_preds)
print(classification_report(Yt,Yp))
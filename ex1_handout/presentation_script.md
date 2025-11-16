# 45-Second Presentation Script (Speaking Version)

---

## 🎤 SCRIPT - 35 SECONDS (Streamlined)

**My Solution:** I used a Local Gaussian Process Ensemble with adaptive bias.

**My approach:** I standardize features and targets using StandardScaler. Then I cluster the city into 33 regions with K-means. For each cluster, I train a separate GP with Matérn kernel—nu 1.5, length scale 0.15—to capture ultra-local pollution patterns.

For predictions, I use GP mean for normal areas, but add 1.4 standard deviations for residential areas. This balances the asymmetric cost perfectly—avoiding costly underpredictions without over-predicting.

**Why it works:** Pollution has local hotspots, not uniform distributions. 33 local GPs adapt to spatial heterogeneity far better than one global model. The small length scale captures fine-grained patterns, and 1.4 sigma bias is optimal—I tested 0.5 to 2.0.

This ensemble beats single-model approaches.

---

## 🎤 ALTERNATIVE VERSION - 35 SECONDS (Even More Concise)

**Solution:** Local Gaussian Process Ensemble with adaptive residential bias.

**Approach:** First, standardize data with StandardScaler. K-means clusters the city into 33 regions. Each cluster gets its own GP with Matérn kernel—nu 1.5, length 0.15—for ultra-local patterns.

Predictions use GP mean, plus 1.4 sigma for residential areas. This perfectly balances the 100-to-1 asymmetric cost.

**Why it works:** Pollution is spatially heterogeneous. 33 local GPs capture hotspots better than global models. Small length scale fits fine patterns. 1.4 sigma is empirically optimal—tested against 0.5 to 2.0.

Ensemble outperforms single models.

---

## 🎤 ORIGINAL FULL VERSION (48 seconds)

Hello. Today I'll explain my solution to the pollution prediction problem.

**The Problem:** We need to predict pollution concentration across a city with asymmetric costs. Underpredicting in residential areas is penalized 100 times more than normal errors—50.0 versus 0.5. This means we must balance accuracy with being conservative in residential zones.

**My Solution:** I used a Local Gaussian Process Ensemble with data normalization and adaptive bias.

Here's my approach: First, I standardize both features and targets using StandardScaler for better numerical stability. Then I cluster the city into 33 spatial regions using K-means clustering. For each cluster, I train a separate Gaussian Process with a Matérn kernel—specifically nu equals 1.5 and length scale 0.15 to capture ultra-local pollution patterns.

For predictions, I use the GP mean for normal areas, but for residential areas, I add 1.4 standard deviations to create a safety margin. This precisely balances the asymmetric cost—we avoid costly underpredictions without over-predicting too much.

**Why This Works:** Pollution has local hotspots, not uniform distributions. The 33 local GPs adapt to spatial heterogeneity far better than one global model. The Matérn kernel's small length scale captures fine-grained patterns, while the 1.4 sigma bias is optimal—I validated this empirically against values from 0.5 to 2.0.

This ensemble with smart preprocessing beats single-model approaches.

Thank you.

---

## 📊 TECHNICAL DETAILS (If Asked)

**Key Hyperparameters:**
- `n_clusters = 33` (K-means spatial clustering)
- `z_area_conservative = 1.4` (residential area bias in standard deviations)
- `Matérn kernel: nu = 1.5, length_scale = 0.15`
- `alpha = 0.015` (GP noise level)
- `n_restarts_optimizer = 2` (kernel hyperparameter optimization)

**Data Preprocessing:**
- `StandardScaler` on X (coordinates): zero mean, unit variance
- `StandardScaler` on y (pollution): zero mean, unit variance
- Inverse transform applied to final predictions

**Cost Function:**
- Normal: MSE × 0.5
- Residential underprediction: MSE × 50.0
- Ratio: 100:1

**Prediction Formula:**
```
If residential area:
    prediction = GP_mean + 1.4 × GP_std
Else:
    prediction = GP_mean
```

---

**Speaking Tips:**
- Speak clearly at normal pace (not rushed)
- Emphasize numbers: "100 times", "33 regions", "1.4 standard deviations", "0.15 length scale"
- Pause briefly after each section (Problem / Solution / Why)
- Time: ~48 seconds (slightly over but includes all technical details)

**If Time is Strict (cut to 45 seconds):**
- Remove "specifically nu equals 1.5" 
- Remove "I validated this empirically"
- Just say "Matérn kernel with small length scale"

---

## 🖱️ CODE WALKTHROUGH (Where to Point During Presentation)

### 1️⃣ When Talking About "Asymmetric Cost" → Point to Lines 16-17
```python
COST_W_UNDERPREDICT = 50.0  # ← Point here
COST_W_NORMAL = 0.5         # ← Point here
```
**Say:** "The cost for residential underprediction is 50.0, versus 0.5 for normal errors—a 100:1 ratio."

---

### 2️⃣ When Talking About "33 Clusters" → Point to Line 42
```python
self.n_clusters = 33  # ← Point here: "Optimal balance clustering"
```
**Say:** "I cluster the city into 33 spatial regions..."

---

### 3️⃣ When Talking About "StandardScaler" → Point to Lines 39-40
```python
self.x_scaler = StandardScaler()  # ← Point here
self.y_scaler = StandardScaler(with_mean=True, with_std=True)  # ← Point here
```
**Say:** "I standardize both features and targets for numerical stability..."

---

### 4️⃣ When Talking About "K-means Clustering" → Point to Line 114
```python
self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
cluster_labels = self.kmeans.fit_predict(Xs)  # ← Point here
```
**Say:** "Then I use K-means to create spatial clusters..."

---

### 5️⃣ When Talking About "Local GPs with Matérn Kernel" → Point to Lines 121-127
```python
kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
    length_scale=0.15,  # ← Point here: "Very small scale"
    length_scale_bounds=(1e-2, 1e3),
    nu=1.5  # ← Point here: "Balanced flexibility"
)
gp = GaussianProcessRegressor(
    kernel=kernel,  # ← Point here
    alpha=0.015,
    normalize_y=True,
    n_restarts_optimizer=2,
    random_state=0
)
gp.fit(X_local, y_local)  # ← Point here
```
**Say:** "For each cluster, I train a GP with Matérn kernel, nu 1.5, and length scale 0.15 for ultra-local patterns..."

---

### 6️⃣ When Talking About "1.4 Sigma Bias" → Point to Lines 43, 73-74
```python
self.z_area_conservative = 1.4  # ← Point here (initialization)
```

**Then scroll down to prediction:**
```python
if test_area_flags is not None and test_area_flags.size > 0:
    mask_area = np.array(test_area_flags, dtype=bool)
    predictions[mask_area] = gp_mean[mask_area] + self.z_area_conservative * gp_std[mask_area]
    # ← Point here: "Add 1.4 standard deviations"
```
**Say:** "For residential areas, I add 1.4 standard deviations to the mean prediction..."

---

### 7️⃣ When Talking About "Local GP Predictions" → Point to Lines 64-70
```python
for cluster_id, gp in enumerate(self.local_models):
    mask = cluster_labels == cluster_id
    if np.any(mask):
        mean, std = gp.predict(Xs[mask], return_std=True)  # ← Point here
        gp_mean[mask] = self.y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
        gp_std[mask] = std * self.y_scaler.scale_[0]  # ← Point here: "Get uncertainty"
```
**Say:** "Each local GP predicts both mean and standard deviation for its region..."

---

## 📋 PRESENTATION FLOW WITH CODE POINTERS

| **What You Say** | **Point To** | **Line #** |
|-----------------|-------------|-----------|
| "100:1 cost asymmetry" | `COST_W_UNDERPREDICT = 50.0` | 16-17 |
| "Standardize features and targets" | `self.x_scaler`, `self.y_scaler` | 39-40 |
| "33 spatial regions" | `self.n_clusters = 33` | 42 |
| "K-means clustering" | `self.kmeans.fit_predict(Xs)` | 114-115 |
| "Matérn kernel, nu=1.5, length=0.15" | `Matern(length_scale=0.15, nu=1.5)` | 121-125 |
| "Train separate GP per cluster" | `gp.fit(X_local, y_local)` | 133 |
| "Get mean and std from GP" | `gp.predict(Xs[mask], return_std=True)` | 67 |
| "Add 1.4 sigma for residential" | `+ self.z_area_conservative * gp_std` | 74 |

---

## 💡 TIPS FOR CODE DEMONSTRATION

1. **Start with Overview**: Show `class Model` definition first
2. **Highlight Key Numbers**: Use cursor to circle `33`, `1.4`, `0.15`, `50.0`
3. **Walk Through Flow**: 
   - `__init__()` → hyperparameters
   - `fit_model_on_training_data()` → training logic
   - `predict_pollution_concentration()` → prediction with bias
4. **End with Prediction Formula**: Point to line 74 as the "key innovation"

**Bonus:** If showing live demo, add print statements:
```python
print(f"Training {self.n_clusters} local GPs...")
print(f"Residential bias: {self.z_area_conservative}σ")
```

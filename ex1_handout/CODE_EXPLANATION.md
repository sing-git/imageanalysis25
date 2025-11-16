# 🔍 PCA + Vector Quantization 코드 상세 설명

## 📦 Import & Class Definition

```python
import numpy as np
```
**왜?** NumPy는 행렬 연산, SVD, 통계 함수를 제공. 외부 라이브러리 사용 불가 제약 때문에 NumPy만 사용.

```python
class ImageCompressor:
```
**왜?** 클라이언트 측에서 이미지를 압축하는 역할. 학습(train)과 압축(compress) 기능 분리.

---

## 🎯 `__init__`: 초기화

```python
def __init__(self, n_components=50, n_clusters=64, use_vector_quantization=True, max_kmeans_iters=50):
```
**왜?** 
- `n_components=50`: PCA로 27,648차원을 50차원으로 축소 (기본값)
- `n_clusters=64`: K-means로 50차원 벡터를 64개 클러스터로 압축
- `use_vector_quantization=True`: VQ 사용 여부 (실험용 플래그)
- `max_kmeans_iters=50`: K-means가 너무 오래 돌지 않도록 제한

```python
self.n_components = n_components
self.n_clusters = n_clusters
self.use_vector_quantization = use_vector_quantization
self.max_kmeans_iters = max_kmeans_iters
```
**왜?** 파라미터를 인스턴스 변수로 저장해서 나중에 train(), compress()에서 사용.

```python
self.mean_image = None
self.components = None
self.explained_variance = None
```
**왜?** PCA 학습 결과를 저장할 변수들. 초기값은 None (아직 학습 안 함).
- `mean_image`: 모든 학습 이미지의 평균 (27,648 벡터)
- `components`: PCA 주성분들 (n_components × 27,648 행렬)
- `explained_variance`: 각 주성분이 설명하는 분산 비율

```python
self.vq_centroids = None
```
**왜?** Vector Quantization 클러스터 중심점들 저장 (n_clusters × n_components 행렬).

---

## 🔄 `_kmeans`: K-means 클러스터링 (NumPy 구현)

```python
def _kmeans(self, X, k, max_iters=50):
```
**왜?** sklearn 사용 불가 → NumPy로 K-means 직접 구현. `_`는 private 메서드 의미.

```python
N, d = X.shape
```
**왜?** N = 데이터 개수, d = 차원 (예: 100개 이미지, 50차원).

```python
np.random.seed(42)
init_indices = np.random.choice(N, k, replace=False)
centroids = X[init_indices].copy()
```
**왜?** 
- `seed(42)`: 재현 가능한 랜덤 (디버깅 용이)
- `np.random.choice(N, k, replace=False)`: N개 중 k개 랜덤 선택 (중복 X)
- 초기 중심점을 랜덤하게 선택 (K-means 표준 방법)

```python
for iteration in range(max_iters):
```
**왜?** K-means는 반복 알고리즘. 최대 50번 반복.

```python
distances = np.zeros((N, k))
for i in range(k):
    diff = X - centroids[i]
    distances[:, i] = np.sum(diff ** 2, axis=1)
```
**왜?** 
- 각 데이터와 모든 중심점 간의 유클리드 거리^2 계산
- `X - centroids[i]`: Broadcasting으로 (N, d) - (d,) = (N, d)
- `np.sum(..., axis=1)`: 각 행(데이터)마다 거리 합산 → (N,)
- 결과: `distances[j, i]` = j번째 데이터와 i번째 중심점 간 거리

```python
labels = np.argmin(distances, axis=1)
```
**왜?** 각 데이터를 가장 가까운 중심점에 할당. `labels[j]` = j번째 데이터가 속한 클러스터 번호.

```python
new_centroids = np.zeros_like(centroids)
for i in range(k):
    cluster_points = X[labels == i]
    if len(cluster_points) > 0:
        new_centroids[i] = np.mean(cluster_points, axis=0)
    else:
        new_centroids[i] = centroids[i]
```
**왜?** 
- 각 클러스터에 속한 데이터들의 평균으로 중심점 업데이트
- `X[labels == i]`: i번 클러스터에 속한 모든 데이터
- `if len(cluster_points) > 0`: 빈 클러스터 방지 (있으면 기존 중심점 유지)

```python
centroid_shift = np.sum((new_centroids - centroids) ** 2)
centroids = new_centroids
```
**왜?** 
- 중심점이 얼마나 움직였는지 측정 (수렴 체크용)
- 중심점 업데이트

```python
if centroid_shift < 1e-6:
    print(f"    Converged at iteration {iteration}")
    break
```
**왜?** 중심점이 거의 안 움직이면 수렴한 것 → 조기 종료 (시간 절약).

---

## 📚 `get_codebook`: 코드북 생성

```python
if self.mean_image is None or self.components is None:
    return np.array([])
```
**왜?** 학습 안 했으면 빈 배열 반환 (에러 방지).

```python
if self.use_vector_quantization:
```
**왜?** VQ 사용 여부에 따라 코드북 구조가 다름.

```python
metadata = np.array([
    self.n_components,
    self.n_clusters
], dtype=np.float32)
```
**왜?** 
- 복원 시 파싱하기 위해 파라미터 정보 저장
- 코드북 앞부분에 메타데이터 삽입

```python
codebook = np.concatenate([
    metadata,
    self.mean_image.flatten(),
    self.components.flatten(),
    self.vq_centroids.flatten()
])
```
**왜?** 
- 모든 정보를 1차원 배열로 합침
- **구조**: [n_components, n_clusters] + [평균 이미지 27,648] + [주성분 n_components×27,648] + [VQ 중심점 n_clusters×n_components]
- 예: [50, 64] + [27,648] + [50×27,648] + [64×50] = 총 1,384,402개 값

```python
return codebook.astype(np.float16)
```
**왜?** 
- float32 → float16: 메모리 절반으로 절약 (정확도 약간 손실, 압축에선 OK)
- 코드북 크기: ~2.6MB (float16) vs ~5.2MB (float32)

---

## 🎓 `train`: PCA + VQ 학습

### STEP 1: PCA 학습

```python
image_vectors = []
for img in train_images:
    img_vector = img.astype(np.float32).flatten()
    image_vectors.append(img_vector)
```
**왜?** 
- 각 이미지 (96×96×3) → 1차원 벡터 (27,648)
- `astype(np.float32)`: uint8 (0-255) → float32 (PCA는 실수 연산 필요)

```python
X = np.array(image_vectors)  # (N, 27648)
```
**왜?** 리스트 → NumPy 배열로 변환. 행렬 연산 가능.

```python
self.mean_image = np.mean(X, axis=0)
X_centered = X - self.mean_image
```
**왜?** 
- PCA는 데이터를 중심화(평균=0)해야 함 (이론적 요구사항)
- `axis=0`: 각 열(픽셀)의 평균 계산 → (27,648,)
- `X - self.mean_image`: Broadcasting으로 각 이미지에서 평균 빼기

```python
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
```
**왜?** 
- **SVD (Singular Value Decomposition)**: X = U × S × Vt
- PCA는 SVD의 특수 케이스
- `full_matrices=False`: 메모리 절약 (작은 행렬만 반환)
- **결과**:
  - `U`: (N, N) 또는 (N, min(N, d))
  - `S`: (min(N, d),) - 특이값들 (중요도 순서)
  - `Vt`: (min(N, d), d) - **주성분 벡터들** (우리가 원하는 것!)

```python
explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
cumsum_variance = np.cumsum(explained_variance_ratio)
```
**왜?** 
- 각 주성분이 설명하는 분산 비율 계산 (중요도 측정)
- `S ** 2`: 특이값 → 분산으로 변환
- `cumsum`: 누적 합 (예: 첫 50개 주성분이 전체 분산의 95% 설명)

```python
if self.n_components > len(S):
    self.n_components = len(S)
```
**왜?** 주성분 개수가 데이터보다 많으면 조정 (에러 방지).

```python
self.components = Vt[:self.n_components]  # (n_components, 27648)
```
**왜?** 상위 n_components개 주성분만 선택 (차원 축소).

### STEP 2: Vector Quantization 학습

```python
pca_coefficients = []
for img_vector in image_vectors:
    img_centered = img_vector - self.mean_image
    coeffs = np.dot(self.components, img_centered)
    pca_coefficients.append(coeffs)
```
**왜?** 
- 각 학습 이미지를 PCA 공간으로 변환
- `np.dot(self.components, img_centered)`: (n_components, 27648) × (27648,) = (n_components,)
- 27,648차원 → n_components차원 (예: 50차원)

```python
pca_coeffs_matrix = np.array(pca_coefficients)  # (N, n_components)
```
**왜?** 리스트 → 행렬 변환. K-means 입력용.

```python
self.vq_centroids = self._kmeans(
    pca_coeffs_matrix, 
    self.n_clusters, 
    max_iters=self.max_kmeans_iters
)
```
**왜?** 
- PCA 계수들을 n_clusters개로 클러스터링
- **핵심 아이디어**: 비슷한 PCA 계수를 가진 이미지들을 같은 클러스터로 묶기
- 압축 시 클러스터 번호만 저장 (1 byte)

---

## 🗜️ `compress`: 이미지 압축

```python
if self.mean_image is None or self.components is None:
    raise ValueError("Model has not been trained yet!")
```
**왜?** 학습 안 했으면 에러 발생 (안전장치).

```python
img_vector = test_image.astype(np.float32).flatten()
img_centered = img_vector - self.mean_image
```
**왜?** 
- 이미지를 벡터로 변환
- 학습 때와 동일하게 중심화 (평균 빼기)

```python
pca_coefficients = np.dot(self.components, img_centered)
```
**왜?** 
- PCA 변환: 27,648차원 → n_components차원
- 이게 기본 압축 결과 (VQ 안 쓰면 이걸 반환)

```python
distances = np.zeros(self.n_clusters)
for i in range(self.n_clusters):
    diff = pca_coefficients - self.vq_centroids[i]
    distances[i] = np.sum(diff ** 2)
```
**왜?** 
- 현재 PCA 계수와 모든 클러스터 중심점 간 거리 계산
- 가장 가까운 클러스터 찾기 위함

```python
cluster_idx = np.argmin(distances)
```
**왜?** 가장 가까운 클러스터의 인덱스 찾기.

```python
if self.n_clusters <= 256:
    return np.array([cluster_idx], dtype=np.uint8)
else:
    return np.array([cluster_idx], dtype=np.uint16)
```
**왜?** 
- **압축 핵심!** 27,648 bytes → **1 byte** (또는 2 bytes)
- 256개 이하: uint8 (0-255, 1 byte)
- 257개 이상: uint16 (0-65535, 2 bytes)

---

## 🔧 `ImageReconstructor`: 이미지 복원

### `__init__`

```python
self.codebook = codebook
if len(codebook) > 0:
    self._parse_codebook()
```
**왜?** 코드북 받아서 바로 파싱 (서버 측에서 사용).

### `_parse_codebook`

```python
img_size = 96 * 96 * 3  # 27648
```
**왜?** 이미지 크기 상수 (코드북 파싱 시 사용).

```python
if len(self.codebook) > img_size + 2:
    # VQ 모드
```
**왜?** 
- 코드북 크기로 VQ 사용 여부 판단
- VQ 모드: metadata(2) + mean(27,648) + components + centroids
- PCA 모드: mean(27,648) + components

```python
self.n_components = int(self.codebook[0])
self.n_clusters = int(self.codebook[1])
```
**왜?** 메타데이터에서 파라미터 복원.

```python
start_idx = 2
end_idx = start_idx + img_size
self.mean_image = self.codebook[start_idx:end_idx]
```
**왜?** 
- 인덱스 2~27,649: 평균 이미지
- 슬라이싱으로 추출

```python
start_idx = end_idx
end_idx = start_idx + (self.n_components * img_size)
components_flat = self.codebook[start_idx:end_idx]
self.components = components_flat.reshape(self.n_components, img_size)
```
**왜?** 
- 다음 n_components × 27,648개 값: 주성분들
- 1차원 → 2차원 행렬로 reshape

```python
start_idx = end_idx
vq_flat = self.codebook[start_idx:]
self.vq_centroids = vq_flat.reshape(self.n_clusters, self.n_components)
```
**왜?** 
- 나머지: VQ 중심점들
- (n_clusters, n_components) 행렬로 reshape

### `reconstruct`

```python
if self.use_vq:
    cluster_idx = int(test_code[0])
    pca_coefficients = self.vq_centroids[cluster_idx]
else:
    pca_coefficients = test_code
```
**왜?** 
- VQ 모드: 클러스터 인덱스 → 해당 중심점(PCA 계수) 가져오기
- PCA 모드: 계수 직접 사용

```python
reconstructed_vector = self.mean_image.copy()
for i, coeff in enumerate(pca_coefficients):
    reconstructed_vector += coeff * self.components[i]
```
**왜?** 
- **PCA 역변환 공식**: 이미지 = 평균 + Σ(계수_i × 주성분_i)
- 각 주성분에 계수 곱해서 합산

```python
reconstructed_image = reconstructed_vector.reshape(96, 96, 3)
```
**왜?** 1차원 벡터 → 3차원 이미지로 변환.

```python
reconstructed_image = np.clip(reconstructed_image, 0, 255)
```
**왜?** 
- PCA 역변환 결과가 [0, 255] 범위 벗어날 수 있음
- 음수나 255 초과 값을 0과 255로 제한

```python
reconstructed_image = self._quantize_colors(reconstructed_image)
```
**왜?** 틱택토 특성 반영: 4가지 색상으로 양자화.

### `_quantize_colors`

```python
colors = np.array([
    [255, 255, 255],  # 흰색 (배경)
    [0, 0, 0],        # 검은색 (격자)
    [255, 0, 0],      # 빨간색 (X)
    [0, 255, 0]       # 녹색 (O)
])
```
**왜?** 틱택토는 4가지 색상만 사용 → 대표 색상 정의.

```python
distances = np.sum((colors - pixel) ** 2, axis=1)
closest_color_idx = np.argmin(distances)
quantized[i, j] = colors[closest_color_idx]
```
**왜?** 
- 각 픽셀을 가장 가까운 대표 색상으로 매핑
- 유클리드 거리 사용 (RGB 공간)

---

## 🎯 전체 흐름 요약

### 학습 단계:
1. **이미지 → 벡터** (96×96×3 → 27,648)
2. **PCA**: 27,648차원 → 50차원
3. **K-means**: 50차원 벡터들을 64개 클러스터로 그룹화
4. **코드북 생성**: 평균 + 주성분 + 클러스터 중심점 저장

### 압축 단계:
1. **이미지 → 벡터** (27,648)
2. **PCA 변환** (27,648 → 50차원)
3. **가장 가까운 클러스터 찾기**
4. **클러스터 인덱스 저장** (1 byte!)

### 복원 단계:
1. **클러스터 인덱스 → PCA 계수** (1 byte → 50개 float)
2. **PCA 역변환** (50차원 → 27,648차원)
3. **벡터 → 이미지** (27,648 → 96×96×3)
4. **4색 양자화** (틱택토 특성)

---

## 💡 핵심 아이디어

**왜 PCA + VQ가 효과적인가?**

1. **PCA**: 27,648차원은 너무 크다 → 50차원으로 축소 (99% 정보 유지)
2. **VQ**: 50개 float(100 bytes) 저장도 크다 → 클러스터 번호(1 byte)로 대체
3. **압축 비율**: 27,648 bytes → **1 byte** (27,648:1!)

**트레이드오프**:
- 압축률 ↑ (1 byte 매우 작음)
- 품질 ↓ (비슷한 이미지들이 같은 클러스터로 묶임 → 세부사항 손실)

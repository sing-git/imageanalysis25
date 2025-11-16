import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self, n_components=50, n_clusters=64, use_vector_quantization=True, max_kmeans_iters=50):
        """
        PCA + Vector Quantization 하이브리드 압축
        
        Args:
            n_components: PCA 주성분 개수 (차원 축소)
            n_clusters: VQ 클러스터 개수 (코드북 크기)
            use_vector_quantization: VQ 사용 여부
            max_kmeans_iters: K-means 최대 반복 횟수
        """
        
        self.n_components = n_components
    
        self.n_clusters = n_clusters
        self.use_vector_quantization = use_vector_quantization
        self.max_kmeans_iters = max_kmeans_iters
        
        # PCA 관련 변수들
        self.mean_image = None
        self.components = None  # 주성분들
        self.explained_variance = None
        
        # Vector Quantization 관련 변수들
        self.vq_centroids = None  # VQ 코드북 (클러스터 중심점들)

    def _kmeans(self, X, k, max_iters=50):
        """
        NumPy만 사용한 K-means 클러스터링 구현
        
        Args:
            X: (N, d) 데이터 행렬
            k: 클러스터 개수
            max_iters: 최대 반복 횟수
            
        Returns:
            centroids: (k, d) 클러스터 중심점들
        """
        N, d = X.shape
        
        # 1. 초기 중심점: 랜덤하게 k개 샘플 선택
        np.random.seed(42)
        init_indices = np.random.choice(N, k, replace=False)
        centroids = X[init_indices].copy()
        
        print(f"  K-means clustering: {k} clusters, {max_iters} max iterations...")
        
        for iteration in range(max_iters):
            # 2. 각 데이터를 가장 가까운 중심점에 할당
            # 거리 계산: (N, k)
            distances = np.zeros((N, k))
            for i in range(k):
                diff = X - centroids[i]
                distances[:, i] = np.sum(diff ** 2, axis=1)
            
            labels = np.argmin(distances, axis=1)
            
            # 3. 중심점 업데이트
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = np.mean(cluster_points, axis=0)
                else:
                    # 빈 클러스터: 기존 중심점 유지
                    new_centroids[i] = centroids[i]
            
            # 4. 수렴 체크
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            
            if iteration % 10 == 0:
                print(f"    Iteration {iteration}: centroid shift = {centroid_shift:.6f}")
            
            if centroid_shift < 1e-6:
                print(f"    Converged at iteration {iteration}")
                break
        
        return centroids

    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        
        if self.mean_image is None or self.components is None:
            return np.array([])
        
        if self.use_vector_quantization:
            # VQ 모드: 평균 이미지 + 주성분들 + VQ 중심점들
            if self.vq_centroids is None:
                return np.array([])
            
            # 크기 정보도 함께 저장 (복원 시 파싱하기 위해)
            metadata = np.array([
                self.n_components,
                self.n_clusters
            ], dtype=np.float32)
            
            codebook = np.concatenate([
                metadata,
                self.mean_image.flatten(),
                self.components.flatten(),
                self.vq_centroids.flatten()
            ])
            
            # 메모리 절약: float16으로 저장
            return codebook.astype(np.float16)
        else:
            # 기존 PCA 모드
            codebook = np.concatenate([
                self.mean_image.flatten(), 
                self.components.flatten()
            ])
            return codebook.astype(np.float16)

    def train(self, train_images):
        """
        Training phase: PCA + Vector Quantization
        
        Args:
            train_images: A list of NumPy arrays (H x W x C)
        """
        N = len(train_images)
        print(f"Training PCA + VQ with {N} images...")
        
        # === STEP 1: PCA 학습 ===
        print("\n[STEP 1] PCA Training...")
        
        # 1-1. 모든 이미지를 벡터로 변환
        image_vectors = []
        for img in train_images:
            img_vector = img.astype(np.float32).flatten()
            image_vectors.append(img_vector)
        
        X = np.array(image_vectors)  # (N, 27648)
        print(f"  Data matrix: {X.shape}")
        
        # 1-2. 평균 이미지 계산 및 중심화
        self.mean_image = np.mean(X, axis=0)
        X_centered = X - self.mean_image
        
        # 1-3. SVD 기반 PCA
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 설명 가능한 분산 비율
        explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
        cumsum_variance = np.cumsum(explained_variance_ratio)
        
        # n_components 조정
        if self.n_components > len(S):
            self.n_components = len(S)
        
        self.components = Vt[:self.n_components]  # (n_components, 27648)
        self.explained_variance = explained_variance_ratio[:self.n_components]
        
        print(f"  PCA completed: {self.n_components} components")
        print(f"  Cumulative variance explained: {cumsum_variance[self.n_components-1]:.4f}")
        
        # === STEP 2: Vector Quantization 학습 ===
        if self.use_vector_quantization:
            print(f"\n[STEP 2] Vector Quantization Training...")
            
            # 2-1. 모든 학습 이미지를 PCA 공간으로 변환
            pca_coefficients = []
            for img_vector in image_vectors:
                img_centered = img_vector - self.mean_image
                coeffs = np.dot(self.components, img_centered)
                pca_coefficients.append(coeffs)
            
            pca_coeffs_matrix = np.array(pca_coefficients)  # (N, n_components)
            print(f"  PCA coefficients matrix: {pca_coeffs_matrix.shape}")
            
            # 2-2. K-means로 PCA 계수들을 클러스터링
            self.vq_centroids = self._kmeans(
                pca_coeffs_matrix, 
                self.n_clusters, 
                max_iters=self.max_kmeans_iters
            )
            
            print(f"  VQ codebook created: {self.vq_centroids.shape}")
            print(f"\n✅ Training completed!")
            print(f"   - PCA: {self.n_components} components")
            print(f"   - VQ: {self.n_clusters} clusters")
            
        else:
            print(f"\n✅ PCA Training completed!")
            print(f"   - Using {self.n_components} components (no VQ)")

    def compress(self, test_image):
        """ 
        Given an array of shape H x W x C, return compressed code 
        
        Returns:
            If VQ enabled: cluster index (uint8, 1 byte)
            If VQ disabled: PCA coefficients (float16, n_components * 2 bytes)
        """

        if self.mean_image is None or self.components is None:
            raise ValueError("Model has not been trained yet!")
        
        # 1. 이미지를 벡터로 변환하고 중심화
        img_vector = test_image.astype(np.float32).flatten()
        img_centered = img_vector - self.mean_image
        
        # 2. PCA 변환 (주성분에 투영)
        pca_coefficients = np.dot(self.components, img_centered)
        
        if self.use_vector_quantization:
            # 3. Vector Quantization: 가장 가까운 클러스터 찾기
            if self.vq_centroids is None:
                raise ValueError("VQ centroids not trained!")
            
            # 모든 클러스터와의 거리 계산
            distances = np.zeros(self.n_clusters)
            for i in range(self.n_clusters):
                diff = pca_coefficients - self.vq_centroids[i]
                distances[i] = np.sum(diff ** 2)
            
            # 가장 가까운 클러스터의 인덱스 반환
            cluster_idx = np.argmin(distances)
            
            # 압축 결과: 단 1 바이트! (0~255)
            if self.n_clusters <= 256:
                return np.array([cluster_idx], dtype=np.uint8)
            else:
                # 256개 이상 클러스터: uint16 사용 (2 bytes)
                return np.array([cluster_idx], dtype=np.uint16)
        else:
            # PCA만 사용: 계수들을 float16으로 반환
            return pca_coefficients.astype(np.float16)


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook
        
        # 코드북에서 정보 파싱
        if len(codebook) > 0:
            self._parse_codebook()
    
    def _parse_codebook(self):
        """ 코드북에서 평균 이미지, 주성분들, VQ 중심점들을 분리 """
        img_size = 96 * 96 * 3  # 27648
        
        # 메타데이터 확인 (첫 2개 값: n_components, n_clusters)
        if len(self.codebook) > img_size + 2:
            # VQ 모드
            self.use_vq = True
            self.n_components = int(self.codebook[0])
            self.n_clusters = int(self.codebook[1])
            
            # 평균 이미지
            start_idx = 2
            end_idx = start_idx + img_size
            self.mean_image = self.codebook[start_idx:end_idx]
            
            # 주성분들
            start_idx = end_idx
            end_idx = start_idx + (self.n_components * img_size)
            components_flat = self.codebook[start_idx:end_idx]
            self.components = components_flat.reshape(self.n_components, img_size)
            
            # VQ 중심점들
            start_idx = end_idx
            vq_flat = self.codebook[start_idx:]
            self.vq_centroids = vq_flat.reshape(self.n_clusters, self.n_components)
            
        else:
            # 기존 PCA 모드
            self.use_vq = False
            self.mean_image = self.codebook[:img_size]
            
            components_flat = self.codebook[img_size:]
            n_components = len(components_flat) // img_size
            self.components = components_flat.reshape(n_components, img_size)

    def reconstruct(self, test_code):
        """ 
        Given a compressed code, reconstruct the original image 
        
        Args:
            test_code: If VQ enabled, cluster index (1 byte)
                      If VQ disabled, PCA coefficients (n_components floats)
        """
        
        if len(self.codebook) == 0:
            return np.zeros((96, 96, 3), dtype=np.uint8)
        
        if self.use_vq:
            # VQ 모드: 클러스터 인덱스로부터 복원
            cluster_idx = int(test_code[0])
            
            # 해당 클러스터의 중심점 (PCA 계수들)
            pca_coefficients = self.vq_centroids[cluster_idx]
            
        else:
            # PCA 모드: 계수들 직접 사용
            pca_coefficients = test_code
        
        # PCA 역변환: 복원된이미지 = 평균 + Σ(계수_i × 주성분_i)
        reconstructed_vector = self.mean_image.copy()
        for i, coeff in enumerate(pca_coefficients):
            reconstructed_vector += coeff * self.components[i]
        
        # 벡터를 이미지 형태로 변환
        reconstructed_image = reconstructed_vector.reshape(96, 96, 3)
        
        # [0, 255] 범위로 클리핑
        reconstructed_image = np.clip(reconstructed_image, 0, 255)
        
        # 4색 양자화 (틱택토 특성)
        reconstructed_image = self._quantize_colors(reconstructed_image)
        
        return reconstructed_image.astype(np.uint8)
    
    def _quantize_colors(self, image):
        """ 틱택토 게임의 4가지 색상으로 양자화 """
        # 간단한 K-means 기반 색상 양자화
        # 실제로는 더 정교한 방법을 사용할 수 있지만, 
        # 여기서는 간단히 임계값 기반으로 처리
        
        # RGB 각 채널에 대해 이진화 비슷한 처리
        quantized = image.copy()
        
        # 각 픽셀을 가장 가까운 대표 색상으로 매핑
        # 흰색(255,255,255), 검은색(0,0,0), 빨간색(255,0,0), 녹색(0,255,0)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j]
                
                # 4가지 대표 색상과의 거리 계산
                colors = np.array([
                    [255, 255, 255],  # 흰색 (배경)
                    [0, 0, 0],        # 검은색 (격자)
                    [255, 0, 0],      # 빨간색 (X)
                    [0, 255, 0]       # 녹색 (O)
                ])
                
                distances = np.sum((colors - pixel) ** 2, axis=1)
                closest_color_idx = np.argmin(distances)
                quantized[i, j] = colors[closest_color_idx]
        
        return quantized
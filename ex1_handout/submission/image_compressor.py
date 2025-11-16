import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self, n_components=50, use_ultra_compression=False, use_enhanced_pca=True):
        """
        Feel free to add any number of parameters here.
        But be sure to set default values. Those will be used on the evaluation server
        """
        
        # 압축에 사용할 주성분의 개수 (이 값을 조정해서 성능 튜닝)
        self.n_components = n_components
        self.use_ultra_compression = use_ultra_compression
        self.use_enhanced_pca = use_enhanced_pca
        
        # 극단적 압축을 위한 데이터 타입 선택
        if use_ultra_compression:
            self.dtype = np.int8  # 더 작은 데이터 타입
        else:
            self.dtype = np.float16
        
        # PCA 관련 변수들
        self.mean_image = None
        self.components = None  # 주성분들
        self.explained_variance = None  # 각 주성분의 설명 가능한 분산


    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        
        # 코드북에는 평균 이미지와 주성분들을 포함
        if self.mean_image is None or self.components is None:
            return np.array([])
        
        # 평균 이미지와 주성분들을 하나의 배열로 합침
        codebook = np.concatenate([
            self.mean_image.flatten(), 
            self.components.flatten()
        ])
        return codebook.astype(self.dtype)

    def train(self, train_images):
        """
        Training phase of your algorithm - e.g. here you can perform PCA on training data
        
        Args:
            train_images  ... A list of NumPy arrays.
                              Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
        """
        N = len(train_images)
        print(f"Training with {N} images...")
        
        # 1. 모든 이미지를 벡터로 변환 (96x96x3 -> 27648)
        image_vectors = []
        for img in train_images:
            # 이미지를 float32로 변환하고 벡터화
            img_vector = img.astype(np.float32).flatten()
            image_vectors.append(img_vector)
        
        # 2. 데이터 행렬 생성 (N x 27648)
        X = np.array(image_vectors)
        
        # 3. 평균 이미지 계산
        self.mean_image = np.mean(X, axis=0)
        
        # 4. 중심화 (평균 제거)
        X_centered = X - self.mean_image
        
        # 5. Enhanced PCA with better numerical stability
        if self.use_enhanced_pca:
            # 개선된 SVD 기반 PCA
            # 수치적 안정성을 위해 정규화 추가
            X_normalized = X_centered / (np.std(X_centered) + 1e-8)
            
            # SVD 수행
            U, S, Vt = np.linalg.svd(X_normalized, full_matrices=False)
            
            # 설명 가능한 분산 계산
            explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
            
            # 누적 분산이 95% 이상인 주성분만 선택 (최적화)
            if self.n_components > len(S):
                self.n_components = len(S)
                
            cumsum_variance = np.cumsum(explained_variance_ratio)
            optimal_components = np.where(cumsum_variance >= 0.95)[0]
            if len(optimal_components) > 0 and optimal_components[0] < self.n_components:
                actual_components = min(self.n_components, optimal_components[0] + 1)
            else:
                actual_components = self.n_components
            
            # 주성분 선택 (Vt가 이미 전치된 상태)
            self.components = Vt[:actual_components]  # (n_components, 27648)
            self.explained_variance = explained_variance_ratio[:actual_components]
            
            print(f"Enhanced PCA completed. Using {actual_components} components (cumvar: {cumsum_variance[actual_components-1]:.3f})")
        
        else:
            # 기존 방식
            U, S, Vt = np.linalg.svd(X_centered.T, full_matrices=False)
            self.components = U[:, :self.n_components].T
            print(f"Standard PCA completed. Using {self.n_components} components.")

    def compress(self, test_image):
        """ Given an array of shape H x W x C return compressed code """

        if self.mean_image is None or self.components is None:
            raise ValueError("Model has not been trained yet!")
        
        # 1. 이미지를 벡터로 변환
        img_vector = test_image.astype(np.float32).flatten()
        
        # 2. 중심화 (평균 제거)
        img_centered = img_vector - self.mean_image
        
        # 3. PCA 변환 (주성분에 투영)
        pca_coefficients = np.dot(self.components, img_centered)
        
        # 4. 극단적 압축을 위한 양자화
        if self.use_ultra_compression:
            # PCA 계수를 -128~127 범위로 양자화 (int8)
            coeffs_scaled = pca_coefficients / np.std(pca_coefficients) * 30  # 스케일링
            coeffs_quantized = np.clip(coeffs_scaled, -127, 127)
            return coeffs_quantized.astype(self.dtype)
        else:
            return pca_coefficients.astype(self.dtype)


class ImageReconstructor:
    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook
        
        # 코드북에서 평균 이미지와 주성분들을 분리
        if len(codebook) > 0:
            self._parse_codebook()
    
    def _parse_codebook(self):
        """ 코드북에서 평균 이미지와 주성분들을 분리 """
        # 이미지 크기: 96 x 96 x 3 = 27648
        img_size = 96 * 96 * 3
        
        # 평균 이미지 추출
        self.mean_image = self.codebook[:img_size]
        
        # 주성분들 추출
        components_flat = self.codebook[img_size:]
        n_components = len(components_flat) // img_size
        self.components = components_flat.reshape(n_components, img_size)

    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """
        
        if len(self.codebook) == 0:
            # 코드북이 비어있으면 임시 이미지 반환
            return np.zeros((96, 96, 3), dtype=np.uint8)
        
        # 1. PCA 계수로부터 이미지 복원
        # 복원 공식: 복원된이미지 = 평균 + Σ(계수_i × 주성분_i)
        reconstructed_vector = self.mean_image.copy()
        for i, coeff in enumerate(test_code):
            reconstructed_vector += coeff * self.components[i]
        
        # 2. 벡터를 이미지 형태로 변환
        reconstructed_image = reconstructed_vector.reshape(96, 96, 3)
        
        # 3. 값의 범위를 [0, 255]로 클리핑하고 정수형으로 변환
        reconstructed_image = np.clip(reconstructed_image, 0, 255)
        
        # 4. 노이즈 제거를 위한 후처리
        # 틱택토 게임의 특성상 4가지 색상만 존재하므로 양자화
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
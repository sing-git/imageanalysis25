import numpy as np


def compute_rmse(img_pred, img_true):
    # Convert to same dtype
    img_true = img_true.astype(np.float32)
    img_pred = img_pred.astype(np.float32)
    return np.sqrt(np.mean((img_true - img_pred)**2))



def weight_scores(rmse, img_code_size, codebook_size):
    """ Define wighting of different terms in score """
    w_img_code_size = 1.e-1 * img_code_size
    w_codebook_size = 1.e-6 * codebook_size

    w_score = rmse + w_img_code_size + w_codebook_size
    return w_score, rmse, w_img_code_size, w_codebook_size


def compute_evaluation_score(img_code, img_true, reconstructor):
    """ """
    
    # Some format checks
    assert isinstance(img_code, dict), "Expect dict inputs"
    assert isinstance(img_true, dict), "Expect dict inputs"

    assert all(k in img_true for k in img_code), \
        "Ground truth dict is missing keys"

    assert all(k in img_code for k in img_true), \
        "Compressed image dict is missing keys"     

    total_rmse = 0.0
    total_size = 0.0
    for k in img_code:
        img_rec = reconstructor.reconstruct(img_code[k])
  
        total_rmse += compute_rmse(img_rec, img_true[k])
        total_size += img_code[k].nbytes
    
    codebook_size = reconstructor.codebook.nbytes

    mean_rmse = total_rmse / len(img_true)
    mean_size = total_size / len(img_true)

    return weight_scores(mean_rmse, mean_size, codebook_size)

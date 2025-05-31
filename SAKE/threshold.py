from scipy.spatial import distance
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def cosine_similarity(u,v):
    # normalize u,v
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return np.dot(u,v)

def hid_threshold(last_h, mean_embs, t, iv = None, dist_type = "euc"):
    if dist_type == "cos":
        max_similarity = -1
        max_index = -1
        for i, v in enumerate(mean_embs):
            similarity = cosine_similarity(np.array(last_h), np.array(v))
            if similarity > max_similarity:
                max_similarity = similarity
                max_index = i
        if max_similarity >= t:
            return max_similarity, max_index
        else:
            return [None, None]
    
    if dist_type == "euc":
        min_distance = 100000
        min_index = -1
        for i, v in enumerate(mean_embs):
            d = distance.euclidean(np.array(last_h), np.array(v))
            if d < min_distance:
                min_distance = d
                min_index = i
        if min_distance <= t:
            return min_distance, min_index
        else:
            return [None, None]
        
    # mahalanobis distance
    if dist_type == "mal":
        min_distance = 100000
        min_index = -1
        for i, v in enumerate(mean_embs):
            d = distance.mahalanobis(np.array(last_h), np.array(v), iv)
            if d < min_distance:
                min_distance = d
                min_index = i
        if min_distance <= t:
            return min_distance, min_index
        else:
            return [None, None]

def prompt_threshold(prompt_enc, prompt_embs, t, iv = None, dist_type = "euc"):
    if dist_type == "cos":
        max_similarity = -1
        max_index = -1
        for i, v in enumerate(prompt_embs):
            similarity = cosine_similarity(np.array(prompt_enc), np.array(v))
            if similarity > max_similarity:
                max_similarity = similarity
                max_index = i
        if max_similarity >= t:
            return max_similarity, max_index
        else:
            return [None, None]
    
    if dist_type == "euc":
        min_distance = 100000
        min_index = -1
        for i, v in enumerate(prompt_embs):
            d = distance.euclidean(np.array(prompt_enc), np.array(v))
            if d < min_distance:
                min_distance = d
                min_index = i
        if min_distance <= t:
            return min_distance, min_index
        else:
            return [None, None]
        
    # mahalanobis distance
    if dist_type == "mal":
        min_distance = 100000
        min_index = -1
        for i, v in enumerate(prompt_embs):
            d = distance.mahalanobis(np.array(prompt_enc), np.array(v), iv)
            if d < min_distance:
                min_distance = d
                min_index = i
        if min_distance <= t:
            return min_distance, min_index
        else:
            return [None, None]

def softmax(x):
    """
    Compute the softmax of a vector.
    
    Parameters:
    x (numpy.ndarray): Input vector.
    
    Returns:
    numpy.ndarray: Softmax-transformed vector.
    """
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def decompose_function(h, n):
    """
    Decompose function to sample a decomposition of the input.
    
    Parameters:
    h (numpy.ndarray): A vector in R^d.
    n (int): The dimension of the randomly generated vector a.
    
    Returns:
    numpy.ndarray: A matrix G in R^(d x n) such that G @ a = h.
    """
    h = np.array(h.cpu())
    d = len(h)
    
    # Generate a random vector a and apply softmax to ensure ||a|| = 1
    a = np.random.uniform(0, 1, size=n)
    a = softmax(a)
    
    G = np.zeros((d, n))  # Initialize G as a zero matrix of size d x n
    a_plus = a.reshape(1, -1) / np.dot(a, a)
   
    for i in range(d):
        eta = np.random.uniform(0, 1, size=n)  # Sample eta from U([0, 1]^n)
        # Update G[i, :] using the corrected formula
        G[i, :] = (a_plus + eta @ (np.eye(n) - np.outer(a, a_plus))) * h[i]
    return G, a

def transform_decomposition(G, a, cf, idx):
    for i in range(len(G[0])):
        # only map rows closer to mean source embeddings
        if distance.euclidean(G[:,i], cf[-3][idx]) < distance.euclidean(G[:,i], cf[-2][idx]):
            G[:,i] = cf[-1][idx].transform(Xs = G[:,i])

    return np.dot(G, a)

def compute_means(cf, indexes=(0,10)):
    source_mean_embs = []
    target_mean_embs = []
    for i in range(indexes[0], indexes[1]):
        source_mean_embs.append(np.mean(np.concatenate([cf[i]['source_embs'], cf[i]['forced_source_embs']], axis = 0), axis = 0).tolist())
        target_mean_embs.append(np.mean(cf[i]['target_embs'], axis = 0).tolist())
    
    return source_mean_embs, target_mean_embs

def compute_prompt_means_counterfact(cf, emb_model, indexes=(0,10)):
    prompt_embs = []
    for i in tqdm(range(indexes[0], indexes[1])):
        prompt = cf[i]['requested_rewrite']['prompt'].replace('{}', cf[i]['requested_rewrite']['subject'])
        prompt_list = cf[i]['source_list']
        prompt_list.append(prompt)
        prompt_list_embs = []
        for j in range(len(prompt_list)):
            prompt_list_embs.append(emb_model.encode(prompt_list[j]))
        prompt_embs.append(np.mean(prompt_list_embs, axis = 0).tolist())
    
    return prompt_embs

def compute_prompt_means_popular(pop, emb_model, indexes=(0,10)):
    prompt_embs = []
    for i in tqdm(range(indexes[0], indexes[1])):
        prompt_list = pop[i]['source_list']
        prompt_list_embs = []
        for j in range(len(prompt_list)):
            prompt_list_embs.append(emb_model.encode(prompt_list[j]))
        prompt_embs.append(np.mean(prompt_list_embs, axis = 0).tolist())
    
    return prompt_embs
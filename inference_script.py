"""
Inference Script for HPC Music Recommendation System
演示如何加载训练好的模型并为特定用户生成 top-K 推荐。
"""
import torch
import numpy as np
import pickle
from pathlib import Path

# 导入项目中定义的模型结构
from layer3.model import TransformerRecommender
from utils.hdf5_io import read_all_vectors

def load_inference_context(features_h5, als_h5, history_pickle):
    """加载推理所需的元数据和索引映射"""
    # 1. 加载 ID 映射
    (user_ids, user_vecs), (song_ids, song_vecs) = read_all_vectors(als_h5)
    
    # 2. 加载内容特征 (Layer 1 输出)
    import h5py
    with h5py.File(features_h5, "r") as f:
        c_vecs = f["features"][:]
        c_ids = f["song_ids"][:].astype(str)

    # 3. 加载用户历史记录 (用于构建输入序列)
    with open(history_pickle, "rb") as f:
        user_histories = pickle.load(f)

    # 构建索引查找表
    song_to_idx = {s: i for i, s in enumerate(song_ids)}
    
    return {
        "user_ids": user_ids,
        "song_ids": song_ids,
        "user_vecs": user_vecs,
        "song_vecs": song_vecs, # 这里实际包含 ALS 向量
        "content_vecs": c_vecs,
        "user_histories": user_histories,
        "song_to_idx": song_to_idx,
        "n_songs": len(song_ids)
    }

def recommend(user_id, context, model, device, top_k=10):
    """为指定用户生成推荐"""
    model.eval()
    
    # 获取用户历史
    if user_id not in context["user_histories"]:
        print(f"User {user_id} not found.")
        return []
    
    history = context["user_histories"][user_id]
    history_len = len(history)
    
    # 构建模型输入 (添加 Batch 维度)
    # 注意：这里的 token 索引通常是 song_idx + 1 (因为 0 是 PAD_ID)
    seq = torch.tensor([history], dtype=torch.long, device=device)
    h_len_t = torch.tensor([history_len], dtype=torch.long, device=device)
    
    # 获取最后听过的一首歌的特征用于 Cold-start Fusion 逻辑
    last_song_idx = history[-1]
    als_v = torch.tensor(context["song_vecs"][last_song_idx], dtype=torch.float32, device=device).unsqueeze(0)
    content_v = torch.tensor(context["content_vecs"][last_song_idx], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        # 模型前向传播获取 Logits
        logits = model(seq, als_v, content_v, h_len_t)
        
        # 排除用户已经听过的歌 (可选)
        for s_idx in history:
            logits[0, s_idx] = -float('inf')
        
        # 获取 Top-K
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idxs = torch.topk(probs, top_k)
    
    # 映射回原始歌曲 ID
    recommendations = []
    for i in range(top_k):
        idx = top_idxs[0, i].item()
        recommendations.append({
            "song_id": context["song_ids"][idx],
            "score": top_probs[0, i].item()
        })
    
    return recommendations

def main():
    # 配置路径 (根据实际部署环境修改)
    CHECKPOINT = "data/checkpoints/ckpt_epoch10.pt"
    FEATURES_H5 = "data/features.h5"
    ALS_H5 = "data/als_vectors.h5"
    HISTORIES = "data/user_histories.pkl"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化上下文和数据
    ctx = load_inference_context(FEATURES_H5, ALS_H5, HISTORIES)

    # 2. 初始化并加载模型
    model = TransformerRecommender(
        n_songs=ctx["n_songs"],
        als_dim=ctx["user_vecs"].shape[1],
        content_dim=ctx["content_vecs"].shape[1]
    ).to(device)
    
    # 加载权重 (处理 DDP 保存时的 module. 前缀)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")

    # 3. 执行推理演示
    test_user_idx = list(ctx["user_histories"].keys())[0]
    print(f"\nGenerating recommendations for User Index: {test_user_idx}")
    
    recs = recommend(test_user_idx, ctx, model, device, top_k=5)
    
    print("\nTop 5 Recommendations:")
    for i, r in enumerate(recs):
        print(f"{i+1}. SongID: {r['song_id']} (Confidence: {r['score']:.4f})")

if __name__ == "__main__":
    main()

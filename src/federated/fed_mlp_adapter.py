#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简适配器：把 curriculum 的 actor_critic 模型从 ckpt 里还原出来，并进行离线评测。
"""

from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch

from src.federated.consistency_test_fixed import CurriculumTrainer, build_agent, eval_one_stage


def _load_agent_from_ckpt(port: str, ckpt_path: Path, device) -> Tuple[object, Optional[list]]:
    """加载 ckpt，推断 hidden_dim/state_dim，构造 agent 并严格加载权重"""
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = ckpt.get("model_state_dict") or ckpt  # 兜底

    # 推断 hidden_dim 与 state_dim
    if "actor.0.weight" in sd:
        hidden_dim = int(sd["actor.0.weight"].shape[0])
    else:
        hidden_dim = 256  # 兜底

    if "feature_fusion.0.weight" in sd:
        feature_input_dim = int(sd["feature_fusion.0.weight"].shape[1])
        state_dim = int(feature_input_dim - hidden_dim // 4)
    else:
        state_dim = 56  # 兜底

    agent = build_agent(
        port,
        hidden_dim=hidden_dim, learning_rate=3e-4, batch_size=32,
        device=device, num_heads=4, dropout=0.1,
        state_dim=state_dim, action_dim=15, node_feature_dim=8,
        entropy_coef=0.02, ppo_epochs=6,
    )
    agent.actor_critic.to(device).eval()

    # 严格加载
    agent.actor_critic.load_state_dict(sd, strict=True)

    return agent, ckpt.get("test_data")


def eval_port_with_ckpt(port: str, ckpt: Path, *,
                        n_samples: int = 200, seed: int = 42, k_baseline: int = 50,
                        no_cache: bool = True) -> Dict:
    """
    用 curriculum 的 ckpt 对该 port 的所有阶段做一次评测，聚合端口级指标。
    返回: {"success_rate": float, "avg_reward": float|None, "num_samples": int}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = CurriculumTrainer(port)

    agent, saved = _load_agent_from_ckpt(port, ckpt, device)

    perfs = []
    for stage in trainer.curriculum_stages:
        perf, std, stable, n_used = eval_one_stage(
            trainer, agent, stage,
            n_samples=n_samples,
            fixed_test_data=saved,
            k_baseline=k_baseline,
            device=device,
            seed=seed,
            no_cache=no_cache,
        )
        perfs.append(perf)

    # 聚合
    wr = float(np.mean([p["win_rate"] for p in perfs])) if perfs else float("nan")
    avg = float(np.mean([p["avg_reward"] for p in perfs])) if perfs else float("nan")
    nsum = int(sum([p.get("n_samples", 0) for p in perfs]))

    return {"success_rate": wr, "avg_reward": avg, "num_samples": nsum}
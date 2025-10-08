#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flower 联邦学习客户端（容错版）
- 训练阶段：任何指标缺失(None)都不会报错
- 评估阶段：沿用 FedAvgClient.evaluate()，但也做容错
- 启动方式：start_client(..., client=...to_client())
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import flwr as fl

# 把项目根加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.federated.fedavg_client import FedAvgClient  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_float(x, default=None):
    """把 x 安全转为 float。x 为 None 或无法转换时，返回 default。"""
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


class FlowerClient(fl.client.NumPyClient):
    """Flower NumPyClient 封装"""

    def __init__(
        self,
        port: str,
        server_address: str = "localhost:8080",
        init_weights: Optional[str] = None,
        episodes: int = 8,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
    ):
        self.port = port
        self.server_address = server_address
        self.init_weights = init_weights
        self.episodes = episodes
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

        self.client = FedAvgClient(port=port, init_weights=init_weights)

        # 固定参数键顺序
        self.param_keys = list(self.client.get_parameters().keys())
        logger.info(f"🔑 参数键顺序: {self.param_keys}")
        logger.info(
            f"⚙️ 训练参数: episodes={episodes}, ppo_epochs={ppo_epochs}, "
            f"batch_size={batch_size}, entropy_coef={entropy_coef}"
        )

    # ---------- Flower NumPyClient 接口 ----------

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        params = self.client.get_parameters()
        return [params[k] for k in self.param_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = {k: v for k, v in zip(self.param_keys, parameters)}
        self.client.set_parameters(params_dict)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """本地训练（容错收集指标，不对 None 做 float()）"""
        # 下发全局参数
        self.set_parameters(parameters)

        logger.info(f"🏋️ 开始本地训练 - 港口: {self.port}")
        logger.info(
            f"📊 训练配置: episodes={self.episodes}, ppo_epochs={self.ppo_epochs}, "
            f"batch_size={self.batch_size}"
        )

        # 运行本地训练，可能返回 None 或缺字段
        train_stats = self.client.train(
            episodes=self.episodes,
            ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size,
            entropy_coef=self.entropy_coef,
        ) or {}

        # 训练后参数
        new_params = self.get_parameters(config)

        # ---- 关键：容错整理指标 ----
        # 注意：dict.get("x", 0.0) 对“键存在但值为 None”的情况不会用到默认值
        # 所以必须先取出原值，再 _safe_float(x, default)。
        loss = _safe_float(train_stats.get("loss"), None)
        reward = _safe_float(
            train_stats.get("avg_reward", train_stats.get("reward")), None
        )
        success_rate = _safe_float(
            train_stats.get("success_rate", train_stats.get("accuracy")), None
        )
        num_samples = int(
            train_stats.get(
                "num_samples",
                max(1, int(self.batch_size))
                * max(1, int(self.episodes))
                * max(1, int(self.ppo_epochs)),
            )
        )

        metrics: Dict[str, float] = {
            "port": self.port,
            "episodes": self.episodes,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "num_samples": num_samples,
        }
        if loss is not None:
            metrics["loss"] = loss
        if reward is not None:
            metrics["reward"] = reward
        if success_rate is not None:
            metrics["success_rate"] = success_rate

        return new_params, num_samples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """本地评估（保持 Flower 协议：第一个返回值需要是 float）"""
        self.set_parameters(parameters)

        logger.info(f"📊 开始本地评估 - 港口: {self.port}")
        eval_metrics = self.client.evaluate() or {}

        # Flower evaluate 必须返回 (loss: float, num_examples: int, metrics: Dict)
        loss = _safe_float(eval_metrics.get("loss"), 0.0)  # 没有就给 0.0
        reward = _safe_float(
            eval_metrics.get("avg_reward", eval_metrics.get("reward")), None
        )
        success_rate = _safe_float(
            eval_metrics.get("success_rate", eval_metrics.get("accuracy")), None
        )
        num_samples = int(eval_metrics.get("num_samples", 800))

        metrics = {"port": self.port, "num_samples": num_samples}
        if reward is not None:
            metrics["avg_reward"] = reward
        if success_rate is not None:
            metrics["success_rate"] = success_rate

        return float(loss), num_samples, metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Flower 联邦学习客户端（容错版）")
    parser.add_argument("--port", type=str, required=True, help="港口名称")
    parser.add_argument("--server", type=str, default="localhost:8080", help="服务器地址")
    parser.add_argument("--init", type=str, help="初始权重文件路径")

    # 训练参数
    parser.add_argument("--episodes", type=int, default=8, help="训练 episodes 数")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵系数")

    args = parser.parse_args()

    logger.info(f"🚀 启动Flower客户端 - 港口: {args.port}")
    logger.info(f"🌐 服务器地址: {args.server}")
    logger.info(
        f"⚙️ 训练参数: episodes={args.episodes}, ppo_epochs={args.ppo_epochs}, "
        f"batch_size={args.batch_size}, entropy_coef={args.entropy_coef}"
    )

    client = FlowerClient(
        port=args.port,
        server_address=args.server,
        init_weights=args.init,
        episodes=args.episodes,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
    )

    # 官方推荐启动方式（不再用 start_numpy_client）
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()

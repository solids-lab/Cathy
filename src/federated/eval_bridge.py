# src/federated/eval_bridge.py
from __future__ import annotations
import json, glob, os
from pathlib import Path
from typing import Optional, Dict, Any

def _latest_json_for(port: str) -> Optional[Path]:
    """
    优先：models/releases/<yyyy-mm-dd>/consistency_<port>_*.json
    回溯：models/releases/*/consistency_<port>_*.json
    兜底：reports/FLW_*/nightly/forced_<port>_*.json
    """
    rel_root = Path("models/releases")

    # 1) 今天目录
    today_str = os.popen("date +%F").read().strip()
    if today_str:
        today = rel_root / today_str
        if today.exists():
            cand = sorted(today.glob(f"consistency_{port}_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if cand:
                return cand[0]

    # 2) 所有 releases 回溯
    cand2 = sorted(glob.glob(f"models/releases/*/consistency_{port}_*.json"),
                   key=lambda f: Path(f).stat().st_mtime, reverse=True)
    if cand2:
        return Path(cand2[0])

    # 3) 兜底：forced nightly（通过 scripts/force_eval_to_json.py 生成）
    cand3 = sorted(glob.glob(f"reports/FLW_*/nightly/forced_{port}_*.json"),
                   key=lambda f: Path(f).stat().st_mtime, reverse=True)
    if cand3:
        return Path(cand3[0])

    return None

def _read_metrics_from_json(p: Path) -> Optional[Dict[str, Any]]:
    """从一致性/forced JSON 提取 success_rate / avg_reward / num_samples"""
    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None

    # nightly/forced 直接就是 flat 字段
    sr = obj.get("success_rate")
    rew = obj.get("avg_reward")
    n = obj.get("num_samples") or obj.get("samples") or obj.get("n_samples")

    if sr is None:
        # 一致性 JSON: 在 stages 内部给 win_rate/threshold/pass；需要一个汇总策略
        stages = obj.get("stages") or []
        if stages:
            # 简单平均 win_rate（如果需要，可按样本数加权）
            wrs = [s.get("win_rate") for s in stages if s.get("win_rate") is not None]
            sr = float(sum(wrs)/len(wrs)) if wrs else None
            # 平均 n
            ns = [s.get("n_samples") for s in stages if s.get("n_samples") is not None]
            n = int(sum(ns)) if ns else n
            # avg_reward 一致性 JSON 未必有，留空
            rew = rew

    # success/total 兜底
    if sr is None and obj.get("total"):
        try:
            sr = float(obj.get("success", 0)) / float(obj["total"])
        except Exception:
            pass

    if sr is None and rew is None and n is None:
        return None

    return {
        "success_rate": sr,
        "avg_reward": rew,
        "num_samples": n,
        "source": str(p),
    }

def read_latest_eval(port: str) -> Optional[Dict[str, Any]]:
    """外部调用：读取该 port 最新评测 JSON 的核心指标"""
    p = _latest_json_for(port)
    if not p:
        return None
    return _read_metrics_from_json(p)

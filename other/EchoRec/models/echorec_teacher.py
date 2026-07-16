"""Load and freeze a pretrained sequential teacher."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from SeqRec.bert4rec.model import BERT4Rec
from SeqRec.gru4rec.model import GRU4Rec
from SeqRec.sasrec.model import SASRec
from SeqRec.sasrec.echorec_sa_backbone import EchoRecSABackbone


def build_recsys_model(usernum: int, itemnum: int, args):
    backbone = getattr(args, "recsys_backbone", "sasrec")
    if backbone in {"echorec_sa", "sa_transformer"}:
        model_cls = EchoRecSABackbone
    elif backbone == "bert4rec":
        model_cls = BERT4Rec
    elif backbone == "gru4rec":
        model_cls = GRU4Rec
    else:
        model_cls = SASRec
    return model_cls(usernum, itemnum, args)


def load_teacher_checkpoint(recsys: str, dataset: str, current_args=None):
    """Load teacher checkpoint kwargs and state dict."""
    explicit = getattr(current_args, "recsys_ckpt_path", None) if current_args is not None else None
    if explicit:
        chosen_path = Path(explicit).resolve()
        if not chosen_path.is_file():
            raise FileNotFoundError(f"Explicit recsys checkpoint not found: {chosen_path}")
    else:
        base_dir = Path(f"./SeqRec/{recsys}/{dataset}")
        candidates = [
            base_dir / "model_best.pth",
            base_dir / "model.pth",
            base_dir / "sa_pretrain" / "model_best.pth",
            base_dir / "sa_pretrain" / "model.pth",
        ]
        chosen_path = next((path for path in candidates if path.is_file()), None)
        if chosen_path is None:
            fallback = sorted(base_dir.glob("*.pth")) + sorted(base_dir.glob("*/*.pth"))
            if len(fallback) != 1:
                searched = "\n".join(f"  - {p}" for p in candidates)
                raise FileNotFoundError(
                    "Could not find model_best.pth or model.pth.\n"
                    f"Searched:\n{searched}\n"
                    f"Fallback .pth count: {len(fallback)}"
                )
            chosen_path = fallback[0]

    ckpt_obj = torch.load(str(chosen_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt_obj, dict) and "kwargs" in ckpt_obj and "state_dict" in ckpt_obj:
        kwargs = ckpt_obj["kwargs"]
        checkpoint = ckpt_obj["state_dict"]
        logging.info("load checkpoint from %s (dict format)", chosen_path)
    elif isinstance(ckpt_obj, (list, tuple)) and len(ckpt_obj) >= 2:
        kwargs, checkpoint = ckpt_obj[0], ckpt_obj[1]
        logging.info("load checkpoint from %s (tuple format)", chosen_path)
    else:
        raise ValueError(f"Unsupported checkpoint format from {chosen_path}: {type(ckpt_obj)}")
    return kwargs, checkpoint


class EchoRecTeacher(nn.Module):
    """Frozen recommender teacher used by the SI stage."""

    def __init__(self, recsys_model, pre_trained_data, device, current_args=None):
        super().__init__()
        kwargs, checkpoint = load_teacher_checkpoint(recsys_model, pre_trained_data, current_args=current_args)
        kwargs["args"].device = device

        model = build_recsys_model(kwargs["user_num"], kwargs["item_num"], kwargs["args"])
        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = False

        self.item_num = model.item_num
        self.user_num = model.user_num
        self.hidden_units = kwargs["args"].hidden_units
        self.model = model.to(device)


TeacherBackbone = EchoRecTeacher

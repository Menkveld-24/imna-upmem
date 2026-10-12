import torch
from imna.utils import fuseSignsAndShifts


class ModelTransformer:
    """Transforms a model's state_dict weight format for inference on the DPU."""

    def __init__(self, state_dict: dict):
        self.state_dict = dict(state_dict)

    def to_fused_ss(self) -> "ModelTransformer":
        """Fuse sign+shift pairs into a single fused_ss int8 buffer per layer.

        Works on either fp parameters (sign_fp/shift_fp) or already-quantized
        buffers (sign/shift). Returns self for chaining.
        """
        prefixes = set()
        for key in self.state_dict:
            for suffix in (".sign_fp", ".shift_fp", ".sign", ".shift"):
                if key.endswith(suffix):
                    prefixes.add(key[: -len(suffix)])

        consumed = set()
        new_sd = {}
        for key, tensor in self.state_dict.items():
            if key in consumed:
                continue

            prefix = None
            for suffix in (".sign_fp", ".shift_fp", ".sign", ".shift"):
                if key.endswith(suffix):
                    prefix = key[: -len(suffix)]
                    break

            if prefix is None or prefix not in prefixes:
                new_sd[key] = tensor
                continue

            sign_key  = prefix + ".sign_fp"  if (prefix + ".sign_fp")  in self.state_dict else prefix + ".sign"
            shift_key = prefix + ".shift_fp" if (prefix + ".shift_fp") in self.state_dict else prefix + ".shift"

            if sign_key not in self.state_dict or shift_key not in self.state_dict:
                new_sd[key] = tensor
                continue

            sign_t  = self.state_dict[sign_key]
            shift_t = self.state_dict[shift_key]

            if sign_key.endswith("_fp"):
                sign_t = torch.sign(torch.round(sign_t)).to(torch.int8)
            if shift_key.endswith("_fp"):
                shift_t = torch.round(shift_t).to(torch.int8)

            fused_key = prefix + ".fused_ss"
            new_sd[fused_key] = fuseSignsAndShifts(sign_t, shift_t)
            consumed.add(sign_key)
            consumed.add(shift_key)

        self.state_dict = new_sd
        return self

    def save(self, path: str) -> None:
        torch.save(self.state_dict, path)

    def get(self) -> dict:
        return self.state_dict

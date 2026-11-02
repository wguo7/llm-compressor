import importlib

import pytest
import torch

from llmcompressor.modeling.offset_norm import (
    CalibrationOffsetNorm,
    norm_calibration_context,
)

# (transformers model module, norm class) for norms computing `x * (1 + weight)`
OFFSET_NORMS = [
    ("gemma", "GemmaRMSNorm"),
    ("gemma2", "Gemma2RMSNorm"),
    ("gemma3", "Gemma3RMSNorm"),
    ("qwen3_next", "Qwen3NextRMSNorm"),
    ("vaultgemma", "VaultGemmaRMSNorm"),
    ("recurrent_gemma", "RecurrentGemmaRMSNorm"),
    ("t5gemma", "T5GemmaRMSNorm"),
    ("t5gemma2", "T5Gemma2RMSNorm"),
    ("step3p7", "Step3p7RMSNorm"),
    ("minimax_m3_vl", "MiniMaxM3VLRMSNorm"),
    ("muse_spark", "MuseSparkTextCenteredRMSNorm"),
]


def _load_norm_class(module_name: str, class_name: str):
    try:
        module = importlib.import_module(
            f"transformers.models.{module_name}.modeling_{module_name}"
        )
    except ImportError:
        pytest.skip(f"transformers does not provide {module_name}")
    if not hasattr(module, class_name):
        pytest.skip(f"transformers does not provide {class_name}")
    return getattr(module, class_name)


class _Wrapper(torch.nn.Module):
    config = None

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm


@pytest.mark.unit
@pytest.mark.parametrize("module_name,class_name", OFFSET_NORMS)
def test_offset_norm_smoothing_is_preserved(module_name, class_name):
    norm_cls = _load_norm_class(module_name, class_name)
    torch.manual_seed(0)
    norm = norm_cls(16, eps=1e-6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16) * 0.5)
    model = _Wrapper(norm)

    x = torch.randn(4, 16)
    scales = torch.rand(16) + 0.5
    expected = norm(x) / scales

    # smoothing modifiers divide the norm weight by their scales
    with norm_calibration_context(model):
        assert isinstance(model.norm, CalibrationOffsetNorm)
        with torch.no_grad():
            model.norm.weight.div_(scales)

    assert model.norm is norm
    torch.testing.assert_close(model.norm(x), expected)

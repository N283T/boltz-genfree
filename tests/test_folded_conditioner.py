import pytest
import torch

from boltz.model.modules.affinity import folded_zero_conditioning
from boltz.model.modules.encodersv2 import PairwiseConditioning


@pytest.mark.parametrize("scale", [0, 0.01, 1, 10])
def test_folded_preserves_full_width_affine_projection(scale):
    torch.manual_seed(91)
    m = PairwiseConditioning(128, 128).eval()
    before = set(m.state_dict())
    with torch.no_grad():
        m.dim_pairwise_init_proj[0].weight.uniform_(0.5, 1.5)
        m.dim_pairwise_init_proj[0].bias.uniform_(-0.2, 0.2)
        p = torch.randn(2, 9, 9, 128) * scale
        reference = m(p, torch.zeros_like(p))
        actual = folded_zero_conditioning(m, p)
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
    assert set(m.state_dict()) == before

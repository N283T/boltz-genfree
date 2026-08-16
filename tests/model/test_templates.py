import torch

from boltz.data.feature.featurizerv2 import load_dummy_templates_features
from boltz.model.modules.trunkv2 import TemplateV2Module


def test_template_v2_has_zero_contribution_without_template() -> None:
    num_tokens = 3
    features = {
        key: value.unsqueeze(0)
        for key, value in load_dummy_templates_features(
            tdim=1,
            num_tokens=num_tokens,
        ).items()
    }
    module = TemplateV2Module(
        token_z=8,
        template_dim=4,
        template_blocks=1,
        dropout=0.0,
        pairwise_head_width=2,
        pairwise_num_heads=2,
    )

    output = module(
        z=torch.randn(1, num_tokens, num_tokens, 8),
        feats=features,
        pair_mask=torch.ones(1, num_tokens, num_tokens),
    )

    assert torch.count_nonzero(features["template_mask"]) == 0
    assert torch.count_nonzero(output) == 0

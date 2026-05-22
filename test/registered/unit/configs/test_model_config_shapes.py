"""Unit tests for model shape derivation."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestModelConfigShapes(unittest.TestCase):
    def test_glm_nextn_uses_mha_value_head_dim(self):
        config = ModelConfig.__new__(ModelConfig)
        config.hf_config = SimpleNamespace(architectures=["Glm4MoeForCausalLMNextN"])
        config.hf_text_config = SimpleNamespace(
            head_dim=64,
            hidden_size=4096,
            num_attention_heads=64,
            v_head_dim=256,
        )

        config._derive_model_shapes()

        self.assertEqual(config.attention_arch, AttentionArch.MHA)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.v_head_dim, 64)

    def test_glm_nextn_head_dim_fallback_matches_mha_decoder(self):
        config = ModelConfig.__new__(ModelConfig)
        config.hf_config = SimpleNamespace(architectures=["Glm4MoeForCausalLMNextN"])
        config.hf_text_config = SimpleNamespace(
            head_dim=None,
            hidden_size=4096,
            num_attention_heads=64,
            v_head_dim=256,
        )

        config._derive_model_shapes()

        self.assertEqual(config.head_dim, 64)
        self.assertEqual(config.v_head_dim, 64)


if __name__ == "__main__":
    unittest.main()

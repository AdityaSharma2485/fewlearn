"""
Tests for the MINDS framework.
"""

import unittest
import torch

class TestMINDS(unittest.TestCase):
    """Test cases for the MINDS framework."""

    def test_minds_initialization(self):
        """Test that MINDS can be initialized properly."""
        try:
            from fewlearn.core.minds import MINDS
            minds = MINDS()
            self.assertIsNotNone(minds)
            self.assertTrue(hasattr(minds, 'device'))
            self.assertTrue(hasattr(minds, 'models'))
            self.assertIsInstance(minds.models, dict)
        except ImportError:
            self.skipTest("fewlearn not properly installed")
    
    def test_models_registration(self):
        """Test that models can be registered with MINDS."""
        try:
            from fewlearn.core.minds import MINDS
            from fewlearn.models.prototypical import PrototypicalNetworks
            
            minds = MINDS()
            # Test with a simple model
            minds.add_model("test_model", PrototypicalNetworks(backbone="resnet18"))
            self.assertIn("test_model", minds.models)
            self.assertIsNotNone(minds.models["test_model"])
        except ImportError:
            self.skipTest("fewlearn not properly installed")
            
if __name__ == '__main__':
    unittest.main() 
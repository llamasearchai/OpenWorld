import unittest
from openworld_backend.worldmodel_agent import WorldModelAgent
from openworld_backend.config import OpenWorldConfig

class TestBasicFunctionality(unittest.TestCase):
    def test_agent_initialization(self):
        config = OpenWorldConfig.from_dict({
            "physics": {"d_model": 512},
            "transformer": {"d_model": 512}
        })
        agent = WorldModelAgent(config)
        self.assertIsNotNone(agent)

if __name__ == "__main__":
    unittest.main() 
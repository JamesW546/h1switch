import unittest

import numpy as np

from example_plugins import HoldPositionAdapter
from policy_switcher import PolicyCommand, PolicySwitcher, RobotState, blend_commands, smoothstep


class PolicySwitcherTest(unittest.TestCase):
    def test_smoothstep_endpoints(self):
        self.assertEqual(smoothstep(-1.0), 0.0)
        self.assertEqual(smoothstep(0.0), 0.0)
        self.assertEqual(smoothstep(1.0), 1.0)
        self.assertEqual(smoothstep(2.0), 1.0)

    def test_command_blend(self):
        first = PolicyCommand(np.zeros(2), np.full(2, 100.0), np.full(2, 5.0))
        second = PolicyCommand(np.ones(2), np.full(2, 200.0), np.full(2, 15.0))
        mixed = blend_commands(first, second, 0.25)
        np.testing.assert_allclose(mixed.q, 0.25)
        np.testing.assert_allclose(mixed.kp, 125.0)
        np.testing.assert_allclose(mixed.kd, 7.5)

    def test_bidirectional_switch(self):
        first = HoldPositionAdapter("a", [0.0, 0.0])
        second = HoldPositionAdapter("b", [1.0, -1.0])
        switcher = PolicySwitcher(first, second, "a", 0.1, 0.02)
        state = RobotState(0.0, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
        switcher.reset(state)

        self.assertTrue(switcher.request_switch("b"))
        outputs = []
        for _ in range(7):
            command, _ = switcher.step(state)
            outputs.append(command.q.copy())
        self.assertEqual(switcher.active, "b")
        self.assertGreater(outputs[-1][0], outputs[0][0])

        self.assertTrue(switcher.request_switch("a"))
        for _ in range(7):
            switcher.step(state)
        self.assertEqual(switcher.active, "a")


if __name__ == "__main__":
    unittest.main()

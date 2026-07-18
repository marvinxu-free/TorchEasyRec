# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for loss.proto (L2Loss transform / huber_delta extensions)."""

import unittest

from google.protobuf import text_format

from tzrec.protos import loss_pb2


class L2LossProtoTest(unittest.TestCase):
    def test_l2_loss_defaults(self) -> None:
        cfg = loss_pb2.L2Loss()
        self.assertEqual(cfg.transform, loss_pb2.L2Loss.NONE)
        self.assertAlmostEqual(cfg.huber_delta, 0.0)

    def test_l2_loss_supports_signed_log_transform(self) -> None:
        cfg = loss_pb2.L2Loss()
        text_format.Parse("transform: SIGNED_LOG1P huber_delta: 1.0", cfg)
        self.assertEqual(cfg.transform, loss_pb2.L2Loss.SIGNED_LOG1P)
        self.assertAlmostEqual(cfg.huber_delta, 1.0)

    def test_loss_config_nests_transformed_l2(self) -> None:
        cfg = loss_pb2.LossConfig()
        text_format.Parse(
            "l2_loss { transform: SIGNED_LOG1P huber_delta: 1.0 }", cfg
        )
        self.assertEqual(cfg.WhichOneof("loss"), "l2_loss")
        self.assertEqual(cfg.l2_loss.transform, loss_pb2.L2Loss.SIGNED_LOG1P)
        self.assertAlmostEqual(cfg.l2_loss.huber_delta, 1.0)


if __name__ == "__main__":
    unittest.main()

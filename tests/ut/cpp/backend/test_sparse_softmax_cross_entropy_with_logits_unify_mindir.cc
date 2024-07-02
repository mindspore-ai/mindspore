/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <memory>
#include "common/graph_optimizer_test_framework.h"
#include "ops/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/device/ascend/optimizer/mindir/sparse_softmax_cross_entropy_with_logits_unify_mindir.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
class SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR : public UT::Common {
 public:
  SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR() {}
};

/// Feature: A backend pass: SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR
/// Description: Convert SparseSoftmaxCrossEntropyWithLogits(is_grad=false) to
///              OneHot+SoftmaxCrossEntropyWithLogits+ReduceMean
/// Expectation: After optimize, match OneHot+SoftmaxCrossEntropyWithLogits+ReduceMean.
TEST_F(SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR,
       DISABLED_test_sparse_softmax_cross_entropy_with_logits_is_grad_is_false) {
  test::ConstructGraph c;
  auto logits = c.NewTensorInput("logits", kFloat, {2, 3});
  auto labels = c.NewTensorInput("labels", kInt32, {2});
  auto node = c.NewCNode("SparseSoftmaxCrossEntropyWithLogits", {logits, labels}, {{"is_grad", MakeValue(false)}});
  c.SetOutput(node);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("x3")
    .AddSeqVar("any")
    .AddCNode("one_hot", {std::make_shared<Primitive>("OneHot"), "any"})
    .AddCNode("softmax_cross_entropy_with_logits",
              {std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits"), "x1", "one_hot"})
    .AddCNode("tuple_get_item",
              {std::make_shared<Primitive>(kTupleGetItemOpName), "softmax_cross_entropy_with_logits", "x2"})
    .AddCNode("reduce_mean", {std::make_shared<Primitive>("ReduceMean"), "tuple_get_item", "x3"});
  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR
/// Description: When SparseSoftmaxCrossEntropyWithLogits(is_grad=true), this pass do not change this node.
/// Expectation: After optimize, not match SoftmaxCrossEntropyWithLogits
TEST_F(SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR, test_sparse_softmax_cross_entropy_with_logits_is_grad_is_true) {
  test::ConstructGraph c;
  auto logits = c.NewTensorInput("logits", kFloat32, {2, 3});
  auto labels = c.NewTensorInput("labels", kInt32, {2});
  auto node = c.NewCNode("SparseSoftmaxCrossEntropyWithLogits", {logits, labels}, {{"is_grad", MakeValue(true)}});
  c.SetOutput(node);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsUnifyMindIR>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("x2")
    .AddVar("x3")
    .AddSeqVar("any")
    .AddCNode("one_hot", {std::make_shared<Primitive>("OneHot"), "any"})
    .AddCNode("softmax_cross_entropy_with_logits",
              {std::make_shared<Primitive>("SoftmaxCrossEntropyWithLogits"), "x1", "one_hot"})
    .AddCNode("tuple_get_item",
              {std::make_shared<Primitive>(prim::kPrimTupleGetItem->name()), "softmax_cross_entropy_with_logits", "x2"})
    .AddCNode("reduce_mean", {std::make_shared<Primitive>("ReduceMean"), "tuple_get_item", "x3"});
  EXPECT_FALSE(checker.build_pattern_map(c.GetGraph()->output()));
}
}  // namespace mindspore

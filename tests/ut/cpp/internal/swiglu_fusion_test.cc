/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "common/graph_optimizer_test_framework.h"
#include "primitive/sequence_ops.h"
#include "common/common_test.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_swiglu_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "utils/phase.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

namespace mindspore {
namespace opt {
class InferenceSwiGLUFusionUT : public UT::Common {
 public:
  InferenceSwiGLUFusionUT() {}
};

/// Feature: A backend pass: InferenceSwiGLUFusion
/// Description: Convert SplitWithSize + Mul + Silu to Swiglu
/// Expectation: After optimize, match Swiglu.
TEST_F(InferenceSwiGLUFusionUT, InferenceSwiGLUFusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1, 1024, 11264});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{5632, 5632}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {input, split_size, dim}, {});

  auto index_0 = c.NewValueNode(MakeValue((int64_t)0));
  auto tuple_getitem_0 = c.NewCNode("TupleGetItem", {split_with_size, index_0}, {});
  auto index_1 = c.NewValueNode(MakeValue((int64_t)1));
  auto tuple_getitem_1 = c.NewCNode("TupleGetItem", {split_with_size, index_1}, {});
  auto activation = c.NewCNode("SiLU", {tuple_getitem_0}, {});
  auto mul = c.NewCNode("Mul", {tuple_getitem_1, activation}, {});
  c.SetOutput(mul);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceSwiGLUFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("dim").AddCNode(
    "Swiglu", {std::make_shared<Primitive>("Swiglu"), "input", "dim"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore

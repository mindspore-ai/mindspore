/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/swiglu_dynamic_quant_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "utils/phase.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
class InferenceSwiGLUDynamicQuantFusionUT : public UT::Common {
 public:
  InferenceSwiGLUDynamicQuantFusionUT() {}
};

/// Feature: A backend pass: SwiGLUDynamicQuantFusion
/// Description: Convert Swiglu + DynamicQuantExt to SwiGLUDynamicQuant
/// Expectation: After optimize, match SwiGLUDynamicQuant.
TEST_F(InferenceSwiGLUDynamicQuantFusionUT, InferenceSwiGLUDynamicQuantFusionTest) {
  MsContext::GetInstance()->SetJitInferBoost("on");
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1024, 11264});
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto swiglu = c.NewCNode("Swiglu", {input, dim}, {});
  auto swiglu_prim = common::AnfAlgo::GetCNodePrimitive(swiglu);
  MS_EXCEPTION_IF_NULL(swiglu_prim);
  swiglu_prim->AddAttr("FusionType", MakeValue("swiglu_v1"));

  auto none_ = c.NewValueNode(kNone);
  auto dynamic_quant = c.NewCNode("DynamicQuantExt", {swiglu, none_}, {});
  c.SetOutput(dynamic_quant);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::SwiGLUDynamicQuantFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("none_").AddCNode(
    "SwiGLUDynamicQuant", {std::make_shared<Primitive>("SwiGLUDynamicQuant"), "input", "none_"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore

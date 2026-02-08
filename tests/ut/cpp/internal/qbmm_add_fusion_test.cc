/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_add_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "utils/phase.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace opt {
class QbmmAddFusionUT : public UT::Common {
 public:
  QbmmAddFusionUT() {}
};

/// Feature: A backend pass: QbmmAddFusion
/// Description: Convert QuantBatchMatmul + Add to QuantBatchMatmul
/// Expectation: After optimize, match QuantBatchMatmul.
TEST_F(QbmmAddFusionUT, QbmmAddFusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kInt8, {1024, 1024});
  auto input_1 = c.NewTensorInput("input_1", kInt8, {1024, 1024});
  auto input_2 = c.NewTensorInput("input_2", kInt64, {1024}); // scale: should have data
  auto input_3 = c.NewTensorInput("input_3", kFloat16, {1024}); // bias: should have data
  ShapeVector sv = {1024};
  ParamInfoPtr param_info = std::make_shared<ParamInfo>(); // fake
  tensor::TensorPtr tensor2 = tensor::from_spec(kNumberTypeInt64, sv, device::DeviceType::kCPU);
  tensor::TensorPtr tensor3 = tensor::from_spec(kNumberTypeFloat16, sv, device::DeviceType::kCPU);
  tensor2->set_param_info(param_info);
  tensor3->set_param_info(param_info);
  input_2->set_default_param(tensor2);
  input_3->set_default_param(tensor3);
  auto load2 = c.NewCNode("Load", {input_2, input_2}, {})->cast<AnfNodePtr>();
  auto load3 = c.NewCNode("Load", {input_3, input_3}, {})->cast<AnfNodePtr>();
  auto ta = c.NewValueNode(MakeValue<bool>(false));
  auto tb = c.NewValueNode(MakeValue<bool>(false));
  auto dtype = c.NewValueNode(MakeValue<int64_t>(42));
  auto none_ = c.NewValueNode(kNone);
  auto qbmm = c.NewCNode("QuantBatchMatmul", {input_0, input_1, load2, none_, none_, none_, ta, tb, dtype}, {});
  auto add = c.NewCNode("Add", {qbmm, load3}, {});

  c.SetOutput(add);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::QbmmAddFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0").AddVar("input_1").AddVar("input_2").AddVar("input_3").AddVar("input_4").AddVar("input_5").AddVar("input_6").AddVar("input_7").AddVar("input_8").AddCNode(
    "qbmm_add", {std::make_shared<Primitive>("QuantBatchMatmul"), "input_0", "input_1", "input_2", "input_3", "input_4", "input_5", "input_6", "input_7", "input_8"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore

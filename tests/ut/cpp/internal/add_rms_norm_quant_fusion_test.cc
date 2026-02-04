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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/add_rms_norm_quant_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "utils/phase.h"
#include "ir/tensor.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"

#include "ir/tensor_new.h"
namespace mindspore {

using tensor::Tensor;

namespace opt {
class AddRmsNormQuantFusionUT : public UT::Common {
 public:
  AddRmsNormQuantFusionUT() {}
};

/// Feature: A backend pass: AddRmsNormQuantFusion
/// Description: Convert QuantV2(RmsNorm(Add)) to AddRmsNormQuantV2
/// Expectation: After optimize, match AddLayernorm.
TEST_F(AddRmsNormQuantFusionUT, AddRmsNormQuantFusionTest) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input_0 = c.NewTensorInput("input_0", kFloat16, {1, 1024, 11264});
  auto input_1 = c.NewTensorInput("input_1", kFloat16, {1, 1024, 11264});
  auto add = c.NewCNode("Add", {input_0, input_1}, {});

  auto gamma = c.NewTensorInput("gamma", kFloat16, {11264});
  auto eps = c.NewValueNode(MakeValue<float>(1e-5));
  auto rmsnorm = c.NewCNode("RmsNorm", {add, gamma, eps}, {});

  auto idx = c.NewValueNode(MakeValue<int64_t>(0));
  auto tuple_get_item_0 = c.NewCNode("TupleGetItem", {rmsnorm, idx}, {});

  auto monod = c.NewValueNode(kUMonad);
  ParamInfoPtr param_info = std::make_shared<ParamInfo>();

  auto scale_value = tensor::from_spec(kNumberTypeFloat16, ShapeVector({1}), device::DeviceType::kCPU);
  scale_value->set_param_info(param_info);
  auto scale_para = c.NewTensorInput("scale_para", kFloat16, {1});
  scale_para->set_default_param(scale_value);
  auto scale = c.NewCNode("Load", {scale_para, monod}, {});

  auto offset_value = tensor::from_spec(kNumberTypeInt8, ShapeVector({1}), device::DeviceType::kCPU);
  offset_value->set_param_info(param_info);
  auto offset_para = c.NewTensorInput("input_1", kInt8, {1});
  offset_para->set_default_param(offset_value);
  auto offset = c.NewCNode("Load", {offset_para, monod}, {});

  auto sqrt_mode = c.NewValueNode(MakeValue<int64_t>(0));
  auto rounding_mode = c.NewValueNode(MakeValue<std::string>("ROUND"));
  auto dst_type = c.NewValueNode(MakeValue<int64_t>(32));

  auto quant = c.NewCNode("QuantV2", {tuple_get_item_0, scale, offset, sqrt_mode, rounding_mode, dst_type}, {});

  c.SetOutput(quant);
  test::RunPass(c.GetGraph(), {std::make_shared<opt::AddRmsNormQuantFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input_0")
    .AddVar("input_1")
    .AddVar("gamma")
    .AddVar("scale")
    .AddVar("offset")
    .AddVar("eps")
    .AddVar("idx")
    .AddCNode("add_rms_norm_quant", {std::make_shared<Primitive>("AddRmsNormQuantV2"), "input_0", "input_1", "gamma",
                                     "scale", "offset", "eps"})
    .AddCNode("get_item", {std::make_shared<Primitive>("TupleGetItem"), "add_rms_norm_quant", "idx"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore

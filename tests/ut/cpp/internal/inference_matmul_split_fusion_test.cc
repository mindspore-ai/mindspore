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
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_add_fusion.h"
#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_matmul_split_fusion.h"
#include "include/utils/anfalgo.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "utils/phase.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "ir/tensor_new.h"

namespace mindspore {
namespace opt {
class InferenceMatmulSplitFusionUT : public UT::Common {
 public:
  InferenceMatmulSplitFusionUT() {}
};

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + Matmul + Reshape + SplitWithSize to MatmulSplitOut3
/// Expectation: After optimize, match MatmulSplitOut3.
TEST_F(InferenceMatmulSplitFusionUT, TestMatmulSplitOut3) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1, 16, 8192});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{16, 8192}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kFloat16, {3072, 8192});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto matmul = c.NewCNode("MatMul", {reshape_0, weight, trans_a, trans_b}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 16, 3072}));
  auto reshape_1 = c.NewCNode("Reshape", {matmul, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{1024, 1024, 1024}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddCNode(
    "MatmulSplitOut3", {std::make_shared<Primitive>("MatmulSplitOut3"), "input", "weight", "dim"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + Matmul + Reshape + SplitWithSize to MatmulSplitOut2
/// Expectation: After optimize, match MatmulSplitOut2.
TEST_F(InferenceMatmulSplitFusionUT, TestMatmulSplitOut2) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1, 256, 12288});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{256, 12288}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kFloat16, {10752, 12288});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto matmul = c.NewCNode("MatMul", {reshape_0, weight, trans_a, trans_b}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 256, 10752}));
  auto reshape_1 = c.NewCNode("Reshape", {matmul, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{5376, 5376}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddCNode(
    "MatmulSplitOut2", {std::make_shared<Primitive>("MatmulSplitOut2"), "input", "weight", "dim"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + Matmul + BiasAdd + Reshape + SplitWithSize to MatmulBiasSplitOut3
/// Expectation: After optimize, match MatmulBiasSplitOut3.
TEST_F(InferenceMatmulSplitFusionUT, TestMatmulBiasSplitOut3) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1, 16, 12288});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{16, 12288}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kFloat16, {4608, 12288});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto matmul = c.NewCNode("MatMul", {reshape_0, weight, trans_a, trans_b}, {});

  auto bias = c.NewTensorInput("bias", kFloat16, {4608});
  auto add = c.NewCNode("Add", {matmul, bias}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 16, 4608}));
  auto reshape_1 = c.NewCNode("Reshape", {add, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{1536, 1536, 1536}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddCNode(
    "MatmulBiasSplitOut3", {std::make_shared<Primitive>("MatmulBiasSplitOut3"), "input", "weight", "dim", "bias"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + Matmul + BiasAdd + Reshape + SplitWithSize to MatmulBiasSplitOut2
/// Expectation: After optimize, match MatmulBiasSplitOut2.
TEST_F(InferenceMatmulSplitFusionUT, TestMatmulBiasSplitOut2) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kFloat16, {1, 1024, 8192});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{1024, 8192}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kFloat16, {5152, 8192});
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto matmul = c.NewCNode("MatMul", {reshape_0, weight, trans_a, trans_b}, {});

  auto bias = c.NewTensorInput("bias", kFloat16, {5152});
  auto add = c.NewCNode("Add", {matmul, bias}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 1024, 5152}));
  auto reshape_1 = c.NewCNode("Reshape", {add, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{2576, 2576}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddCNode(
    "MatmulBiasSplitOut2", {std::make_shared<Primitive>("MatmulBiasSplitOut2"), "input", "weight", "dim", "bias"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + QuantBatchMatmul + Reshape + SplitWithSize to QuantbatchmatmulSplitOut3
/// Expectation: After optimize, match QuantbatchmatmulSplitOut3.
TEST_F(InferenceMatmulSplitFusionUT, TestQuantbatchmatmulSplitOut3) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kInt8, {1, 32, 4096});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{32, 4096}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kInt8, {12288, 4096});
  auto scale = c.NewTensorInput("scale", kInt64, {12288});
  auto bias = c.NewTensorInput("bias", kInt32, {12288});
  auto none_ = c.NewValueNode(kNone);
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto dtype = c.NewValueNode(MakeValue<int64_t>(42));
  auto matmul = c.NewCNode("QuantBatchMatmul", {reshape_0, weight, scale, none_, bias, none_, trans_a, trans_b, dtype}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 32, 12288}));
  auto reshape_1 = c.NewCNode("Reshape", {matmul, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{4096, 4096, 4096}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddVar("scale").AddCNode(
    "QuantbatchmatmulSplitOut3", {std::make_shared<Primitive>("QuantbatchmatmulSplitOut3"),
                                  "input", "weight", "dim", "bias", "scale"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: InferenceMatmulSplitFusion
/// Description: Convert Reshape + QuantBatchMatmul + Reshape + SplitWithSize to QuantbatchmatmulSplitOut2
/// Expectation: After optimize, match QuantbatchmatmulSplitOut2.
TEST_F(InferenceMatmulSplitFusionUT, TestQuantbatchmatmulSplitOut2) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kInt8, {1, 256, 4096});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{256, 4096}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kInt8, {22016, 4096});
  auto scale = c.NewTensorInput("scale", kInt64, {22016});
  auto bias = c.NewTensorInput("bias", kInt32, {22016});
  auto none_ = c.NewValueNode(kNone);
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto dtype = c.NewValueNode(MakeValue<int64_t>(42));
  auto matmul = c.NewCNode("QuantBatchMatmul", {reshape_0, weight, scale, none_, bias, none_, trans_a, trans_b, dtype}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 256, 22016}));
  auto reshape_1 = c.NewCNode("Reshape", {matmul, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{11008, 11008}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddVar("scale").AddCNode(
    "QuantbatchmatmulSplitOut2", {std::make_shared<Primitive>("QuantbatchmatmulSplitOut2"),
                                  "input", "weight", "dim", "bias", "scale"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: QbmmAddFusion + InferenceMatmulSplitFusion
/// Description: Convert Reshape + QuantBatchMatmul + Add + Reshape + SplitWithSize to QuantbatchmatmulSplitOut3
/// Expectation: After optimize, match QuantbatchmatmulSplitOut3.
TEST_F(InferenceMatmulSplitFusionUT, TestQuantbatchmatmulAddSplitOut3) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kInt8, {1, 32, 4096});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{32, 4096}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kInt8, {12288, 4096});
  auto scale = c.NewTensorInput("scale", kInt64, {12288});
  auto bias = c.NewTensorInput("bias", kInt32, {12288});
  auto none_ = c.NewValueNode(kNone);
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto dtype = c.NewValueNode(MakeValue<int64_t>(42));

  auto input_2 = c.NewTensorInput("input_2", kInt64, {12288}); // scale: should have data
  auto input_3 = c.NewTensorInput("input_3", kFloat16, {12288}); // bias: should have data
  ShapeVector sv = {12288};
  ParamInfoPtr param_info = std::make_shared<ParamInfo>(); // fake
  tensor::TensorPtr tensor2 = tensor::from_spec(kNumberTypeInt64, sv, device::DeviceType::kCPU);
  tensor::TensorPtr tensor3 = tensor::from_spec(kNumberTypeFloat16, sv, device::DeviceType::kCPU);
  tensor2->set_param_info(param_info);
  tensor3->set_param_info(param_info);
  input_2->set_default_param(tensor2);
  input_3->set_default_param(tensor3);
  auto load2 = c.NewCNode("Load", {input_2, input_2}, {})->cast<AnfNodePtr>();
  auto load3 = c.NewCNode("Load", {input_3, input_3}, {})->cast<AnfNodePtr>();
  auto matmul = c.NewCNode("QuantBatchMatmul", {reshape_0, weight, load2, none_, none_, none_, trans_a, trans_b, dtype}, {});
  auto add = c.NewCNode("Add", {matmul, load3}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 32, 12288}));
  auto reshape_1 = c.NewCNode("Reshape", {add, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{4096, 4096, 4096}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::QbmmAddFusion>()});
  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddVar("scale").AddCNode(
    "QuantbatchmatmulSplitOut3", {std::make_shared<Primitive>("QuantbatchmatmulSplitOut3"),
                                  "input", "weight", "dim", "bias", "scale"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

/// Feature: A backend pass: QbmmAddFusion + InferenceMatmulSplitFusion
/// Description: Convert Reshape + QuantBatchMatmul + Add + Reshape + SplitWithSize to QuantbatchmatmulSplitOut2
/// Expectation: After optimize, match QuantbatchmatmulSplitOut2.
TEST_F(InferenceMatmulSplitFusionUT, TestQuantbatchmatmulAddSplitOut2) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->SetMsInternalEnableCustomKernelList();
  test::ConstructGraph c;
  auto input = c.NewTensorInput("input", kInt8, {1, 256, 4096});
  auto tuple_0 = c.NewValueNode(MakeValue(std::vector<int64_t>{256, 4096}));
  auto reshape_0 = c.NewCNode("Reshape", {input, tuple_0}, {});

  auto weight = c.NewTensorInput("weight", kInt8, {22016, 4096});
  auto scale = c.NewTensorInput("scale", kInt64, {22016});
  auto bias = c.NewTensorInput("bias", kInt32, {22016});
  auto none_ = c.NewValueNode(kNone);
  auto trans_a = c.NewValueNode(MakeValue<bool>(false));
  auto trans_b = c.NewValueNode(MakeValue<bool>(true));
  auto dtype = c.NewValueNode(MakeValue<int64_t>(42));

  auto input_2 = c.NewTensorInput("input_2", kInt64, {22016}); // scale: should have data
  auto input_3 = c.NewTensorInput("input_3", kFloat16, {22016}); // bias: should have data
  ShapeVector sv = {22016};
  ParamInfoPtr param_info = std::make_shared<ParamInfo>(); // fake
  tensor::TensorPtr tensor2 = tensor::from_spec(kNumberTypeInt64, sv, device::DeviceType::kCPU);
  tensor::TensorPtr tensor3 = tensor::from_spec(kNumberTypeFloat16, sv, device::DeviceType::kCPU);
  tensor2->set_param_info(param_info);
  tensor3->set_param_info(param_info);
  input_2->set_default_param(tensor2);
  input_3->set_default_param(tensor3);
  auto load2 = c.NewCNode("Load", {input_2, input_2}, {})->cast<AnfNodePtr>();
  auto load3 = c.NewCNode("Load", {input_3, input_3}, {})->cast<AnfNodePtr>();
  auto matmul = c.NewCNode("QuantBatchMatmul", {reshape_0, weight, load2, none_, none_, none_, trans_a, trans_b, dtype}, {});
  auto add = c.NewCNode("Add", {matmul, load3}, {});

  auto tuple_1 = c.NewValueNode(MakeValue(std::vector<int64_t>{1, 256, 22016}));
  auto reshape_1 = c.NewCNode("Reshape", {add, tuple_1}, {});
  auto split_size = c.NewValueNode(MakeValue(std::vector<int64_t>{11008, 11008}));
  auto dim = c.NewValueNode(MakeValue<int64_t>(-1));
  auto split_with_size = c.NewCNode("SplitWithSize", {reshape_1, split_size, dim}, {});
  c.SetOutput(split_with_size);

  test::RunPass(c.GetGraph(), {std::make_shared<opt::QbmmAddFusion>()});
  test::RunPass(c.GetGraph(), {std::make_shared<opt::InferenceMatmulSplitFusion>()});
  opt::CheckPattern checker;
  checker.src_pattern_.AddVar("input").AddVar("weight").AddVar("dim").AddVar("bias").AddVar("scale").AddCNode(
    "QuantbatchmatmulSplitOut2", {std::make_shared<Primitive>("QuantbatchmatmulSplitOut2"),
                                  "input", "weight", "dim", "bias", "scale"});

  EXPECT_TRUE(checker.build_pattern_map(c.GetGraph()->output()));
}

}  // namespace opt
}  // namespace mindspore

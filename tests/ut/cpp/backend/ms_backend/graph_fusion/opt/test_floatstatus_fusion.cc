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

#include <mindspore/core/include/ir/core_ops_primitive.h>
#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/floatstatus_fusion.h"
#include "base/float16.h"
#include "utils/anf_utils.h"
#include "ir/functor.h"
#include "ir/tensor_new.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/math_ops.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore::graphkernel::test {
namespace {
enum FloatStatusPattern {
  kFloatStatusBaseFusion,
  kFloatStatusReshapeFusion,
  kCastFloatStatusBaseFusion,
  kCastFloatStatusReshapeFusion,
};

struct Params {
  FloatStatusPattern pattern;
  TypePtr input_type;
  TypePtr isfinite_type;
  TypePtr output_type;
  bool can_fuse;
};

ValueNodePtr ConstOneTensor(TypeId type_id) {
  ShapeVector shape{};
  tensor::TensorPtr tensor;
  if (type_id == kNumberTypeFloat16) {
    tensor = tensor::from_scalar(static_cast<float16>(1.0f), kFloat16);
  } else if (type_id == kNumberTypeFloat32) {
    tensor = tensor::from_scalar(1.0f, kFloat32);
  } else if (type_id == kNumberTypeBFloat16) {
    tensor = tensor::from_scalar(static_cast<bfloat16>(1.0f), kBFloat16);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported data type: " << TypeIdToString(type_id);
  }
  // create value node
  auto value_node = NewValueNode(tensor);
  value_node->set_abstract(tensor->ToAbstract());
  // set build info for value node
  value_node->set_kernel_info(std::make_shared<device::KernelInfo>());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetOutputsFormat(std::vector<std::string>(1, "DefaultFormat"));
  info_builder.SetOutputsDeviceType(std::vector<TypeId>{type_id});
  info_builder.SetOutputsKernelObjectType(std::vector<kernel::KernelObjectType>{kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), value_node.get());
  return value_node;
}
}  // namespace

/// Feature: Test pass FloatStatusFusion
/// Description: check FloatStatusFusion with different data types
/// Expectation: fuse success if input data type is supported
class TestFloatStatusFusion : public GraphKernelCommonTestSuite, public testing::WithParamInterface<Params> {
 public:
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestFloatStatusFusion, floatstatus_fusion) {
  const auto &param = GetParam();
  constexpr size_t num = 4;
  std::vector<AnfNodePtr> ops;
  ConstructGraph c;
  for (size_t i = 0; i < num; ++i) {
    AnfNodePtr x = c.NewTensorInput("x", param.input_type, {3});
    if (param.pattern == kCastFloatStatusBaseFusion || param.pattern == kCastFloatStatusReshapeFusion) {
      MS_EXCEPTION_IF_NULL(param.isfinite_type);
      x = c.NewCNodeWithBuildInfo("Cast", {x, c.NewValueNode<int64_t>(param.isfinite_type->type_id())});
    }
    auto y1 = c.NewCNodeWithBuildInfo("IsFinite", {x});
    ShapeVector axis{0};
    auto y2 = c.NewCNodeWithBuildInfo(
      "ReduceAll", {y1, c.NewValueNode(tensor::from_vector(axis)), c.NewValueNode(MakeValue<bool>(false))});
    auto y3 = c.NewCNodeWithBuildInfo("Cast", {y2, c.NewValueNode<int64_t>(param.output_type->type_id())});
    auto y4 = c.NewCNodeWithBuildInfo("Sub", {ConstOneTensor(param.output_type->type_id()), y3});
    if (param.pattern == kFloatStatusReshapeFusion || param.pattern == kCastFloatStatusReshapeFusion) {
      y4 = c.NewCNodeWithBuildInfo("Reshape", {y4, c.NewValueNode<std::vector<int64_t>>({1})});
    }
    ops.push_back(y4);
  }
  auto output = c.NewCNodeWithBuildInfo("AddN", ops);
  c.SetOutput(output);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::FloatStatusFusion>()});

  // check output
  size_t floatstatus_num = 0;
  for (auto &node : GetAllNodes(fg)) {
    if (node != nullptr && IsPrimitiveCNode(node, prim::kPrimAddN)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      for (size_t i = 1; i < cnode->inputs().size(); ++i) {
        auto input_node = cnode->input(i);
        if (IsPrimitiveCNode(input_node, prim::kPrimFloatStatus)) {
          floatstatus_num += 1;
        }
      }
      break;
    }
  }
  size_t target_num = param.can_fuse ? num : 0;
  ASSERT_EQ(floatstatus_num, target_num);
}

INSTANTIATE_TEST_CASE_P(TestPassFloatStatusFusion, TestFloatStatusFusion,
                        testing::Values(Params{kFloatStatusBaseFusion, kFloat16, nullptr, kFloat32, true},
                                        Params{kFloatStatusBaseFusion, kFloat32, nullptr, kFloat32, true},
                                        Params{kFloatStatusBaseFusion, kBFloat16, nullptr, kFloat32, true},
                                        Params{kFloatStatusBaseFusion, kFloat16, nullptr, kFloat16, false},
                                        Params{kFloatStatusBaseFusion, kFloat32, nullptr, kFloat16, false},
                                        Params{kFloatStatusBaseFusion, kBFloat16, nullptr, kFloat16, false},
                                        Params{kFloatStatusBaseFusion, kFloat16, nullptr, kBFloat16, false},
                                        Params{kFloatStatusBaseFusion, kFloat32, nullptr, kBFloat16, false},
                                        Params{kFloatStatusBaseFusion, kBFloat16, nullptr, kBFloat16, false},
                                        Params{kFloatStatusReshapeFusion, kFloat16, nullptr, kFloat32, true},
                                        Params{kFloatStatusReshapeFusion, kFloat32, nullptr, kFloat32, true},
                                        Params{kFloatStatusReshapeFusion, kFloat16, nullptr, kFloat16, false},
                                        Params{kFloatStatusReshapeFusion, kFloat32, nullptr, kFloat16, false},
                                        Params{kCastFloatStatusBaseFusion, kFloat16, kFloat32, kFloat32, true},
                                        Params{kCastFloatStatusBaseFusion, kFloat32, kFloat16, kFloat32, true},
                                        Params{kCastFloatStatusBaseFusion, kBFloat16, kFloat16, kFloat32, true},
                                        Params{kCastFloatStatusBaseFusion, kBFloat16, kFloat32, kFloat32, true},
                                        Params{kCastFloatStatusBaseFusion, kFloat16, kFloat32, kFloat16, false},
                                        Params{kCastFloatStatusBaseFusion, kFloat32, kFloat16, kFloat16, false},
                                        Params{kCastFloatStatusReshapeFusion, kFloat16, kFloat32, kFloat32, true},
                                        Params{kCastFloatStatusReshapeFusion, kFloat32, kFloat16, kFloat32, true},
                                        Params{kCastFloatStatusReshapeFusion, kFloat16, kFloat32, kFloat16, false},
                                        Params{kCastFloatStatusReshapeFusion, kFloat32, kFloat16, kFloat16, false}));
}  // namespace mindspore::graphkernel::test

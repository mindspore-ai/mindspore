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

#include "backend/ms_backend/graph_fusion/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "backend/ms_backend/graph_fusion/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  TypePtr shape_type;
  bool shape_is_tuple;
  bool shape_is_const;
  TypePtr value_type;
  bool value_is_const;
};
}  // namespace

/// Feature: Test FillV2 expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestFillV2Expander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestFillV2Expander, fill_v2) {
  const auto &param = GetParam();
  ConstructGraph c;

  // shape
  AnfNodePtr shape;
  if (param.shape_is_const) {
    if (param.shape_is_tuple) {
      // shape is tuple
      shape = c.NewValueNode(MakeValue(ShapeVector{2, 3}));
    } else {
      // shape is tensor
      auto shape_type_id = param.shape_type->type_id();
      if (shape_type_id == kNumberTypeInt32) {
        std::vector<int32_t> shape_value{2, 3};
        shape = c.NewValueNode(tensor::from_vector(shape_value));
      } else if (shape_type_id == kNumberTypeInt64) {
        std::vector<int64_t> shape_value{2, 3};
        shape = c.NewValueNode(tensor::from_vector(shape_value));
      } else {
        MS_LOG(ERROR) << "Unsupported shape type: " << TypeIdToString(shape_type_id);
        ASSERT_TRUE(false);
      }
    }
  } else {
    shape = c.NewTensorInput("shape", param.shape_type, {2});
  }

  // value
  AnfNodePtr value;
  if (param.value_is_const) {
    value = c.NewValueNode(NewScalar(param.value_type, 2, false));  // Tensor shape ()
  } else {
    value = c.NewTensorInput("value", param.value_type, {});
  }
  auto op = c.NewCNodeWithBuildInfo("FillV2", {shape, value});
  c.SetOutput(op);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  size_t gk_size = param.can_expand ? 1 : 0;
  ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
}

INSTANTIATE_TEST_CASE_P(
  TestOpFillV2, TestFillV2Expander,
  testing::Values(Params{true, kInt32, false, true, kFloat16, true}, Params{true, kInt32, false, true, kFloat32, true},
                  Params{true, kInt32, false, true, kBFloat16, true}, Params{true, kInt32, false, true, kInt32, true},
                  Params{true, kInt64, false, true, kFloat16, true}, Params{true, kInt64, false, true, kFloat32, true},
                  Params{true, kInt64, false, true, kBFloat16, true}, Params{true, kInt64, false, true, kInt32, true},
                  Params{true, kInt64, true, true, kFloat16, true}, Params{true, kInt64, true, true, kFloat32, true},
                  Params{true, kInt64, true, true, kBFloat16, true}, Params{true, kInt64, true, true, kInt32, true},
                  Params{false, kInt64, false, true, kInt64, true}, Params{false, kInt64, false, true, kInt16, true},
                  Params{false, kInt64, false, true, kInt8, true}, Params{false, kInt64, false, true, kFloat16, false},
                  Params{false, kInt64, false, false, kFloat16, true}));
}  // namespace mindspore::graphkernel::test

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

#include "test_view.h"
#include "mindspore/core/ops/view/broadcast_to_strides_calc.h"

namespace mindspore {
namespace ops {
class TestViewBroadcastTo : public TestView {
 public:
  TestViewBroadcastTo() {}
};

/// Feature: BroadcastTo strides calculator
/// Description: Test view BroadcastTo strides calculator is right
/// Expectation: success
TEST_F(TestViewBroadcastTo, func) {
  auto prim = std::make_shared<Primitive>("BroadcastTo");

  std::vector<int64_t> perm_data({2, 1, 4});
  auto input_perm = MakeValue(perm_data);
  prim->AddAttr(kShape, input_perm);

  std::vector<int64_t> tensor_data = {1, 2, 3, 4};
  auto input_tensor = std::make_shared<tensor::Tensor>(tensor_data, kInt64);
  input_tensor->set_shape({1, 4});

  auto storage_list = BroadCastToCalc(prim, std::vector<ValuePtr>({input_tensor}));
  std::vector<int64_t> expect_shape({2, 1, 4});
  std::vector<int64_t> expect_strides({0, 4, 1});
  size_t expect_size = 1;

  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides);
}

/// Feature: BroadcastTo strides calculator
/// Description: Test view BroadcastTo strides calculator is right
/// Expectation: success
TEST_F(TestViewBroadcastTo, BroadDim) {
  auto prim = std::make_shared<Primitive>("BroadcastTo");

  std::vector<int64_t> input_perm_data({2, -1, -1, 3});
  auto input_perm = MakeValue(input_perm_data);
  prim->AddAttr(kShape, input_perm);

  std::vector<int64_t> tensor_shape = {1, 2, 3};
  size_t tensor_total_length = 1;
  for (auto it : tensor_shape) {
    tensor_total_length *= it;
  }
  std::vector<int64_t> tensor_data(tensor_total_length, 0);
  auto input_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shape, (void *)tensor_data.data(),
                                                       tensor_total_length * sizeof(int64_t));

  std::vector<ValuePtr> inputs{input_tensor};
  auto storage_list = BroadCastToCalc(prim, inputs);
  std::vector<int64_t> expect_shape({2, 1, 2, 3});
  std::vector<int64_t> expect_strides({0, 6, 3, 1});
  size_t expect_size = 1;
  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides);

  std::vector<int64_t> perm_data({3, 2, 3});
  input_perm = MakeValue(perm_data);
  prim->AddAttr(kShape, input_perm);

  storage_list = BroadCastToCalc(prim, inputs);
  std::vector<int64_t> expect_shape_2({3, 2, 3});
  std::vector<int64_t> expect_strides_2({0, 3, 1});
  ASSERT_EQ(storage_list.size(), expect_size);
  ASSERT_FALSE(storage_list[0]->is_contiguous);
  ASSERT_TRUE(storage_list[0]->shape == expect_shape_2);
  ASSERT_TRUE(storage_list[0]->strides == expect_strides_2);
}
}  // namespace ops
}  // namespace mindspore
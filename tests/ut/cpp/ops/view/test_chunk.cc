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

#include "test_view.h"
#include "ir/tensor_new.h"
#include "view/chunk_view_strides_calc.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace ops {
class TestViewChunk : public TestView {
 public:
  TestViewChunk() {}
};

/// Feature: Chunk strides calculator
/// Description: Test view Chunk strides calculator is right
/// Expectation: success
TEST_F(TestViewChunk, ChunkFunc) {
  // input with rank 0
  auto zero_dim_tensor = tensor::from_scalar(2);
  ASSERT_THROW(ChunkViewBasicTypeCalc(zero_dim_tensor, 1, 1), std::exception);
  // output num < 0
  zero_dim_tensor->set_shape({-4, 1});
  ASSERT_THROW(ChunkViewBasicTypeCalc(zero_dim_tensor, 2, 0), std::exception);

  std::vector<int64_t> tensor_data{};
  auto input_tensor = tensor::from_vector(tensor_data, kInt64);
  input_tensor->set_shape({0, 4});
  // chunks < 1
  ASSERT_THROW(ChunkViewBasicTypeCalc(input_tensor, -1, 1), std::exception);

  auto storage_list = ChunkViewBasicTypeCalc(input_tensor, 2, 0);
  std::vector<int64_t> expect_shape{0, 4};
  std::vector<int64_t> expect_strides{4, 1};
  constexpr size_t expect_size = 2;
  ASSERT_EQ(storage_list.size(), expect_size);
  for (size_t i = 0; i < expect_size; ++i) {
    ASSERT_TRUE(storage_list[i]->is_contiguous);
    ASSERT_TRUE(storage_list[i]->shape == expect_shape);
    ASSERT_TRUE(storage_list[i]->strides == expect_strides);
  }
}
}  // namespace ops
}  // namespace mindspore
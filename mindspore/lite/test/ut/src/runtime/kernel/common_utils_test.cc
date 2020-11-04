/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <vector>
#include "common/common_test.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace kernel {
class CommonUtilTest : public mindspore::CommonTest {
 public:
  CommonUtilTest() = default;
};

TEST_F(CommonUtilTest, BucketReduceSparseGradient1) {
  // The indices is a vector and the grad is a tensor with shape (6, 2)
  /* 0
   * 0
   * 1
   * 1
   * 0
   * 3
   */
  std::vector<int64_t> indices{0, 0, 1, 1, 0, 3};
  /* 0 1
   * 2 3
   * 4 5
   * 6 7
   * 8 9
   * 10 11
   */
  std::vector<float> grad;
  for (int i = 0; i < 6 * 2; i++) {
    grad.push_back(i);
  }
  std::vector<int64_t> unique_indices(6);
  std::vector<float> summed_grad(12);
  std::vector<int64_t> tmp_indices(6);
  std::vector<float> tmp_grad(12);

  SparseGradient unique_grad({summed_grad.data(), unique_indices.data(), 6});
  SparseGradient workspace_grad({tmp_grad.data(), tmp_indices.data(), 6});
  SparseGradient input_grad({grad.data(), indices.data(), 6});

  ReduceSparseGradientParam param;
  param.input_grad_ = &input_grad;
  param.workspace_grad_ = &workspace_grad;
  param.output_grad_ = &unique_grad;
  param.max_index_ = 6;
  param.value_stride_ = 2;
  BucketReduceSparseGradient(param);

  EXPECT_EQ(unique_grad.indices_size_, 3);
  std::vector<int64_t> expect_indices({0, 1, 3});
  for (size_t i = 0; i < unique_grad.indices_size_; ++i) {
    EXPECT_EQ(unique_grad.indices_[i], expect_indices[i]);
  }
  /* 10 13
   * 10 12
   * 10 11
   */
  std::vector<int64_t> expect_value({10, 13, 10, 12, 10, 11});
  for (size_t i = 0; i < unique_grad.indices_size_ * 2; ++i) {
    EXPECT_EQ(unique_grad.value_[i], expect_value[i]);
  }
}

TEST_F(CommonUtilTest, BucketReduceSparseGradient2) {
  // The indices is a vector and the grad is a tensor with shape (6, 2)
  /* 0
   * 0
   * 1
   * 1
   * 0
   * 6
   */
  std::vector<int64_t> indices{0, 0, 1, 1, 0, 6};
  /* 0 1
   * 2 3
   * 4 5
   * 6 7
   * 8 9
   * 10 11
   */
  std::vector<float> grad;
  for (int i = 0; i < 6 * 2; i++) {
    grad.push_back(i);
  }
  std::vector<int64_t> unique_indices(6);
  std::vector<float> summed_grad(12);
  std::vector<int64_t> tmp_indices(6);
  std::vector<float> tmp_grad(12);
  SparseGradient unique_grad({summed_grad.data(), unique_indices.data(), 6});
  SparseGradient workspace_grad({tmp_grad.data(), tmp_indices.data(), 6});
  SparseGradient input_grad({grad.data(), indices.data(), 6});

  ReduceSparseGradientParam param;
  param.input_grad_ = &input_grad;
  param.workspace_grad_ = &workspace_grad;
  param.output_grad_ = &unique_grad;
  param.max_index_ = 6;
  param.value_stride_ = 2;
  BucketReduceSparseGradient(param);

  EXPECT_EQ(unique_grad.indices_size_, 2);

  std::vector<int64_t> expect_indices({0, 1});
  for (size_t i = 0; i < unique_grad.indices_size_; ++i) {
    EXPECT_EQ(unique_grad.indices_[i], expect_indices[i]);
  }

  /* 10 13
   * 10 12
   */
  std::vector<int64_t> expect_value({10, 13, 10, 12});
  for (size_t i = 0; i < unique_grad.indices_size_ * 2; ++i) {
    EXPECT_EQ(unique_grad.value_[i], expect_value[i]);
  }
}
}  // namespace kernel
}  // namespace mindspore

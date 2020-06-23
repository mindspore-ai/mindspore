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
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class CommonUtilTest : public UT::Common {
 public:
  CommonUtilTest() = default;
};

TEST_F(CommonUtilTest, DeduplicateIndexedSlicesTest1) {
  // The indices is a vector and the grad is a tensor with shape (6, 2)
  /* 0
   * 0
   * 1
   * 1
   * 0
   * 3
   */
  std::vector<int> indices{0, 0, 1, 1, 0, 3};
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
  std::vector<int> unique_indices(3);
  std::vector<float> summed_grad(6);
  SparseGradient unique_grad({summed_grad.data(), unique_indices.data(), 0});
  ReduceSparseGradient(SparseGradient({grad.data(), indices.data(), 6}), &unique_grad, 6, 2);
  EXPECT_EQ(unique_grad.indices_size_, 3);
  EXPECT_EQ(unique_indices, std::vector<int>({0, 1, 3}));
  /* 10 13
   * 10 12
   * 10 11
   */
  EXPECT_EQ(summed_grad, std::vector<float>({10, 13, 10, 12, 10, 11}));
}

TEST_F(CommonUtilTest, DeduplicateIndexedSlicesTest2) {
  // The indices is a vector and the grad is a tensor with shape (6, 2)
  /* 0
   * 0
   * 1
   * 1
   * 0
   * 6
   */
  std::vector<int> indices{0, 0, 1, 1, 0, 6};
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
  std::vector<int> unique_indices(2);
  std::vector<float> summed_grad(4);
  SparseGradient unique_grad({summed_grad.data(), unique_indices.data(), 0});
  ReduceSparseGradient(SparseGradient({grad.data(), indices.data(), 6}), &unique_grad, 6, 2);
  EXPECT_EQ(unique_grad.indices_size_, 2);
  EXPECT_EQ(unique_indices, std::vector<int>({0, 1}));
  /* 10 13
   * 10 12
   */
  EXPECT_EQ(summed_grad, std::vector<float>({10, 13, 10, 12}));
}
}  // namespace kernel
}  // namespace mindspore

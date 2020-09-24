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

#include <cmath>
#include <memory>
#include "common/common_test.h"
#include "internal/include/vector.h"
#include "nnacl/op_base.h"

namespace mindspore {
class VectorTest : public mindspore::CommonTest {
 public:
  VectorTest() {}
};

void CheckArrValue(Vector<int> arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    ASSERT_EQ(arr[i], i);
  }
}

TEST_F(VectorTest, VectorTest1) {
  constexpr int kLen1 = 10;
  Vector<int> arr1(kLen1);
  for (int i = 0; i < kLen1; ++i) {
    arr1[i] = i;
  }
  Vector<int> arr2 = arr1;
  ASSERT_EQ(arr2.size(), kLen1);
  for (int i = 0; i < kLen1; ++i) {
    ASSERT_EQ(arr2[i], i);
  }

  Vector<int> arr3;
  for (int i = 0; i < kLen1; ++i) {
    arr3.push_back(std::move(arr1[i]));
  }
  CheckArrValue(arr3);
}

}  // namespace mindspore

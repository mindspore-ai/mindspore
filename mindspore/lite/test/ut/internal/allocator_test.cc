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

#include "common/common_test.h"
#include "internal/include/model.h"
#include "internal/include/errorcode.h"
#include "nnacl/op_base.h"
#undef private
#define private public
#include "internal/src/allocator.h"
#undef private

namespace mindspore {
class AllocatorTest : public mindspore::CommonTest {
 public:
  AllocatorTest() {}
};

TEST_F(AllocatorTest, AllocatorTest1) {
  lite::DefaultAllocator allocator;
  constexpr int data1_size = 10 * sizeof(float);
  ASSERT_EQ(allocator.allocated_list_[0], nullptr);
  float *data1 = reinterpret_cast<float *>(allocator.Malloc(data1_size));
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(allocator.allocated_list_[0], nullptr);

  ASSERT_EQ(allocator.free_list_[0], nullptr);
  allocator.Free(data1);
  ASSERT_EQ(allocator.allocated_list_[0], nullptr);
  ASSERT_NE(allocator.free_list_[0], nullptr);
}

TEST_F(AllocatorTest, AllocatorTest2) {
  lite::DefaultAllocator allocator;
  constexpr int data1_size = 10 * sizeof(float);
  ASSERT_EQ(allocator.allocated_list_[0], nullptr);
  float *data1 = reinterpret_cast<float *>(allocator.Malloc(data1_size));
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(allocator.allocated_list_[0], nullptr);

  constexpr int data2_size = (1024 << lite::kBlockRange);
  ASSERT_EQ(allocator.large_mem_list_, nullptr);
  float *data2 = reinterpret_cast<float *>(allocator.Malloc(data2_size));
  ASSERT_NE(data2, nullptr);
  ASSERT_NE(allocator.large_mem_list_, nullptr);

  constexpr int data3_size = (1024 << 3);
  ASSERT_EQ(allocator.allocated_list_[3], nullptr);
  float *data3 = reinterpret_cast<float *>(allocator.Malloc(data3_size));
  ASSERT_NE(data3, nullptr);
  ASSERT_NE(allocator.allocated_list_[3], nullptr);

  int expect_total_size = data1_size + data2_size + data3_size;
  size_t total_size = allocator.GetTotalSize();
  ASSERT_EQ(total_size, expect_total_size);

  allocator.Clear();
  total_size = allocator.GetTotalSize();
  ASSERT_EQ(total_size, 0);
}

TEST_F(AllocatorTest, AllocatorTest3) {
  lite::DefaultAllocator allocator;
  constexpr int data1_size = 10 * sizeof(float);
  ASSERT_EQ(allocator.allocated_list_[0], nullptr);
  float *data1 = reinterpret_cast<float *>(allocator.Malloc(data1_size));
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(allocator.allocated_list_[0], nullptr);

  constexpr int data2_size = 11 * sizeof(float);
  float *data2 = reinterpret_cast<float *>(allocator.Malloc(data2_size));
  ASSERT_NE(data2, nullptr);

  constexpr int data3_size = 12 * sizeof(float);
  float *data3 = reinterpret_cast<float *>(allocator.Malloc(data3_size));
  ASSERT_NE(data3, nullptr);

  int expect_total_size = data1_size + data2_size + data3_size;
  size_t total_size = allocator.GetTotalSize();
  ASSERT_EQ(total_size, expect_total_size);

  allocator.Free(data2);
  total_size = allocator.GetTotalSize();
  ASSERT_EQ(total_size, expect_total_size);
}
}  // namespace mindspore

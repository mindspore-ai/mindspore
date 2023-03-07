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

#include <memory>
#include <vector>
#include <string>

#include "backend/common/mem_reuse/kernel_refcount.h"

#include "include/common/utils/utils.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

using mindspore::memreuse::KernelRefCount;
using mindspore::memreuse::KernelRefCountPtr;
using mindspore::memreuse::RefCountType;
namespace mindspore {
class TestKernelRefCount : public UT::Common {
 public:
  TestKernelRefCount() {}
  virtual void SetUp();
  virtual void TearDown();
};

void TestKernelRefCount::SetUp() { UT::InitPythonPath(); }

void TestKernelRefCount::TearDown() {}

TEST_F(TestKernelRefCount, test_InitKernelRefCount) {
  KernelRefCountPtr kernel_ref_count_ptr = std::make_shared<KernelRefCount>();
  int ref_count = 3;
  size_t offset = 512;
  size_t size = 256;
  int index = 2;
  kernel_ref_count_ptr->index_ = index;
  kernel_ref_count_ptr->offset_ = offset;
  kernel_ref_count_ptr->ref_count_ = ref_count;
  kernel_ref_count_ptr->size_ = size;
  ASSERT_NE(kernel_ref_count_ptr, nullptr);
}

TEST_F(TestKernelRefCount, test_RefCount) {
  KernelRefCountPtr kernel_ref_count_ptr = std::make_shared<KernelRefCount>();
  int ref_count = kernel_ref_count_ptr->ref_count_;
  ASSERT_EQ(ref_count, 0);
}

TEST_F(TestKernelRefCount, test_SetKernelRefInfo) {
  KernelRefCountPtr kernel_ref_count_ptr = std::make_shared<KernelRefCount>();
  size_t size = 256;
  int index = 2;
  RefCountType ref_count_type = memreuse::kDynamicRefCount;
  kernel_ref_count_ptr->SetKernelRefCountInfo(index, size, ref_count_type);
  ASSERT_EQ(kernel_ref_count_ptr->index_, index);
}

}  // namespace mindspore

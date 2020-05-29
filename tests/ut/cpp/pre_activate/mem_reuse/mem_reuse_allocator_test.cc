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
#include "operator/ops.h"
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/mem_reuse_allocator.h"

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

using mindspore::memreuse::BestFitMemReuse;
using mindspore::memreuse::KernelDef;
using mindspore::memreuse::KernelDefPtr;
using mindspore::memreuse::KernelRefCount;
using mindspore::memreuse::KernelRefCountPtr;
using mindspore::memreuse::MemReuseUtil;
using mindspore::memreuse::MemReuseUtilPtr;
using mindspore::memreuse::RefCountType;
using MembufPtr = std::shared_ptr<mindspore::memreuse::Membuf>;

namespace mindspore {
namespace memreuse {
class TestMemReuseAllocator : public UT::Common {
 public:
  TestMemReuseAllocator() : getPyFun_("gtest_input.mem_reuse.TestMemReuseAllocator", true) {}
  void SetUp() {}

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

KernelDefPtr GetNewKernelDef(const std::vector<KernelRefCountPtr> &inputs,
                             const std::vector<KernelRefCountPtr> &outputs, uint32_t stream_id) {
  auto kernel_def = std::make_shared<KernelDef>();
  kernel_def->set_input_refs(inputs);
  kernel_def->set_output_refs(outputs);
  kernel_def->set_stream_id(stream_id);
  return kernel_def;
}

void InitMemReuseUtils(MemReuseUtil *mem_reuse_util_ptr) {
  // tensor params: ref_count, offset, size, index,
  auto tensor_0 = std::make_shared<KernelRefCount>();
  tensor_0->index_ = 0;
  tensor_0->size_ = 512;
  tensor_0->ref_count_ = 999;
  ASSERT_NE(tensor_0, nullptr);
  auto tensor_1 = std::make_shared<KernelRefCount>();
  tensor_1->index_ = 1;
  tensor_1->size_ = 1024;
  tensor_1->ref_count_ = 1;
  auto tensor_2 = std::make_shared<KernelRefCount>();
  tensor_2->index_ = 2;
  tensor_2->size_ = 1024;
  tensor_2->ref_count_ = 2;
  auto tensor_3 = std::make_shared<KernelRefCount>();
  tensor_3->index_ = 3;
  tensor_3->size_ = 32;
  tensor_3->ref_count_ = 1;
  auto tensor_4 = std::make_shared<KernelRefCount>();
  tensor_4->index_ = 4;
  tensor_4->size_ = 2048;
  tensor_4->ref_count_ = 1;
  auto tensor_5 = std::make_shared<KernelRefCount>();
  tensor_5->index_ = 5;
  tensor_5->size_ = 256;
  tensor_5->ref_count_ = 1;
  MS_LOG(INFO) << "init all tensor info success.";

  std::vector<KernelRefCountPtr> inputs;
  std::vector<KernelRefCountPtr> outputs;
  inputs = {tensor_0};
  outputs = {tensor_1};
  auto kernel0 = GetNewKernelDef(inputs, outputs, 0);
  inputs = {tensor_1};
  outputs = {tensor_2};
  auto kernel1 = GetNewKernelDef(inputs, outputs, 0);
  inputs = {tensor_2};
  outputs = {tensor_3};
  auto kernel2 = GetNewKernelDef(inputs, outputs, 0);
  inputs = {tensor_2, tensor_3};
  outputs = {tensor_4};
  auto kernel3 = GetNewKernelDef(inputs, outputs, 0);
  inputs = {tensor_4};
  outputs = {tensor_5};
  auto kernel4 = GetNewKernelDef(inputs, outputs, 1);
  MS_LOG(INFO) << "init all op info success.";
  std::vector<KernelRefCountPtr> tensor_ptr_list{tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5};
  std::vector<KernelDefPtr> op_ptr_list{kernel0, kernel1, kernel2, kernel3, kernel4};

  mem_reuse_util_ptr->set_total_refs_list(tensor_ptr_list);
  mem_reuse_util_ptr->set_kernel_def_ptr_list(op_ptr_list);
}

TEST_F(TestMemReuseAllocator, mem_reuse_allocator) {
  MS_LOG(INFO) << "mem_resue_allocator UT";
  auto mem_reuse_util_ptr = std::make_shared<MemReuseUtil>();
  InitMemReuseUtils(mem_reuse_util_ptr.get());
  auto best_fit_mem_reuse = std::make_shared<BestFitMemReuse>();
  best_fit_mem_reuse->Reuse(mem_reuse_util_ptr.get());
  MS_LOG(INFO) << "run mem reuse success";
  size_t total_allocated_size = best_fit_mem_reuse->GetAllocatedSize();
  ASSERT_NE(total_allocated_size, 0);
}

TEST_F(TestMemReuseAllocator, mem_reuse_allocator_add_membuf) {
  auto best_fit_mem_reuse = std::make_shared<BestFitMemReuse>();
  auto tensor_desc = std::make_shared<KernelRefCount>();
  tensor_desc->SetKernelRefCountInfo(0, 1024, kDynamicRefCount);
  best_fit_mem_reuse->AddNewMembufPtr(tensor_desc.get(), kDynamicMem);
  auto allocated_size = best_fit_mem_reuse->GetAllocatedSize();
  ASSERT_EQ(allocated_size, 1024);
}

TEST_F(TestMemReuseAllocator, mem_reuse_allocator_split_membuf) {
  auto best_fit_mem_reuse = std::make_shared<BestFitMemReuse>();
  auto tensor_0 = std::make_shared<KernelRefCount>();
  tensor_0->SetKernelRefCountInfo(0, 2048, kDynamicRefCount);
  best_fit_mem_reuse->AddNewMembufPtr(tensor_0.get(), kDynamicMem);

  auto tensor_1 = std::make_shared<KernelRefCount>();
  tensor_1->SetKernelRefCountInfo(1, 800, kDynamicRefCount);
  auto is_split = best_fit_mem_reuse->IsSplit(tensor_1->size_, tensor_0->size_);
  ASSERT_EQ(is_split, true);

  best_fit_mem_reuse->SplitMembuf(tensor_1.get(), 0);
  auto allocated_size = best_fit_mem_reuse->GetAllocatedSize();
  ASSERT_EQ(allocated_size, 2048);
}

TEST_F(TestMemReuseAllocator, mem_reuse_allocator_align) {
  auto best_fit_mem_reuse = std::make_shared<BestFitMemReuse>();
  auto size = best_fit_mem_reuse->AlignMemorySize(510);
  ASSERT_EQ(size, 1024);
}
}  // namespace memreuse
}  // namespace mindspore

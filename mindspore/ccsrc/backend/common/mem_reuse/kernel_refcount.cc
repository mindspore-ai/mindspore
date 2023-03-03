/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "backend/common/mem_reuse/kernel_refcount.h"
#include <algorithm>
#include <iterator>

namespace mindspore {
namespace memreuse {
/**
 * Add some set && get function
 */
void KernelRefCount::SetKernelRefCountInfo(int index, size_t size, RefCountType reftype) {
  index_ = index;
  size_ = size;
  reftype_ = reftype;
}

std::vector<int> KernelDef::GetInputRefIndexs() const {
  std::vector<int> input_ref_indexs;
  if (input_refs_.empty()) {
    return input_ref_indexs;
  }
  (void)std::transform(input_refs_.begin(), input_refs_.end(), std::back_inserter(input_ref_indexs),
                       [](const KernelRefCountPtr &ref_info) { return ref_info->index_; });
  return input_ref_indexs;
}

std::vector<int> KernelDef::GetOutputRefIndexs() const {
  std::vector<int> output_ref_indexs;
  if (output_refs_.empty()) {
    return output_ref_indexs;
  }
  (void)std::transform(output_refs_.begin(), output_refs_.end(), std::back_inserter(output_ref_indexs),
                       [](const KernelRefCountPtr &ref_info) { return ref_info->index_; });
  return output_ref_indexs;
}

std::vector<int> KernelDef::GetWorkspaceRefIndexs() const {
  std::vector<int> wk_ref_indexs;
  if (wk_space_.empty()) {
    return wk_ref_indexs;
  }
  // only one key
  auto wk_refs_iter = wk_space_.begin();
  auto wk_refs = wk_refs_iter->second;
  (void)std::transform(wk_refs.begin(), wk_refs.end(), std::back_inserter(wk_ref_indexs),
                       [](const KernelRefCountPtr &ref_info) { return ref_info->index_; });
  return wk_ref_indexs;
}
}  // namespace memreuse
}  // namespace mindspore

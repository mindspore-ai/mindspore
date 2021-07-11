/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/api/delegate.h"
namespace mindspore {
const schema::Primitive *DelegateModel::GetPrimitive(kernel::Kernel *kernel) const {
  if (primitives_.find(kernel) != primitives_.end()) {
    return primitives_.at(kernel);
  } else {
    return nullptr;
  }
}

KernelIter DelegateModel::BeginKernelIterator() { return kernels_->begin(); }

KernelIter DelegateModel::EndKernelIterator() { return kernels_->end(); }

KernelIter DelegateModel::Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel) {
  size_t insert_index = from - BeginKernelIterator();
  if (insert_index >= kernels_->size()) {
    return BeginKernelIterator();
  }
  kernels_->erase(from, end);
  kernels_->insert(BeginKernelIterator() + insert_index, graph_kernel);
  return BeginKernelIterator() + insert_index + 1;
}
}  // namespace mindspore

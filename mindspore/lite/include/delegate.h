/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_DELEGATE_DELEGATE_H_
#define MINDSPORE_LITE_DELEGATE_DELEGATE_H_

#include <map>
#include <vector>
#include "include/ms_tensor.h"
#include "include/context.h"
#include "include/kernel.h"

namespace mindspore {
using KernelIter = std::vector<kernel::Kernel *>::iterator;
class DelegateModel {
 public:
  DelegateModel(std::vector<kernel::Kernel *> *kernels,
                std::map<kernel::Kernel *, const schema::Primitive *> primitives)
      : kernels_(kernels), primitives_(primitives) {}

  ~DelegateModel() = default;

  const schema::Primitive *GetPrimitive(kernel::Kernel *kernel);

  KernelIter BeginKernelIterator();

  KernelIter EndKernelIterator();

  KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel);

 protected:
  std::vector<kernel::Kernel *> *kernels_;
  std::map<kernel::Kernel *, const schema::Primitive *> primitives_;
};

class Delegate {
 public:
  Delegate() = default;

  virtual ~Delegate() = default;

  virtual int Init() = 0;

  virtual int Build(DelegateModel *model) = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_DELEGATE_DELEGATE_H_

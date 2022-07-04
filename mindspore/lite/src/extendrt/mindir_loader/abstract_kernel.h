/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_KERNEL_H_

#include <vector>

#include "include/api/kernel.h"
#include "src/tensor.h"

using mindspore::kernel::Kernel;

namespace mindspore::infer {
class Abstractkernel : public Kernel {
 public:
  //   virtual OpParameter *op_parameter() const = 0;

  virtual int Train() = 0;

  virtual bool IsTrain() const = 0;

  virtual int Eval() = 0;

  virtual bool IsEval() const = 0;

  virtual void SetTrainable(bool trainable = true) = 0;

  virtual bool IsTrainable() const = 0;

  virtual void set_in_tensors(const std::vector<lite::Tensor *> &in_tensors) = 0;

  virtual void set_in_tensor(lite::Tensor *in_tensor, size_t index) = 0;

  virtual void set_out_tensors(const std::vector<lite::Tensor *> &out_tensors) = 0;

  virtual void set_out_tensor(lite::Tensor *out_tensor, size_t index) = 0;

  virtual const std::vector<lite::Tensor *> &in_tensors() const = 0;

  virtual const std::vector<lite::Tensor *> &out_tensors() const = 0;
};
}  // namespace mindspore::infer

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_KERNEL_H_

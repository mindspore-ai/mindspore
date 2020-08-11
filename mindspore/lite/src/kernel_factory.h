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

#ifndef MINDSPORE_LITE_SRC_KERNEL_FACTORY_H_
#define MINDSPORE_LITE_SRC_KERNEL_FACTORY_H_

#include <vector>
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/include/context.h"
#include "mindspore/lite/src/ir/tensor.h"
#include "schema/model_generated.h"

namespace mindspore::lite {
class KernelFactory {
 public:
  KernelFactory();
  virtual ~KernelFactory();

  static KernelFactory *GetInstance();
  kernel::LiteKernel *GetKernel(const std::vector<tensor::Tensor *> &in_tensors,
                                const std::vector<tensor::Tensor *> &out_tensors, const lite::Primitive *primitive,
                                const Context *ctx, const kernel::KernelKey &key);
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_KERNEL_FACTORY_H_

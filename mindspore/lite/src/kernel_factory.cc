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

#include "mindspore/lite/src/kernel_factory.h"
#include "utils/log_adapter.h"
#include "src/populate_parameter.h"
#include "schema/model_generated.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::kernel::KernelKey;
using mindspore::kernel::LiteKernel;

namespace mindspore::lite {
KernelFactory::KernelFactory() = default;

KernelFactory::~KernelFactory() = default;

KernelFactory *KernelFactory::GetInstance() {
  static KernelFactory instance;
  return &instance;
}

LiteKernel *KernelFactory::GetKernel(const std::vector<tensor::Tensor *> &in_tensors,
                                     const std::vector<tensor::Tensor *> &out_tensors, const lite::Primitive *primitive,
                                     const Context *ctx, const kernel::KernelKey &key) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(ctx);
  auto parameter = kernel::PopulateParameter(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(primitive->Type());
    return nullptr;
  }
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator != nullptr) {
    auto kernel = creator(in_tensors, out_tensors, parameter, ctx, key, primitive);
    return kernel;
  }
  return nullptr;
}
}  // namespace mindspore::lite

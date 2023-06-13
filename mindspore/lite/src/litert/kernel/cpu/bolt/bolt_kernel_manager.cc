/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "bolt/bolt_kernel_manager.h"
#include "bolt/bolt_parameter_manager.h"

namespace mindspore::kernel::bolt {
bool BoltSupportKernel(int op_type, TypeId data_type) {
  auto creator = BoltKernelRegistry::GetInstance()->Creator({op_type, data_type});
  if (creator != nullptr) {
    return true;
  }
  return false;
}

LiteKernel *BoltKernelRegistry(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               kernel::KernelKey *key) {
  auto creator = BoltKernelRegistry::GetInstance()->Creator({key->type, key->data_type});
  LiteKernel *kernel = nullptr;
  if (creator != nullptr) {
    kernel = creator(param_spec, inputs, outputs, ctx);
  }
  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create bolt kernel failed!";
    return nullptr;
  }
  key->format = BoltKernelRegistry::GetInstance()->GetKernelFormat({key->type, key->data_type});
  return kernel;
}

LiteKernel *BoltKernelRegistry(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                               kernel::KernelKey *key) {
  // convert OpParameter to ParameterSpec
  auto param_spec = BoltParameterRegistry::GetInstance()->CreateBoltParameter(parameter);
  if (param_spec == nullptr) {
    MS_LOG(DEBUG) << "Create bolt ParameterSpec failed!";
    return nullptr;
  }
  return BoltKernelRegistry(*param_spec, inputs, outputs, ctx, key);
}
}  // namespace mindspore::kernel::bolt

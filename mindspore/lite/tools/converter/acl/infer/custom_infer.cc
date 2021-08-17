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

#include "tools/converter/acl/infer/custom_infer.h"
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "include/errorcode.h"
#include "common/log_adapter.h"

namespace mindspore {
namespace lite {
Status CustomInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                              const mindspore::schema::Primitive *primitive) {
  if (inputs == nullptr || (*inputs).empty()) {
    MS_LOG(ERROR) << "Inputs is invalid.";
    return kLiteError;
  }
  if (outputs == nullptr || (*outputs).empty()) {
    MS_LOG(ERROR) << "Outputs is invalid.";
    return kLiteError;
  }
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr.";
    return kLiteError;
  }
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom.";
    return kLiteError;
  }
  return kSuccess;
}

std::shared_ptr<mindspore::kernel::KernelInterface> CustomInferCreater() {
  auto infer = new (std::nothrow) CustomInterface();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return std::shared_ptr<mindspore::kernel::KernelInterface>(infer);
}
}  // namespace lite
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, ACL, lite::CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore

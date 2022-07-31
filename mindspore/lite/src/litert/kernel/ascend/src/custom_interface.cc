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

#include "src/litert/kernel/ascend/src/custom_interface.h"
#include <memory>
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"
#include "common/log_adapter.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore::kernel {
namespace acl {
constexpr auto kBase = 10;
constexpr auto kBufMaxSize = 1024;
constexpr auto kOutputShapes = "outputs_shape";

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
  MS_CHECK_TRUE_MSG(primitive != nullptr, kLiteNullptr, "Primitive is nullptr.");
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom.";
    return kLiteError;
  }
  auto op = primitive->value_as_Custom();
  char buf[kBufMaxSize];
  if (GetCustomAttr(buf, kBufMaxSize, op, kOutputShapes) != kSuccess) {
    MS_LOG(ERROR) << "Get custom attr output shape failed.";
    return kLiteError;
  }
  uint32_t id = 0;
  char delims[] = ",";
  char *save_ptr = nullptr;
  char *res = strtok_r(buf, delims, &save_ptr);
  while (res != nullptr && id < outputs->size()) {
    int64_t dims_num = strtol(res, &res, kBase);
    std::vector<int64_t> shape(dims_num);
    for (int64_t j = 0; j < dims_num; j++) {
      res = strtok_r(nullptr, delims, &save_ptr);
      shape[j] = static_cast<int64_t>(strtol(res, &res, kBase));
    }
    (*outputs)[id].SetShape(shape);
    id += 1;
    res = strtok_r(nullptr, delims, &save_ptr);
  }
  return kSuccess;
}

Status CustomInterface::GetCustomAttr(char *buf, uint32_t buf_size, const mindspore::schema::Custom *op,
                                      const std::string &attr_name) {
  MS_CHECK_TRUE_MSG(buf != nullptr, kLiteNullptr, "Buf is nullptr");
  MS_CHECK_TRUE_MSG(op != nullptr, kLiteNullptr, "Op is nullptr.");
  auto attr_ptr = op->attr();
  MS_CHECK_TRUE_MSG(attr_ptr != nullptr, kLiteNullptr, "Attr ptr is nullptr.");
  for (uint32_t i = 0; i < attr_ptr->size(); i++) {
    auto val = attr_ptr->Get(i);
    MS_CHECK_TRUE_MSG(val != nullptr, kLiteNullptr, "Attr val is nullptr.");
    MS_CHECK_TRUE_MSG(val->name() != nullptr, kLiteNullptr, "Attr val name is nullptr.");
    if (val->name()->str() == attr_name) {
      auto output_info = val->data();
      MS_CHECK_TRUE_MSG(output_info != nullptr, kLiteNullptr, "Output info is nullptr.");
      auto attr_size = output_info->size();
      if (attr_size >= buf_size) {
        MS_LOG(ERROR) << "Attr size[" << attr_size << "] is large than max size[" << buf_size << "]";
        return kLiteError;
      }
      for (uint32_t j = 0; j < attr_size; j++) {
        buf[j] = static_cast<char>(output_info->Get(j));
      }
      buf[attr_size] = 0;
      return kSuccess;
    }
  }
  return kLiteError;
}

std::shared_ptr<mindspore::kernel::KernelInterface> CustomInferCreater() {
  auto infer = new (std::nothrow) CustomInterface();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return std::shared_ptr<mindspore::kernel::KernelInterface>(infer);
}
}  // namespace acl
}  // namespace mindspore::kernel
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, ACL, acl::CustomInferCreater);
}  // namespace kernel
}  // namespace mindspore

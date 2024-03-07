/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/infer/forward_rasterize_infer.h"
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "common/log_adapter.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr uint32_t kNumOutputShape0 = 1;
constexpr uint32_t kNumOutputShape1 = 5;
}  // namespace
std::shared_ptr<KernelInterface> ForwardRasterizeInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<ForwardRasterizeInfer>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return infer;
}

Status ForwardRasterizeInfer::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                    const mindspore::schema::Primitive *primitive) {
  MS_LOG(INFO) << "custom infer shape.";
  if (inputs == nullptr || outputs == nullptr || primitive == nullptr) {
    MS_LOG(ERROR) << "infer failed.";
    return kLiteError;
  }
  if ((*outputs).size() < 1) {
    MS_LOG(ERROR) << "output tensor size is wrong, output size: " << (*outputs).size();
    return kLiteError;
  }

  auto param = primitive->value_as_Custom();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kLiteError;
  }
  uint32_t height = 0;
  uint32_t width = 0;
  if (param->attr() == nullptr) {
    MS_LOG(ERROR) << "param->attr() is nullptr";
    return kLiteError;
  }

  for (uint32_t i = 0; i < static_cast<uint32_t>(param->attr()->size()); i++) {
    if (param->attr()->Get(i) == nullptr || param->attr()->Get(i)->name() == nullptr) {
      MS_LOG(ERROR) << "param->attr()->Get(i) is nullptr or param->attr()->Get(i)->name() is nullptr";
      return kLiteError;
    }
    auto attr_name = param->attr()->Get(i)->name()->str();
    auto attr_val = param->attr()->Get(i)->data();
    std::string attr_val_str(attr_val->begin(), attr_val->end());
    if (attr_name == "h") {
      height = std::stoi(attr_val_str);
    } else if (attr_name == "w") {
      width = std::stoi(attr_val_str);
    }
  }

  auto &output = (*outputs)[0];
  output.SetShape({kNumOutputShape0, kNumOutputShape1, height, width});
  return kSuccess;
}
}  // namespace kernel
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, ForwardRasterize, ForwardRasterizeInferCreater);
}  // namespace kernel
}  // namespace mindspore

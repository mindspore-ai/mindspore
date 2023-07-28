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

#include "tools/converter/adapter/acl/infer/flash_attention_infer.h"
#include <memory>
#include "include/registry/register_kernel_interface.h"
#include "common/log_adapter.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kNumInputSize = 3;
}
std::shared_ptr<KernelInterface> FlashAttentionInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<FlashAttentionInfer>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "new custom infer is nullptr";
    return nullptr;
  }
  return infer;
}

Status FlashAttentionInfer::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                  const mindspore::schema::Primitive *primitive) {
  MS_LOG(INFO) << "custom infer shape.";
  if (inputs == nullptr || outputs == nullptr || primitive == nullptr) {
    MS_LOG(ERROR) << "infer failed.";
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return kLiteError;
  }
  if (param->type() == nullptr) {
    MS_LOG(ERROR) << "param type is nullptr.";
    return kLiteError;
  }
  if ((*inputs).size() < kNumInputSize || (*outputs).size() != 1) {
    MS_LOG(ERROR) << "input or output tensor size is wrong, input size: " << (*inputs).size()
                  << ", output size: " << (*outputs).size();
    return kLiteError;
  }
  const auto &input = (*inputs)[0];
  auto &output = (*outputs)[0];
  output.SetShape(input.Shape());
  MS_LOG(INFO) << "custom op output shape: " << output.Shape();
  return kSuccess;
}
}  // namespace kernel
}  // namespace mindspore
namespace mindspore {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, FlashAttention, FlashAttentionInferCreater);
}  // namespace kernel
}  // namespace mindspore

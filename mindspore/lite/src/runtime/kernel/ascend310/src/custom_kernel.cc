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

#include "src/runtime/kernel/ascend310/src/custom_kernel.h"
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "src/runtime/kernel/ascend310/src/model_infer.h"
#include "src/common/log_util.h"
#include "common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
CustomAscend310Kernel::CustomAscend310Kernel(const std::vector<mindspore::MSTensor> &inputs,
                                             const std::vector<mindspore::MSTensor> &outputs,
                                             const schema::Primitive *primitive, const mindspore::Context *ctx)
    : Kernel(inputs, outputs, primitive, ctx), load_model_(false), model_infer_(nullptr) {}

CustomAscend310Kernel::~CustomAscend310Kernel() {
  if (load_model_) {
    int ret = model_infer_->Finalize();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Model finalize failed.";
    }
  }
}

AclModelOptions CustomAscend310Kernel::GetAclModelOptions(const mindspore::Context *ctx) const {
  AclModelOptions options;
  options.device_id = 0;
  if (ctx == nullptr) {
    MS_LOG(WARNING) << "Context is nullptr.";
    return options;
  }
  auto context = const_cast<mindspore::Context *>(ctx);
  auto device_infos = context->MutableDeviceInfo();
  if (device_infos.size() < 1) {
    MS_LOG(WARNING) << "Size of device infos is less than one.";
    return options;
  }
  if (device_infos[0] == nullptr) {
    MS_LOG(WARNING) << "Device info is nullptr.";
    return options;
  }
  auto ascend31o_info = device_infos[0]->Cast<Ascend310DeviceInfo>();
  if (ascend31o_info == nullptr) {
    MS_LOG(WARNING) << "Ascend310 info is nullptr.";
    return options;
  }

  options.device_id = static_cast<int32_t>(ascend31o_info->GetDeviceID());
  return options;
}

STATUS CustomAscend310Kernel::PrepareModelInfer() {
  if (inputs_.size() < 1) {
    MS_LOG(ERROR) << "Inputs size should not less than 1.";
    return lite::RET_ERROR;
  }
  // last input is om data tensor
  int idx = inputs_.size() - 1;
  Buffer om_data(inputs_[idx].Data().get(), inputs_[idx].DataSize());
  if (model_infer_ == nullptr) {
    auto options = GetAclModelOptions(context_);
    model_infer_ = std::make_shared<ModelInfer>(om_data, options);
    CHECK_NULL_RETURN(model_infer_);
  }
  int ret = model_infer_->Init();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Model infer init failed.";
    return lite::RET_ERROR;
  }
  ret = model_infer_->Load();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Load om data failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Load om data success.";
  return lite::RET_OK;
}

STATUS CustomAscend310Kernel::Prepare() {
  if (load_model_) {
    MS_LOG(INFO) << "Custom kernel has been prepared.";
    return lite::RET_OK;
  }
  if (PrepareModelInfer() != lite::RET_OK) {
    MS_LOG(ERROR) << "Model infer prepare is not ok.";
    return lite::RET_ERROR;
  }
  load_model_ = true;
  return lite::RET_OK;
}

STATUS CustomAscend310Kernel::ReSize() {
  if (load_model_) {
    int ret = model_infer_->Finalize();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Model finalize failed.";
    }
    load_model_ = false;
  }
  return Prepare();
}

STATUS CustomAscend310Kernel::Execute() {
  if (!load_model_) {
    MS_LOG(WARNING) << "Custom kernel has not been prepared.";
    return lite::RET_OK;
  }
  std::vector<mindspore::MSTensor> inputs(inputs_.begin(), inputs_.end() - 1);
  if (model_infer_->Inference(inputs, &outputs_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

std::shared_ptr<kernel::Kernel> CustomCreateKernel(const std::vector<mindspore::MSTensor> &inputs,
                                                   const std::vector<mindspore::MSTensor> &outputs,
                                                   const schema::Primitive *primitive, const mindspore::Context *ctx) {
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr.";
    return nullptr;
  }
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return nullptr;
  }

  auto kernel = std::make_shared<CustomAscend310Kernel>(inputs, outputs, primitive, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New custom kernel is nullptr";
    return nullptr;
  }
  return kernel;
}
}  // namespace acl
}  // namespace mindspore::kernel
namespace mindspore {
namespace registry {
namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kInt8 = DataType::kNumberTypeInt8;
const auto kUInt8 = DataType::kNumberTypeUInt8;
}  // namespace
REGISTER_CUSTOM_KERNEL(ASCEND310, ACL, kFloat32, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND310, ACL, kInt8, ACL, kernel::acl::CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(ASCEND310, ACL, kUInt8, ACL, kernel::acl::CustomCreateKernel)
}  // namespace registry
}  // namespace mindspore

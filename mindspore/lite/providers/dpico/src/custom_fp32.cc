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

#include "src/custom_fp32.h"
#include <vector>
#include <memory>
#include "common/check_base.h"
#include "common/op_attr.h"
#include "include/api/data_type.h"
#include "include/registry/register_kernel.h"
#include "include/registry/register_kernel_interface.h"

namespace mindspore {
namespace lite {
bool CustomCPUKernel::InferShapeDone() const {
  if (std::any_of(inputs_.begin(), inputs_.end(), [](const MSTensor &input) {
        return input.DataType() == mindspore::DataType::kObjectTypeTensorType;
      })) {
    return false;
  }
  auto shape = outputs_.front().Shape();
  return !(std::find(shape.begin(), shape.end(), -1) != shape.end());
}
int CustomCPUKernel::PreProcess() {
  int ret;
  if (!InferShapeDone()) {
    auto custom_kernel_interface = registry::RegisterKernelInterface::GetKernelInterface("", primitive_, this);
    MS_CHECK_TRUE_MSG(custom_kernel_interface != nullptr, RET_ERROR, "get custom kernel interface failed.");
    auto status = custom_kernel_interface->Infer(&inputs_, &outputs_, primitive_, this);
    MS_CHECK_TRUE_MSG(status == kSuccess, RET_ERROR, "custom op infershape failed. " << this->name());
    ret = ReSize();
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "resize failed.");
  }

  // reallocate memory for output data
  for (auto output : outputs_) {
    auto data = output.MutableData();
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "output tensor data is nullptr. " << output.Name());
  }

  ret = acl_model_manager_->UpdateKernelConfig(this->GetConfig(kDpico));
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "acl model manager update kernel config failed.");
  ret = acl_model_manager_->UpdateAclInputs(&inputs_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "acl model manager update inputs failed.");
  ret = acl_model_manager_->UpdateAclOutputs(&outputs_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "acl model manager update outputs failed.");
  return RET_OK;
}
int CustomCPUKernel::Prepare() {
  CHECK_LESS_RETURN(inputs_.size(), kInputSize2);
  CHECK_LESS_RETURN(outputs_.size(), 1);
  if (acl_model_manager_ == nullptr) {
    acl_model_manager_ = std::make_shared<AclModelManager>();
    MS_CHECK_TRUE_MSG(acl_model_manager_ != nullptr, RET_ERROR, "acl_model_manager_ is nullptr");
  }

  // init model manager
  if (acl_model_manager_->Init(this->GetConfig(kDpico), this->GetConfig(kModelSharingSection), primitive_, inputs_,
                               outputs_) != RET_OK) {
    MS_LOG(ERROR) << "init acl model manager failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    MS_LOG(INFO) << "infershape isn't done, skip this procedure";
    return RET_OK;
  }
  return ReSize();
}

int CustomCPUKernel::ReSize() {
  if (acl_model_manager_->UpdateBatchSize(inputs_) != RET_OK) {
    MS_LOG(ERROR) << "update batch size for acl model manager failed.";
    return RET_ERROR;
  }
  if (acl_model_manager_->PrepareAclInputs(&inputs_) != RET_OK) {
    MS_LOG(ERROR) << "prepare inputs for acl model manager failed.";
    return RET_ERROR;
  }
  if (acl_model_manager_->PrepareAclOutputs(&outputs_) != RET_OK) {
    MS_LOG(ERROR) << "prepare outputs for acl model manager failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CustomCPUKernel::Execute() {
  auto ret = PreProcess();
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "kernel preprocess failed." + this->name());

  if (acl_model_manager_->Execute(inputs_, outputs_, this->GetConfig(kModelSharingSection)) != RET_OK) {
    MS_LOG(ERROR) << "acl model manager execute failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::shared_ptr<mindspore::kernel::Kernel> CustomCreateKernel(const std::vector<MSTensor> &inputs,
                                                              const std::vector<MSTensor> &outputs,
                                                              const mindspore::schema::Primitive *primitive,
                                                              const mindspore::Context *ctx) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, nullptr, "primitive is nullptr");
  MS_CHECK_TRUE_MSG(ctx != nullptr, nullptr, "ctx is nullptr");
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  MS_CHECK_TRUE_MSG(op != nullptr, nullptr, "op is nullptr");
  MS_CHECK_TRUE_MSG(op->attr() != nullptr, nullptr, "op attr is nullptr");
  MS_CHECK_TRUE_MSG(op->attr()->size() >= 1, nullptr, "there should be at least 1 attribute of Custom");
  auto kernel = std::make_shared<CustomCPUKernel>(inputs, outputs, primitive, ctx);
  MS_CHECK_TRUE_MSG(kernel != nullptr, nullptr, "new custom kernel is nullptr");
  return kernel;
}

namespace {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const auto kInt8 = DataType::kNumberTypeInt8;
const auto kUInt8 = DataType::kNumberTypeUInt8;
}  // namespace
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kFloat32, DPICO, CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kInt8, DPICO, CustomCreateKernel)
REGISTER_CUSTOM_KERNEL(CPU, DPICO, kUInt8, DPICO, CustomCreateKernel)
}  // namespace lite
}  // namespace mindspore

/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <map>
#include <string>
#include <memory>
#include "common/check_base.h"
#include "common/infer_util.h"
#include "common/op_attr.h"
#include "include/api/data_type.h"
#include "include/registry/register_kernel.h"
#include "include/registry/register_kernel_interface.h"

namespace mindspore {
namespace lite {
constexpr size_t kNumOfInputOm = 1;

bool CustomCPUKernel::InferShapeDone() const {
  if (std::any_of(inputs_.begin(), inputs_.end(), [](const MSTensor &input) {
        return input.DataType() == mindspore::DataType::kObjectTypeTensorType;
      })) {
    return false;
  }
  auto shape = outputs_.front().Shape();
  return !(std::find(shape.begin(), shape.end(), -1) != shape.end());
}

size_t DataTypeSize(mindspore::DataType data_type) {
  switch (data_type) {
    case mindspore::DataType::kNumberTypeFloat64:
      return sizeof(double);
    case mindspore::DataType::kNumberTypeFloat32:
      return sizeof(float);
    case mindspore::DataType::kNumberTypeInt8:
      return sizeof(int8_t);
    case mindspore::DataType::kNumberTypeUInt8:
      return sizeof(uint8_t);
    case mindspore::DataType::kNumberTypeFloat16:
    case mindspore::DataType::kNumberTypeInt16:
      return sizeof(int16_t);
    case mindspore::DataType::kNumberTypeInt32:
      return sizeof(int32_t);
    case mindspore::DataType::kNumberTypeInt64:
      return sizeof(int64_t);
    case mindspore::DataType::kNumberTypeUInt16:
      return sizeof(uint16_t);
    case mindspore::DataType::kNumberTypeUInt32:
      return sizeof(uint32_t);
    case mindspore::DataType::kNumberTypeUInt64:
      return sizeof(uint64_t);
    case mindspore::DataType::kNumberTypeBool:
      return sizeof(bool);
    case mindspore::DataType::kObjectTypeString:
      return sizeof(char);
    case mindspore::DataType::kObjectTypeTensorType:
      return 0;
    default:
      MS_LOG(ERROR) << "Not support the type: " << static_cast<int>(data_type);
      return 0;
  }
}

void CustomCPUKernel::UpdateInputOutputTensor() {
  if (custom_id_ == 0) {
    for (size_t i = 0; i < inputs_.size() - kNumOfInputOm; i++) {
      std::vector<int64_t> shape = inputs_[i].Shape();
      if (shape.empty()) {
        MS_LOG(ERROR) << "Input[" << i << "]`s shape is empty.";
        return;
      }
      size_t shape_len = shape.size();
      size_t data_type_size = DataTypeSize(inputs_[i].DataType());
      if (data_type_size == 0) {
        MS_LOG(ERROR) << "data type size is 0.";
        return;
      }
      int64_t last_dim_size = shape[shape_len - 1] * data_type_size;
      MS_LOG(DEBUG) << "Input last_dim = " << shape[shape_len - 1];
      if (last_dim_size % stride_align_size_ != 0) {
        last_dim_size = (last_dim_size + stride_align_size_ - 1) & (~(stride_align_size_ - 1));
        int64_t last_dim_new = last_dim_size / data_type_size;
        MS_LOG(DEBUG) << "Input new last_dim = " << last_dim_new;
        shape[shape_len - 1] = last_dim_new;
        inputs_[i].SetShape(shape);
      }
    }
  }
  if (custom_id_ == custom_num_ - 1) {
    for (size_t i = 0; i < outputs_.size(); i++) {
      auto shape = outputs_[i].Shape();
      if (shape.empty()) {
        MS_LOG(ERROR) << "Output[" << i << "]`s shape is empty.";
        return;
      }
      size_t shape_len = shape.size();
      size_t data_type_size = DataTypeSize(outputs_[i].DataType());
      if (data_type_size == 0) {
        MS_LOG(ERROR) << "data type size is 0.";
        return;
      }
      int64_t last_dim_size = shape[shape_len - 1] * data_type_size;
      MS_LOG(DEBUG) << "Output last_dim = " << shape[shape_len - 1];
      if (last_dim_size % stride_align_size_ != 0) {
        last_dim_size = (last_dim_size + stride_align_size_ - 1) & (~(stride_align_size_ - 1));
        int64_t last_dim_new = last_dim_size / data_type_size;
        MS_LOG(DEBUG) << "Output new last_dim = " << last_dim_new;
        shape[shape_len - 1] = last_dim_new;
        outputs_[i].SetShape(shape);
      }
    }
  }
}

int CustomCPUKernel::ParseAttrs() {
  std::map<std::string, std::string> attrs;
  std::string internal_stride;
  std::string custom_id;
  std::string custom_num;
  std::string head_tail_op_is_custom;

  if (FetchAttrs(*primitive_, &attrs) == kSuccess) {
    if (attrs.find("head_tail_op_is_custom") != attrs.end()) {
      head_tail_op_is_custom = attrs.at("head_tail_op_is_custom");
      if (head_tail_op_is_custom != "1") {
        MS_LOG(ERROR) << "When setting 'SupportZeroCopy=on', "
                      << "you must ensure that the first and last operators are custom operators.";
        return RET_ERROR;
      }
    } else {
      MS_LOG(ERROR) << "Get head_tail_op_is_custom failed.";
      return RET_ERROR;
    }
    if (attrs.find(kLastDimStride) != attrs.end()) {
      internal_stride = attrs.at(kLastDimStride);
    } else {
      MS_LOG(ERROR) << "Get internal_stride failed.";
      return RET_ERROR;
    }
    if (attrs.find("custom_id") != attrs.end()) {
      custom_id = attrs.at("custom_id");
    } else {
      MS_LOG(ERROR) << "Get custom_id failed.";
      return RET_ERROR;
    }
    if (attrs.find("custom_num") != attrs.end()) {
      custom_num = attrs.at("custom_num");
    } else {
      MS_LOG(ERROR) << "Get custom_num failed.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "Fetch attrs failed.";
    return RET_ERROR;
  }

  stride_align_size_ = static_cast<size_t>(stoi(internal_stride));
  custom_id_ = static_cast<size_t>(stoi(custom_id));
  custom_num_ = static_cast<size_t>(stoi(custom_num));
  return RET_OK;
}

int CustomCPUKernel::Prepare() {
  CHECK_LESS_RETURN(inputs_.size(), kInputSize2);
  CHECK_LESS_RETURN(outputs_.size(), 1);
  if (acl_model_manager_ == nullptr) {
    acl_model_manager_ = std::make_shared<AclModelManager>();
    MS_CHECK_TRUE_MSG(acl_model_manager_ != nullptr, RET_ERROR, "Acl_model_manager_ is nullptr");
  }

  std::map<std::string, std::string> config_info = this->GetConfig(kDpico);
  // init model manager
  if (acl_model_manager_->Init(config_info, this->GetConfig(kModelSharingSection), primitive_, inputs_, outputs_) !=
      RET_OK) {
    MS_LOG(ERROR) << "Init acl model manager failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    MS_LOG(INFO) << "Infershape isn't done, skip this procedure";
    return RET_OK;
  }

  if (config_info.find(kSupportZeroCopy) != config_info.end()) {
    if (config_info[kSupportZeroCopy] == "on") {
      support_zero_copy_ = "on";
      MS_LOG(INFO) << "Set 'SupportZeroCopy=on', will support zero copy.";
    } else {
      MS_LOG(INFO) << "Do not set 'SupportZeroCopy=on', will not support zero copy.";
    }
  } else {
    MS_LOG(INFO) << "Do not set 'SupportZeroCopy', will not support zero copy.";
  }

  if (support_zero_copy_ == "on") {
    if (ParseAttrs() != RET_OK) {
      MS_LOG(ERROR) << "Parse attrs failed or the attr is invalid.";
      return RET_ERROR;
    }
    UpdateInputOutputTensor();
  }
  if (acl_model_manager_->UpdateBatchSize(inputs_) != RET_OK) {
    MS_LOG(ERROR) << "Update batch size for acl model manager failed.";
    return RET_ERROR;
  }
  if (acl_model_manager_->PrepareAclInputs(&inputs_) != RET_OK) {
    MS_LOG(ERROR) << "Prepare inputs for acl model manager failed.";
    return RET_ERROR;
  }
  if (acl_model_manager_->PrepareAclOutputs(&outputs_) != RET_OK) {
    MS_LOG(ERROR) << "Prepare outputs for acl model manager failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CustomCPUKernel::ReSize() {
  if (!InferShapeDone()) {
    auto custom_kernel_interface = registry::RegisterKernelInterface::GetKernelInterface("", primitive_, this);
    MS_CHECK_TRUE_MSG(custom_kernel_interface != nullptr, RET_ERROR, "Get custom kernel interface failed.");
    auto status = custom_kernel_interface->Infer(&inputs_, &outputs_, primitive_, this);
    MS_CHECK_TRUE_MSG(status == kSuccess, RET_ERROR, "Custom op infershape failed. " << this->name());
  }
  if (support_zero_copy_ == "on") {
    UpdateInputOutputTensor();
  }
  if (acl_model_manager_->UpdateBatchSize(inputs_) != RET_OK) {
    MS_LOG(ERROR) << "Update batch size for acl model manager failed.";
    return RET_ERROR;
  }

  auto ret = acl_model_manager_->UpdateKernelConfig(this->GetConfig(kDpico));
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Acl model manager update kernel config failed.");
  ret = acl_model_manager_->UpdateAclInputs(&inputs_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Acl model manager update inputs failed.");
  ret = acl_model_manager_->UpdateAclOutputs(&outputs_);
  MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "Acl model manager update outputs failed.");
  return RET_OK;
}

int CustomCPUKernel::Execute() {
  // reallocate memory for output data
  for (auto output : outputs_) {
    auto data = output.MutableData();
    MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "Output tensor data is nullptr. " << output.Name());
  }
  if (acl_model_manager_->Execute(inputs_, outputs_, this->GetConfig(kModelSharingSection)) != RET_OK) {
    MS_LOG(ERROR) << "Acl model manager execute failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::shared_ptr<mindspore::kernel::Kernel> CustomCreateKernel(const std::vector<MSTensor> &inputs,
                                                              const std::vector<MSTensor> &outputs,
                                                              const mindspore::schema::Primitive *primitive,
                                                              const mindspore::Context *ctx) {
  MS_CHECK_TRUE_MSG(primitive != nullptr, nullptr, "Primitive is nullptr");
  MS_CHECK_TRUE_MSG(ctx != nullptr, nullptr, "Ctx is nullptr");
  if (primitive->value_type() != mindspore::schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom";
    return nullptr;
  }

  auto op = primitive->value_as_Custom();
  MS_CHECK_TRUE_MSG(op != nullptr, nullptr, "Op is nullptr");
  MS_CHECK_TRUE_MSG(op->attr() != nullptr, nullptr, "Op attr is nullptr");
  MS_CHECK_TRUE_MSG(op->attr()->size() >= 1, nullptr, "There should be at least 1 attribute of Custom");
  auto kernel = std::make_shared<CustomCPUKernel>(inputs, outputs, primitive, ctx);
  MS_CHECK_TRUE_MSG(kernel != nullptr, nullptr, "New custom kernel is nullptr");
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

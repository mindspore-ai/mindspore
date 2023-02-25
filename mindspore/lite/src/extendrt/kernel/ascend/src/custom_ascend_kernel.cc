/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/src/custom_ascend_kernel.h"
#include <utility>
#include <algorithm>
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "extendrt/kernel/ascend/model/model_infer.h"
#include "core/ops/custom.h"
#include "plugin/factory/ms_factory.h"
#include "src/common/log_util.h"
#include "src/common/common.h"
#include "common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
CustomAscendKernelMod::CustomAscendKernelMod()
    : load_model_(false), acl_options_(nullptr), model_infer_(nullptr), input_data_idx_(0) {}

CustomAscendKernelMod::~CustomAscendKernelMod() {
  if (load_model_) {
    if (!model_infer_->Finalize()) {
      MS_LOG(ERROR) << "Model finalize failed.";
    }
  }
}

void CustomAscendKernelMod::RecordInputDataIndex(const std::vector<KernelTensorPtr> &inputs) {
  for (size_t idx = 0; idx < inputs.size(); ++idx) {
    if (inputs[idx] == nullptr) {
      MS_LOG(ERROR) << "Input " << idx << " is invalid.";
      return;
    }
    if (inputs[idx]->GetData() == nullptr) {
      input_data_idx_ = idx;
      break;
    }
  }
}

AclModelOptionsPtr CustomAscendKernelMod::GenAclOptions(const BaseOperatorPtr &base_operator) {
  auto acl_options_ptr = std::make_shared<AclModelOptions>();
  if (acl_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Acl options make shared failed.";
    return nullptr;
  }
  auto custom_op = std::dynamic_pointer_cast<ops::Custom>(base_operator);
  if (custom_op == nullptr) {
    MS_LOG(ERROR) << "Cast Custom ops failed!";
    return nullptr;
  }
  auto prim = custom_op->GetPrim();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Get prim from custom op failed.";
    return nullptr;
  }
  auto profiling_path_val = prim->GetAttr(lite::kProfilingPathKey);
  if (profiling_path_val != nullptr) {
    auto val = GetValue<std::string>(profiling_path_val);
    acl_options_ptr->profiling_path = val;
  }
  auto dump_path_val = prim->GetAttr(lite::kDumpPathKey);
  if (dump_path_val != nullptr) {
    auto val = GetValue<std::string>(dump_path_val);
    acl_options_ptr->dump_path = val;
  }
  // set device id
  uint32_t device_count;
  if (aclrtGetDeviceCount(&device_count) != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Get device count failed, set default device id 0.";
    return acl_options_ptr;
  }
  if (device_id_ >= device_count) {
    MS_LOG(WARNING) << "Current device id " << device_id_ << " is larger than max count " << device_count
                    << ",please check the device info of context and set the default device id 0.";
    return acl_options_ptr;
  }
  acl_options_ptr->device_id = static_cast<int32_t>(device_id_);
  MS_LOG(INFO) << "Set device id " << device_id_;
  return acl_options_ptr;
}

bool CustomAscendKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (load_model_) {
    MS_LOG(INFO) << "Om has been loaded in custom kernel.";
    return true;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Custom kernel has empty inputs or outputs, which is invalid.";
    return false;
  }
  inputs_.assign(inputs.begin(), inputs.end() - 1);
  outputs_.assign(outputs.begin(), outputs.end());
  acl_options_ = GenAclOptions(base_operator);
  if (acl_options_ == nullptr) {
    MS_LOG(ERROR) << "Generate acl options failed.";
    return false;
  }
  auto &om_input = inputs.back();
  if (om_input == nullptr || om_input->GetData() == nullptr) {
    MS_LOG(ERROR) << "Om data input is invalid, inputs size " << inputs.size();
    return false;
  }
  auto om_data = om_input->GetData();
  model_infer_ = std::make_shared<ModelInfer>(acl_options_);
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "Create ModelInfer failed.";
    return false;
  }
  if (!model_infer_->Init()) {
    MS_LOG(ERROR) << "Model infer init failed.";
    return false;
  }
  if (!model_infer_->Load(om_data->addr, om_data->size)) {
    MS_LOG(ERROR) << "Load om data failed.";
    return false;
  }
  UpdateInputKernelTensorInfo();
  (void)RetrieveOutputShape();

  MS_LOG(INFO) << "Load om data success.";
  load_model_ = true;
  return true;
}

int CustomAscendKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (!load_model_) {
    MS_LOG(WARNING) << "Model has not been loaded, start to load when resize.";
    if (!Init(base_operator, inputs, outputs)) {
      MS_LOG(ERROR) << "Load model failed when resize.";
      return lite::RET_ERROR;
    }
  }
  if (inputs.size() < 1) {
    MS_LOG(ERROR) << "inputs size is less than one.";
    return lite::RET_ERROR;
  }
  if (!OnNewInputShapes(inputs)) {
    MS_LOG(ERROR) << "Failed to resize inputs";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

template <typename T, typename U>
static bool CheckInputNums(const std::vector<T> &update_info, const std::vector<U> &inputs, size_t input_weight = 0) {
  if (update_info.empty()) {
    MS_LOG(ERROR) << "check update info size empty";
    return false;
  }
  if (update_info.size() + input_weight != inputs.size()) {
    MS_LOG(ERROR) << "update info size and inputs size check failed. update info size: " << update_info.size()
                  << ". inputs' size: " << inputs.size() << ". input weight: " << input_weight;
    return false;
  }
  return true;
}

template <typename T, typename U>
static bool CheckOutputNums(const std::vector<T> &update_info, const std::vector<U> &outputs) {
  if (update_info.empty()) {
    MS_LOG(ERROR) << "check update info size empty";
    return false;
  }
  if (update_info.size() != outputs.size()) {
    MS_LOG(ERROR) << "update info size and outputs size check failed. update info size: " << update_info.size()
                  << ". outputs' size: " << outputs.size();
    return false;
  }
  return true;
}

// In DVPP, model input shape and data type get modified
void CustomAscendKernelMod::UpdateInputKernelTensorInfo() {
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "update input shape fail because model_infer_ is nullptr";
    return;
  }
  const std::vector<ShapeVector> shapes = model_infer_->GetInputShape();
  const std::vector<TypeId> types = model_infer_->GetInputDataType();
  const std::vector<Format> formats = model_infer_->GetInputFormat();
  MS_LOG(INFO) << "check input kernel tensor info nums";
  if (!CheckInputNums(shapes, inputs_) || !CheckInputNums(types, inputs_) || !CheckInputNums(formats, inputs_)) {
    return;
  }

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto &input = inputs_[i];
    input->SetShapeVector(shapes[i]);
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(types[i]), input->GetBaseShape());
    TensorInfo tensor_info{formats[i], new_abstract, input->GetDeviceShapeAdaptively()};
    input->SetTensorInfo(tensor_info);
  }
}

// In DVPP, model input data size gets modified, get updated inputs
std::vector<KernelTensorPtr> CustomAscendKernelMod::GetInputKernelTensor() {
  UpdateInputKernelTensorInfo();
  return inputs_;
}

bool CustomAscendKernelMod::ResetInputOutputShapes() {
  auto input_shapes = model_infer_->GetInputShape();
  if (input_shapes.size() != inputs_.size()) {
    MS_LOG(ERROR) << "The number of input shapes size " << input_shapes.size() << " != the number of inputs "
                  << inputs_.size();
    return false;
  }
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->SetShapeVector(input_shapes[i]);
  }
  auto output_shapes = model_infer_->GetOutputShape();
  if (output_shapes.size() != outputs_.size()) {
    MS_LOG(ERROR) << "The number of output shapes size " << output_shapes.size() << " != the number of outputs "
                  << outputs_.size();
    return false;
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputs_[i]->SetShapeVector(output_shapes[i]);
  }
  return true;
}

bool CustomAscendKernelMod::OnNewInputShapes(const std::vector<KernelTensorPtr> &new_inputs) {
  auto input_shapes = model_infer_->GetInputShape();
  if (input_shapes.size() != new_inputs.size()) {
    MS_LOG(ERROR) << "Invalid new input size " << new_inputs.size() << ", expect input size " << input_shapes.size();
    return false;
  }
  bool input_shape_changed = false;
  for (size_t i = 0; i < new_inputs.size(); i++) {
    auto new_shape = new_inputs[i]->GetShapeVector();
    if (input_shapes[i] != new_shape) {
      input_shape_changed = true;
    }
  }
  if (!input_shape_changed) {
    return true;
  }
  std::vector<ShapeVector> new_shapes;
  std::transform(new_inputs.begin(), new_inputs.end(), std::back_inserter(new_shapes),
                 [](auto &t) { return t->GetShapeVector(); });
  if (!model_infer_->Resize(new_shapes)) {
    MS_LOG(ERROR) << "Failed to Resize";
    return false;
  }
  return ResetInputOutputShapes();
}

bool CustomAscendKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &, void *) {
  if (!load_model_) {
    MS_LOG(ERROR) << "Init custom ascend kernel has been not ready.";
    return false;
  }
  if (!model_infer_->Inference(inputs_, outputs_)) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return false;
  }
  return true;
}

std::vector<KernelTensorPtr> CustomAscendKernelMod::RetrieveOutputShape() {
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "update input shape fail because model_infer_ is nullptr";
    return outputs_;
  }
  const std::vector<ShapeVector> shapes = model_infer_->GetOutputShape();
  const std::vector<TypeId> types = model_infer_->GetOutputDataType();
  const std::vector<Format> formats = model_infer_->GetOutputFormat();
  MS_LOG(INFO) << "check output kernel tensor info nums";
  if (!CheckOutputNums(shapes, outputs_) || !CheckOutputNums(types, outputs_) || !CheckOutputNums(formats, outputs_)) {
    return outputs_;
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto &output = outputs_[i];
    output->SetShapeVector(shapes[i]);
    auto new_abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(types[i]), output->GetBaseShape());
    TensorInfo tensor_info{formats[i], new_abstract, output->GetDeviceShapeAdaptively()};
    output->SetTensorInfo(tensor_info);
  }
  return outputs_;
}
}  // namespace acl
}  // namespace mindspore::kernel

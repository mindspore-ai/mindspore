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
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "extendrt/kernel/ascend/model/model_infer.h"
#include "core/ops/custom.h"
#include "plugin/factory/ms_factory.h"
#include "src/common/log_util.h"
#include "src/common/common.h"
#include "common/log_adapter.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

bool SaveOM(const void *model, size_t length, const std::string &file_path) { return true; }

namespace mindspore::kernel {
namespace acl {
CustomAscendKernelMod::CustomAscendKernelMod()
    : load_model_(false), acl_options_(nullptr), model_infer_(nullptr), input_data_idx_(0) {}

CustomAscendKernelMod::~CustomAscendKernelMod() {
  if (load_model_ || is_multi_model_sharing_mem_prepare_) {
    if (!model_infer_->Finalize()) {
      MS_LOG(ERROR) << "Model finalize failed.";
    }
  }
}

bool CustomAscendKernelMod::Finalize() { return AclEnvGuard::Finalize(); }

void CustomAscendKernelMod::RecordInputDataIndex(const std::vector<KernelTensor *> &inputs) {
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

AclModelOptionsPtr CustomAscendKernelMod::GenAclOptions() {
  auto acl_options_ptr = std::make_shared<AclModelOptions>();
  if (acl_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Acl options make shared failed.";
    return nullptr;
  }
  auto profiling_path_val = primitive_->GetAttr(lite::kProfilingPathKey);
  if (profiling_path_val != nullptr) {
    auto val = GetValue<std::string>(profiling_path_val);
    acl_options_ptr->profiling_path = val;
  }
  auto dump_path_val = primitive_->GetAttr(lite::kDumpPathKey);
  if (dump_path_val != nullptr) {
    auto val = GetValue<std::string>(dump_path_val);
    acl_options_ptr->dump_path = val;
  }
  auto inner_calc_workspace_size = primitive_->GetAttr(lite::kInnerCalcWorkspaceSize);
  if (inner_calc_workspace_size != nullptr) {
    auto val = GetValue<bool>(inner_calc_workspace_size);
    acl_options_ptr->multi_model_sharing_mem_prepare = val;
    is_multi_model_sharing_mem_prepare_ = true;
  }
  auto inner_sharing_workspace = primitive_->GetAttr(lite::kInnerSharingWorkspace);
  if (inner_sharing_workspace != nullptr) {
    auto val = GetValue<bool>(inner_sharing_workspace);
    acl_options_ptr->multi_model_sharing_mem = val;
  }
  auto inner_model_path = primitive_->GetAttr(lite::kInnerModelPath);
  if (inner_model_path != nullptr) {
    auto val = GetValue<std::string>(inner_model_path);
    acl_options_ptr->model_path = val;
  }
  auto workspace_key = primitive_->GetAttr(lite::kInnerWorkspace);
  if (workspace_key != nullptr) {
    auto val = GetValue<bool>(workspace_key);
    acl_options_ptr->share_workspace = val;
  }
  auto weightspace_key = primitive_->GetAttr(lite::kInnerWeightspace);
  if (weightspace_key != nullptr) {
    auto val = GetValue<bool>(weightspace_key);
    acl_options_ptr->share_weightspace = val;
  }
  auto weightspace_workspace_key = primitive_->GetAttr(lite::kInnerWeightspaceWorkspace);
  if (weightspace_workspace_key != nullptr) {
    auto val = GetValue<bool>(weightspace_workspace_key);
    acl_options_ptr->share_weightspace_workspace = val;
  }
  // set device id
  uint32_t device_count;
  if (CALL_ASCEND_API(aclrtGetDeviceCount, &device_count) != ACL_ERROR_NONE) {
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

bool CustomAscendKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  inputs_ = inputs;
  outputs_ = outputs;
  if (load_model_) {
    MS_LOG(INFO) << "Om has been loaded in custom kernel.";
    return true;
  }
  // last input is as specific usage
  inputs_.resize(inputs.size() - 1);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Custom kernel has empty inputs or outputs, which is invalid.";
    return false;
  }
  acl_options_ = GenAclOptions();
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

  SaveOM(om_data->addr, om_data->size, "./");

  if (is_multi_model_sharing_mem_prepare_) {
    MS_LOG(INFO) << "is multi model sharing mem prepare.";
    return true;
  }
  UpdateInputKernelTensorInfo();
  UpdateOutputKernelTensorInfo();
  MS_LOG(INFO) << "Load om data success.";
  load_model_ = true;
  AclEnvGuard::AddModel(model_infer_);
  return true;
}

int CustomAscendKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (!load_model_) {
    MS_LOG(ERROR) << "Load model failed when resize.";
    return lite::RET_ERROR;
  }

  if (KernelMod::Resize(inputs, outputs) != KRET_OK) {
    MS_LOG(WARNING) << "Invalid inputs or output shapes.";
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

bool CustomAscendKernelMod::OnNewInputShapes(const std::vector<KernelTensor *> &new_inputs) {
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
  UpdateInputKernelTensorInfo();
  UpdateOutputKernelTensorInfo();
  return true;
}

bool CustomAscendKernelMod::Launch(const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &, void *stream_ptr) {
  if (!load_model_) {
    MS_LOG(ERROR) << "Init custom ascend kernel has been not ready.";
    return false;
  }
  UpdateOutputKernelTensorInfo();
  if (!model_infer_->Inference(inputs_, outputs_)) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return false;
  }
  return true;
}

void CustomAscendKernelMod::UpdateOutputKernelTensorInfo() {
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "update input shape fail because model_infer_ is nullptr";
    return;
  }
  const std::vector<ShapeVector> shapes = model_infer_->GetOutputShape();
  const std::vector<TypeId> types = model_infer_->GetOutputDataType();
  const std::vector<Format> formats = model_infer_->GetOutputFormat();
  if (!CheckOutputNums(shapes, outputs_) || !CheckOutputNums(types, outputs_) || !CheckOutputNums(formats, outputs_)) {
    return;
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    auto &output = outputs_[i];
    output->SetType(std::make_shared<TensorType>(TypeIdToType(types[i])));
    output->SetShape(std::make_shared<abstract::TensorShape>(shapes[i]));
    output->set_format(formats[i]);
  }
  return;
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
  if (!CheckInputNums(shapes, inputs_) || !CheckInputNums(types, inputs_) || !CheckInputNums(formats, inputs_)) {
    return;
  }

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto &input = inputs_[i];
    input->SetType(std::make_shared<TensorType>(TypeIdToType(types[i])));
    input->SetShape(std::make_shared<abstract::TensorShape>(shapes[i]));
    input->set_format(formats[i]);
  }
}
}  // namespace acl
}  // namespace mindspore::kernel

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
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "extendrt/kernel/ascend/model/model_infer.h"
#include "core/ops/custom.h"
#include "plugin/factory/ms_factory.h"
#include "src/common/log_util.h"
#include "common/log_adapter.h"

namespace mindspore::kernel {
namespace acl {
CustomAscendKernelMod::CustomAscendKernelMod()
    : load_model_(false), acl_options_(nullptr), dyn_shape_proc_(nullptr), model_infer_(nullptr), input_data_idx_(0) {}

CustomAscendKernelMod::~CustomAscendKernelMod() {
  if (load_model_) {
    int ret = model_infer_->Finalize();
    if (ret != lite::RET_OK) {
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

void CustomAscendKernelMod::SetDeviceId() {
  if (acl_options_ == nullptr) {
    MS_LOG(ERROR) << "Acl options is nullptr.";
    return;
  }
  uint32_t device_count;
  if (aclrtGetDeviceCount(&device_count) != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Get device count failed, set default device id 0.";
    return;
  }
  if (device_id_ >= device_count) {
    MS_LOG(WARNING) << "Current device id " << device_id_ << " is larger than max count " << device_count
                    << ",please check the device info of context and set the default device id 0.";
    return;
  }
  acl_options_->device_id = static_cast<int32_t>(device_id_);
  MS_LOG(INFO) << "Set device id " << device_id_;
}

bool CustomAscendKernelMod::InitParam(const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Custom kernel has empty inputs or outputs, which is invalid.";
    return false;
  }
  inputs_.assign(inputs.begin(), inputs.end() - 1);
  outputs_.assign(outputs.begin(), outputs.end());
  acl_options_ = std::make_shared<AclModelOptions>();
  SetDeviceId();
  int idx = inputs.size() - 1;
  if (inputs[idx] == nullptr || inputs[idx]->GetData() == nullptr) {
    MS_LOG(ERROR) << "Input " << idx << " is invalid.";
    return false;
  }
  // buffer deep copy
  Buffer om_data(inputs[idx]->GetData()->addr, inputs[idx]->GetData()->size);
  model_infer_ = std::make_shared<ModelInfer>(om_data, acl_options_);
  if (model_infer_ == nullptr) {
    MS_LOG(ERROR) << "Create ModelInfer failed.";
    return false;
  }
  RecordInputDataIndex(inputs);
  dyn_shape_proc_ = std::make_shared<DynShapeProcess>(acl_options_, input_data_idx_);
  if (dyn_shape_proc_ == nullptr) {
    MS_LOG(ERROR) << "Create DynShapeProcess failed.";
    return false;
  }
  if (inputs[idx]->GetData()->addr != nullptr) {
    free(inputs[idx]->GetData()->addr);
    inputs[idx]->GetData()->addr = nullptr;
    inputs[idx]->GetData()->size = 0;
  }
  return true;
}

bool CustomAscendKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (load_model_) {
    MS_LOG(INFO) << "Om has been loaded in custom kernel.";
    return lite::RET_OK;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Custom>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "Cast Custom ops failed!";
    return false;
  }
  if (!InitParam(inputs, outputs)) {
    MS_LOG(ERROR) << "Init param failed.";
    return false;
  }
  if (LoadModel() != lite::RET_OK) {
    MS_LOG(ERROR) << "Load model failed.";
    return false;
  }

  load_model_ = true;
  return true;
}

int CustomAscendKernelMod::LoadModel() {
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
  acl_options_->batch_size = model_infer_->GetDynamicBatch();
  acl_options_->image_size = model_infer_->GetDynamicImage();
  acl_options_->input_format = model_infer_->GetInputFormat();

  MS_LOG(INFO) << "Load om data success.";
  return lite::RET_OK;
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
  original_data_ = inputs_;
  inputs_.assign(inputs.begin(), inputs.end() - 1);
  return lite::RET_OK;
}

int CustomAscendKernelMod::SetInputAndOutputAddr(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  if ((inputs_.size() + 1) != inputs.size()) {
    MS_LOG(ERROR) << "Size of inputs in init [" << (inputs_.size() + 1) << "] and "
                  << "size of inputs in launch [" << inputs.size() << "] are not equal.";
    return lite::RET_ERROR;
  }
  if (outputs_.size() != outputs.size()) {
    MS_LOG(ERROR) << "Size of outputs in init (" << outputs_.size() << ") and "
                  << "size of outputs in launch (" << outputs.size() << ") are not equal.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs[i] == nullptr || inputs_[i] == nullptr) {
      MS_LOG(ERROR) << "Input " << i << " is nullptr.";
      return lite::RET_ERROR;
    }
    if (inputs[i]->addr == nullptr || inputs[i]->size == 0) {
      MS_LOG(ERROR) << "Input " << i << " addr is invalid.";
      return lite::RET_ERROR;
    }
    inputs_[i]->SetData(inputs[i]);
  }
  for (size_t i = 0; i < outputs_.size(); ++i) {
    if (outputs[i] == nullptr || outputs_[i] == nullptr) {
      MS_LOG(ERROR) << "Output " << i << " is nullptr.";
      return lite::RET_ERROR;
    }
    outputs_[i]->SetData(outputs[i]);
  }
  return lite::RET_OK;
}

bool CustomAscendKernelMod::IsDynamicInput() {
  if (acl_options_->batch_size.empty() && acl_options_->image_size.empty()) {
    MS_LOG(INFO) << "Inputs are not dynamic mode.";
    return false;
  }
  return true;
}

void CustomAscendKernelMod::UpdateOutputAddr(const std::vector<AddressPtr> &outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    if ((outputs[i]->addr != outputs_[i]->GetData()->addr) || (outputs[i]->size != outputs_[i]->GetData()->size)) {
      outputs[i]->addr = outputs_[i]->GetData()->addr;
      outputs[i]->size = outputs_[i]->GetData()->size;
    }
  }
}

bool CustomAscendKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (!load_model_) {
    MS_LOG(ERROR) << "Init custom ascend kernel has been not ready.";
    return false;
  }
  if (SetInputAndOutputAddr(inputs, outputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "Check input and output param failed.";
    return false;
  }
  if (IsDynamicInput()) {
    if (dyn_shape_proc_->ProcDynamicInput(&original_data_, &inputs_) != lite::RET_OK) {
      MS_LOG(ERROR) << "Proc dynamic batch size input failed.";
      return false;
    }
  }
  if (model_infer_->Inference(inputs_, outputs_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return false;
  }
  if (IsDynamicInput()) {
    dyn_shape_proc_->DestroyDynamicInput(&inputs_);
  }
  UpdateOutputAddr(outputs);
  return true;
}
}  // namespace acl
}  // namespace mindspore::kernel

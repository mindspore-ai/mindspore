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
#include "include/registry/register_kernel.h"
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "extendrt/kernel/ascend/model/model_infer.h"
#include "extendrt/kernel/ascend/options/acl_options_parser.h"
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

bool CustomAscendKernelMod::InitParam(const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "Custom kernel has empty inputs or outputs, which is invalid.";
    return false;
  }
  inputs_.assign(inputs.begin(), inputs.end() - 1);
  outputs_.assign(outputs.begin(), outputs.end());
  acl_options_ = std::make_shared<AclModelOptions>();
  if (acl_options_ == nullptr) {
    MS_LOG(ERROR) << "Create AclModelOptions failed.";
    return false;
  }
  //  AclOptionsParser parser;
  //  if (parser.ParseAclOptions(context_, &acl_options_) != lite::RET_OK) {
  //    MS_LOG(ERROR) << "Parse model options failed.";
  //    return false;
  //  }
  // last input is om data tensor
  int idx = inputs.size() - 1;
  if (inputs[idx] == nullptr || inputs[idx]->GetData() == nullptr) {
    MS_LOG(ERROR) << "Input " << idx << " is invalid.";
    return false;
  }
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
    if (inputs[i]->addr == nullptr || inputs[i]->size == 0) {
      MS_LOG(ERROR) << "Input " << i << " addr is invalid.";
      return lite::RET_ERROR;
    }
    inputs_[i]->SetData(inputs[i]);
  }
  for (size_t j = 0; j < outputs_.size(); ++j) {
    if (outputs[j]->addr == nullptr || inputs[j]->size == 0) {
      MS_LOG(ERROR) << "Output " << j << " addr is invalid.";
      return lite::RET_ERROR;
    }
    outputs_[j]->SetData(outputs[j]);
  }
  return lite::RET_OK;
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
  if (dyn_shape_proc_->ProcDynamicInput(&inputs_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Proc dynamic batch size input failed.";
    return false;
  }
  if (model_infer_->Inference(inputs_, outputs_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Custom kernel execute failed.";
    return false;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(KernelMod, CustomAscend, CustomAscendKernelMod);
}  // namespace acl
}  // namespace mindspore::kernel

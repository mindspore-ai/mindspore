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

#include "src/litert/cxx_api/kernel_executor/custom_om_kernel.h"
#include <memory>
#include <algorithm>
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"
#include "compatible/hiai_base_types_cpt.h"
#include "util/base_types.h"
#include "securec/include/securec.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace kernel {
const auto kFloat32 = DataType::kNumberTypeFloat32;
const int MODEL_MAX_RUN_TIME_MS = 100;

int CustomOMKernel::Prepare() {
  if (inputs_.empty()) {
    MS_LOG(ERROR) << "inputs_ is null.";
    return kLiteNullptr;
  }
  auto ret = Build(inputs_.back().MutableData(), inputs_.back().DataSize());
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Build OMModelBuffer failed.";
    return kLiteError;
  }
  return kSuccess;
}

int CustomOMKernel::Execute() {
  auto ret = ConvertMSTensorToHiaiTensor();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ConvertMSTensorToHiaiTensor failed.";
    return kLiteError;
  }
  ret = Predict();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Custom om kernel execute failed.";
    return kLiteError;
  }
  ret = ConvertHiaiTensorToMSTensor();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ConvertHiaiTensorToMSTensor failed.";
    return kLiteError;
  }
  return kSuccess;
}

int CustomOMKernel::ConvertMSTensorToHiaiTensor() {
  if (inputs_.size() == 1) {
    MS_LOG(INFO) << "Don't set inputTensors";
    return kSuccess;
  }
  if (hiai_inputs_.size() != inputs_.size() - 1) {
    MS_LOG(ERROR) << "inputs_ and hiai_inputs_ have different size.";
    return kLiteError;
  }
  for (size_t i = 0; i < hiai_inputs_.size(); i++) {
    if (hiai_inputs_.at(i)->GetSize() != inputs_.at(i).DataSize()) {
      MS_LOG(ERROR) << "ms_input and hiai_input have different dataSize.";
      return kLiteError;
    }
    auto ret = memcpy_s(hiai_inputs_.at(i)->GetData(), hiai_inputs_.at(i)->GetSize(), inputs_.at(i).Data().get(),
                        inputs_.at(i).DataSize());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno=" << ret;
      return kLiteError;
    }
  }
  return kSuccess;
}

int CustomOMKernel::ConvertHiaiTensorToMSTensor() {
  for (size_t i = 0; i < hiai_outputs_.size(); i++) {
    if (hiai_outputs_.at(i)->GetSize() != outputs_.at(i).DataSize()) {
      MS_LOG(ERROR) << "ms_output and hiai_output have different dataSize.";
      return kLiteError;
    }
    auto ret = memcpy_s(outputs_.at(i).MutableData(), outputs_.at(i).DataSize(), hiai_outputs_.at(i)->GetData(),
                        hiai_outputs_.at(i)->GetSize());
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno=" << ret;
      return kLiteError;
    }
  }
  return kSuccess;
}

int CustomOMKernel::Build(void *modelData, size_t modelDataLength) {
  builtModel_ = hiai::CreateBuiltModel();
  if (builtModel_ == nullptr) {
    MS_LOG(ERROR) << "Create BuiltModel failed.";
    return kLiteNullptr;
  }
  builtModel_->SetName(modelName_);
  std::shared_ptr<hiai::IBuffer> modelBuffer =
    hiai::CreateLocalBuffer(static_cast<void *>(modelData), modelDataLength, false);
  if (modelBuffer == nullptr) {
    MS_LOG(ERROR) << "Create LocalBuffer failed.";
    return kLiteNullptr;
  }
  hiai::AIStatus ret = builtModel_->RestoreFromBuffer(modelBuffer);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "RestoreFromBuffer failed.";
    return kLiteError;
  }
  bool compatible = false;
  ret = builtModel_->CheckCompatibility(compatible);
  if (!compatible || ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "CheckCompatibility failed.";
    return kLiteError;
  }
  manager_ = hiai::CreateModelManager();
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "Create manager failed.";
    return kLiteNullptr;
  }
  hiai::ModelInitOptions initOptions;
  initOptions.perfMode = hiai::PerfMode::HIGH;
  ret = manager_->Init(initOptions, builtModel_, nullptr);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Init manager failed. ret=" << ret;
    return kLiteError;
  }
  ret = InitIOTensors();
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "InitIOTensors failed.";
    return kLiteError;
  }
  return kSuccess;
}

int CustomOMKernel::Predict() {
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "manager_ is null";
    return kLiteNullptr;
  }
  hiai::AIStatus ret = manager_->Run(hiai_inputs_, hiai_outputs_);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Predict failed. ret=" << ret;
    return kLiteError;
  }
  return kSuccess;
}

int CustomOMKernel::InitIOTensors() {
  std::vector<hiai::NDTensorDesc> inputTensorDescs = builtModel_->GetInputTensorDescs();
  std::vector<hiai::NDTensorDesc> outputTensorDescs = builtModel_->GetOutputTensorDescs();
  for (auto const &inputTensorDesc : inputTensorDescs) {
    std::shared_ptr<hiai::INDTensorBuffer> inputTensorBuffer = hiai::CreateNDTensorBuffer(inputTensorDesc);
    if (inputTensorBuffer == nullptr) {
      MS_LOG(ERROR) << "Create input data buffer failed.";
      return kLiteNullptr;
    }
    hiai_inputs_.push_back(inputTensorBuffer);
  }
  for (size_t i = 0; i < outputTensorDescs.size(); i++) {
    auto outputTensorDesc = outputTensorDescs.at(i);
    outputTensorDesc.dataType = hiai::DataType::FLOAT32;
    std::shared_ptr<hiai::INDTensorBuffer> outputTensorBuffer = hiai::CreateNDTensorBuffer(outputTensorDesc);
    if (outputTensorBuffer == nullptr) {
      MS_LOG(ERROR) << "Create output data buffer failed.";
      return kLiteNullptr;
    }
    hiai_outputs_.push_back(outputTensorBuffer);
    std::vector<int64_t> shapes;
    std::transform(outputTensorDesc.dims.begin(), outputTensorDesc.dims.end(), std::back_inserter(shapes),
                   [](const int32_t t) { return static_cast<int64_t>(t); });
    outputs_.at(i).SetDataType(kFloat32);
    outputs_.at(i).SetShape(shapes);
  }
  return kSuccess;
}

std::shared_ptr<Kernel> CustomOMKernelCreator(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                              const schema::Primitive *primitive, const mindspore::Context *ctx) {
  return std::make_shared<CustomOMKernel>(inputs, outputs, primitive, ctx);
}
REGISTER_CUSTOM_KERNEL(NPU, Tutorial, kFloat32, Custom_OM, CustomOMKernelCreator)
REGISTER_CUSTOM_KERNEL(CPU, Tutorial, kFloat32, Custom_OM, CustomOMKernelCreator)
}  // namespace kernel
}  // namespace mindspore

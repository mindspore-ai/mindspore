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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_CUSTOM_OM_KERNEL_H
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_CUSTOM_OM_KERNEL_H
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "tensor/nd_tensor_buffer.h"
#include "include/registry/register_kernel.h"
#include "model_manager/model_manager.h"
#include "model/built_model.h"

namespace mindspore {
namespace kernel {
class CustomOMKernel : public Kernel {
 public:
  CustomOMKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                 const schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}

  ~CustomOMKernel() = default;

  int Prepare() override;

  int Execute() override;

  int ReSize() override { return kSuccess; }

 private:
  int ConvertMSTensorToHiaiTensor();
  int ConvertHiaiTensorToMSTensor();
  int Build(void *modelData, size_t modelDataLength);
  int Predict();
  int InitIOTensors();
  std::string modelName_{"OmModel"};
  std::shared_ptr<hiai::IModelManager> manager_{nullptr};
  std::shared_ptr<hiai::IBuiltModel> builtModel_{nullptr};
  std::vector<std::shared_ptr<hiai::INDTensorBuffer>> hiai_inputs_{};
  std::vector<std::shared_ptr<hiai::INDTensorBuffer>> hiai_outputs_{};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_KERNEL_EXECUTOR_CUSTOM_OM_KERNEL_H

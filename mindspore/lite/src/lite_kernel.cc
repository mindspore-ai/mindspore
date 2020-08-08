/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/lite_kernel.h"
#include <algorithm>
#include "src/common/utils.h"

namespace mindspore::kernel {
void LiteKernel::InitOutTensorRefCount() {
  for (auto *tensor : this->outputs_) {
    tensor->SetRefCount(this->out_kernel_.size());
  }
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->outputs_) {
    tensor->decRefCount();
    if (0 >= tensor->RefCount()) {
      auto ret = tensor->FreeData();
      if (0 != ret) {
        MS_LOG(ERROR) << "Free tensor data failed";
        return ret;
      }
    }
  }
  return 0;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphInputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> input_kernels;
  for (const auto kernel : kernels) {
    for (auto input : kernel->GetInKernels()) {
      auto iter = std::find(kernels.begin(), kernels.end(), input);
      if (iter == kernels.end()) {
        input_kernels.emplace_back(input);
      }
    }
  }
  return input_kernels;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphOutputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> output_kernels;
  for (const auto kernel : kernels) {
    for (const auto output : kernel->GetOutKernels()) {
      auto iter = std::find(kernels.begin(), kernels.end(), output);
      if (iter == kernels.end()) {
        output_kernels.emplace_back(output);
      }
    }
  }
  return output_kernels;
}

std::vector<lite::tensor::Tensor *> LiteKernelUtil::SubgraphInputTensors(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::tensor::Tensor *> input_tensors;
  std::vector<lite::tensor::Tensor *> all_output_tensors;
  for (const auto &kernel : kernels) {
    all_output_tensors.insert(all_output_tensors.end(), kernel->GetOutputs().begin(), kernel->GetOutputs().end());
  }
  std::vector<kernel::LiteKernel *> input_kernels = SubgraphInputKernels(kernels);
  for (const auto &kernel : input_kernels) {
    for (const auto &tensor : kernel->GetInputs()) {
      auto iter = std::find(all_output_tensors.begin(), all_output_tensors.end(), tensor);
      if (iter == all_output_tensors.end() && tensor->Data() == nullptr) {
        input_tensors.emplace_back(tensor);
      }
    }
  }
  return input_tensors;
}

std::vector<lite::tensor::Tensor *> LiteKernelUtil::SubgraphOutputTensors(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::tensor::Tensor *> output_tensors;
  std::vector<lite::tensor::Tensor *> all_input_tensors;
  for (const auto &kernel : kernels) {
    all_input_tensors.insert(all_input_tensors.end(), kernel->GetInputs().begin(), kernel->GetInputs().end());
  }
  std::vector<kernel::LiteKernel *> output_kernels = SubgraphOutputKernels(kernels);
  for (const auto &kernel : output_kernels) {
    for (const auto &tensor : kernel->GetOutputs()) {
      auto iter = std::find(all_input_tensors.begin(), all_input_tensors.end(), tensor);
      if (iter == all_input_tensors.end()) {
        output_tensors.emplace_back(tensor);
      }
    }
  }
  return output_tensors;
}

void LiteKernelUtil::TopologicalSortKernels(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    for (auto *search_kernel : kernels) {
      if (search_kernel == kernel) {
        continue;
      }
      for (auto *tensor : kernel->GetInputs()) {
        if (lite::IsContain(search_kernel->GetOutputs(), tensor)) {
          kernel->AddInKernel(search_kernel);
        }
      }
      for (auto *tensor : kernel->GetOutputs()) {
        if (lite::IsContain(search_kernel->GetInputs(), tensor)) {
          kernel->AddOutKernel(search_kernel);
        }
      }
    }
  }
}

void LiteKernelUtil::InitTensorRefCount(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    kernel->InitOutTensorRefCount();
  }
}

int LiteKernelUtil::SetInput(LiteKernel &kernelMod, std::vector<lite::tensor::Tensor *> inputs) { return -1; }
}  // namespace mindspore::kernel

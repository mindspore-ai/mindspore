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
#include "src/tensor.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

void *LiteKernel::workspace_ = nullptr;

void LiteKernel::AllocWorkspace(size_t size) {
  if (size == 0) return;
  workspace_ = malloc(size);
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "fail to alloc " << size;
  }
}

void LiteKernel::FreeWorkspace() {
  free(workspace_);
  workspace_ = nullptr;
}

void LiteKernel::InitOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->SetRefCount(this->out_kernels_.size());
  }
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
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

int LiteKernel::FreeWorkTensor() const {
  for (auto input_kernel : this->in_kernels()) {
    MS_ASSERT(input_kernel != nullptr);
    if (input_kernel->is_model_output()) {
      continue;
    }
    auto ret = input_kernel->DecOutTensorRefCount();
    if (0 != ret) {
      MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << this->name() << " failed";
    }
  }
  return RET_OK;
}

int LiteKernel::PreProcess() {
  if (!InferShapeDone()) {
    (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(true);
    auto ret = (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->InferShape(in_tensors_, out_tensors_);
    if (ret != 0) {
      (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(false);
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  auto outputs = this->out_tensors();
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    output->MallocData();
  }
  return RET_OK;
}

int LiteKernel::Run(const KernelCallBack &before, const KernelCallBack &after) {
  if (before != nullptr) {
    if (!before(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
                {this->name_, this->type_str()})) {
      MS_LOG(WARNING) << "run kernel before_callback failed, name: " << this->name_;
    }
  }
  auto ret = Run();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "run kernel failed, name: " << this->name_;
    return ret;
  }
  if (after != nullptr) {
    if (!after(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
               {this->name_, this->type_str()})) {
      MS_LOG(ERROR) << "run kernel after_callback failed, name: " << this->name_;
    }
  }
  return RET_OK;
}

std::string LiteKernel::ToString() const {
  std::ostringstream oss;
  oss << "LiteKernel: " << this->name_;
  oss << ", Type: " << this->type_str();
  oss << std::endl << this->in_tensors_.size() << " InputTensors:";
  for (auto tensor : in_tensors_) {
    oss << " " << tensor << ":" << tensor->ToString();
  }
  oss << std::endl << this->out_tensors_.size() << " OutputTensors:";
  for (auto tensor : out_tensors_) {
    oss << " " << tensor << ":" << tensor->ToString();
  }
  oss << std::endl << this->in_kernels_.size() << " InputKernels:";
  for (auto in_kernel : in_kernels_) {
    oss << " " << in_kernel->name_;
  }
  oss << std::endl << this->out_kernels_.size() << " OutputKernels:";
  for (auto out_kernel : out_kernels_) {
    oss << " " << out_kernel->name_;
  }
  return oss.str();
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphInputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> input_kernels;
  for (const auto &kernel : kernels) {
    if (kernel->in_kernels().empty() && !kernel->in_tensors().empty()) {
      input_kernels.emplace_back(kernel);
      continue;
    }
    for (const auto &input : kernel->in_kernels()) {
      auto in_kernel_in_graph = std::find(kernels.begin(), kernels.end(), input);
      auto in_kernel_in_ret = std::find(input_kernels.begin(), input_kernels.end(), kernel);
      if (in_kernel_in_graph == kernels.end() && in_kernel_in_ret == input_kernels.end()) {
        input_kernels.emplace_back(kernel);
        break;
      }
    }
  }
  return input_kernels;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphOutputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> output_kernels;
  for (const auto &kernel : kernels) {
    if (kernel->out_kernels().empty() && !kernel->out_tensors().empty()) {
      output_kernels.emplace_back(kernel);
      continue;
    }
    for (const auto &output : kernel->out_kernels()) {
      auto out_kernel_in_graph = std::find(kernels.begin(), kernels.end(), output);
      auto out_kernel_in_ret = std::find(output_kernels.begin(), output_kernels.end(), kernel);
      if (out_kernel_in_graph == kernels.end() && out_kernel_in_ret == output_kernels.end()) {
        output_kernels.emplace_back(kernel);
        break;
      }
    }
  }
  return output_kernels;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> input_tensors;
  std::vector<lite::Tensor *> all_output_tensors;
  for (const auto &kernel : kernels) {
    auto kernel_out_tensors = kernel->out_tensors();
    all_output_tensors.insert(all_output_tensors.end(), kernel_out_tensors.begin(), kernel_out_tensors.end());
  }
  std::vector<kernel::LiteKernel *> input_kernels = SubgraphInputKernels(kernels);
  for (const auto &kernel : input_kernels) {
    for (const auto &tensor : kernel->in_tensors()) {
      auto iter = std::find(all_output_tensors.begin(), all_output_tensors.end(), tensor);
      if (iter == all_output_tensors.end() &&
          !(tensor->category() == mindspore::lite::Tensor::CONST && tensor->data_c() != nullptr)) {
        input_tensors.emplace_back(tensor);
      }
    }
  }
  return input_tensors;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> output_tensors;
  std::vector<lite::Tensor *> all_input_tensors;
  for (const auto &kernel : kernels) {
    auto kernel_in_tensors = kernel->in_tensors();
    all_input_tensors.insert(all_input_tensors.end(), kernel_in_tensors.begin(), kernel_in_tensors.end());
  }
  std::vector<kernel::LiteKernel *> output_kernels = SubgraphOutputKernels(kernels);
  for (const auto &kernel : output_kernels) {
    for (const auto &tensor : kernel->out_tensors()) {
      auto iter = std::find(all_input_tensors.begin(), all_input_tensors.end(), tensor);
      if (iter == all_input_tensors.end()) {
        output_tensors.emplace_back(tensor);
      }
    }
  }
  return output_tensors;
}

void LiteKernelUtil::InitIOKernels(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    // clean io kernels
    kernel->SetInKernel({});
    kernel->SetOutKernel({});
    // find io kernels
    for (auto *search_kernel : kernels) {
      if (search_kernel == kernel) {
        continue;
      }
      for (auto *tensor : kernel->in_tensors()) {
        if (lite::IsContain(search_kernel->out_tensors(), tensor)) {
          kernel->AddInKernel(search_kernel);
        }
      }
      for (auto *tensor : kernel->out_tensors()) {
        if (lite::IsContain(search_kernel->in_tensors(), tensor)) {
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

int LiteKernelUtil::SetInput(LiteKernel &kernelMod, std::vector<lite::Tensor *> inputs) { return -1; }
}  // namespace mindspore::kernel

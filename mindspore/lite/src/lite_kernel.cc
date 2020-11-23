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
#include <queue>
#include "src/tensor.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

void *LiteKernel::workspace_ = nullptr;

void LiteKernel::AllocWorkspace(size_t size) {
  if (size == 0) {
    return;
  }
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
    tensor->set_ref_count(this->out_kernels_.size());
  }
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->DecRefCount();
    if (0 >= tensor->ref_count()) {
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
    (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->set_infer_flag(true);
    auto ret = (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->InferShape(in_tensors_, out_tensors_);
    if (ret != 0) {
      (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->set_infer_flag(false);
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
  oss << ", " << this->in_tensors_.size() << " InputTensors:";
  for (auto tensor : in_tensors_) {
    oss << " " << tensor;
  }
  oss << ", " << this->out_tensors_.size() << " OutputTensors:";
  for (auto tensor : out_tensors_) {
    oss << " " << tensor;
  }
  oss << ", " << this->in_kernels_.size() << " InputKernels:";
  for (auto in_kernel : in_kernels_) {
    oss << " " << in_kernel->name_;
  }
  oss << ", " << this->out_kernels_.size() << " OutputKernels:";
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
  std::vector<kernel::LiteKernel *> input_kernels = SubgraphInputKernels(kernels);
  for (const auto &input_kernel : input_kernels) {
    auto &outer_in_kernels = input_kernel->in_kernels();
    auto &in_kernel_in_tensors = input_kernel->in_tensors();
    if (outer_in_kernels.empty()) {
      for (auto &in_kernel_in_tensor : in_kernel_in_tensors) {
        if (!in_kernel_in_tensor->IsConst()) {
          input_tensors.push_back(in_kernel_in_tensor);
        }
      }
      continue;
    }
    for (auto outer_in_kernel : outer_in_kernels) {
      auto iter = std::find(kernels.begin(), kernels.end(), outer_in_kernel);
      if (iter != kernels.end()) {
        continue;
      }
      auto &outer_in_kernel_out_tensors = outer_in_kernel->out_tensors();
      for (auto in_kernel_in_tensor : in_kernel_in_tensors) {
        auto outer_in_kernel_out_tensors_iter =
          std::find(outer_in_kernel_out_tensors.begin(), outer_in_kernel_out_tensors.end(), in_kernel_in_tensor);
        if (outer_in_kernel_out_tensors_iter != outer_in_kernel_out_tensors.end()) {
          input_tensors.emplace_back(in_kernel_in_tensor);
        }
      }
    }
  }
  return input_tensors;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> output_tensors;
  std::vector<kernel::LiteKernel *> output_kernels = SubgraphOutputKernels(kernels);
  for (const auto &output_kernel : output_kernels) {
    auto &outer_out_kernels = output_kernel->out_kernels();
    auto &out_kernel_out_tensors = output_kernel->out_tensors();
    if (outer_out_kernels.empty()) {
      output_tensors.insert(output_tensors.end(), out_kernel_out_tensors.begin(), out_kernel_out_tensors.end());
      continue;
    }
    for (auto outer_out_kernel : outer_out_kernels) {
      auto iter = std::find(kernels.begin(), kernels.end(), outer_out_kernel);
      if (iter != kernels.end()) {
        continue;
      }
      auto &outer_out_kernel_in_tensors = outer_out_kernel->in_tensors();
      for (auto out_kernel_out_tensor : out_kernel_out_tensors) {
        auto outer_out_kernel_in_tensors_iter =
          std::find(outer_out_kernel_in_tensors.begin(), outer_out_kernel_in_tensors.end(), out_kernel_out_tensor);
        if (outer_out_kernel_in_tensors_iter != outer_out_kernel_in_tensors.end()) {
          output_tensors.emplace_back(out_kernel_out_tensor);
        }
      }
    }
  }
  return output_tensors;
}

int LiteKernelUtil::TopologicalSortKernels(std::vector<kernel::LiteKernel *> *kernels) {
  auto old_kernels = *kernels;
  kernels->clear();
  std::queue<kernel::LiteKernel *> kernel_queue;
  for (auto kernel : old_kernels) {
    if (kernel->in_kernels().empty()) {
      kernel_queue.push(kernel);
      kernels->emplace_back(kernel);
    }
  }
  while (!kernel_queue.empty()) {
    auto cur_kernel = kernel_queue.front();
    kernel_queue.pop();
    MS_ASSERT(cur_kernel != nullptr);
    auto next_kernels = cur_kernel->out_kernels();
    for (auto next_kernel : next_kernels) {
      auto in_kernels = next_kernel->in_kernels();
      if (lite::IsContain(*kernels, const_cast<kernel::LiteKernel *>(next_kernel))) {
        MS_LOG(ERROR) << "TopologicalSortKernels failed, loop exist";
        return RET_ERROR;
      }
      if (std::all_of(in_kernels.begin(), in_kernels.end(), [&](const kernel::LiteKernel *in_kernel) {
            return lite::IsContain(*kernels, const_cast<kernel::LiteKernel *>(in_kernel));
          })) {
        kernel_queue.push(next_kernel);
      }
    }
  }
  if (kernels->size() != old_kernels.size()) {
    MS_LOG(ERROR) << "TopologicalSortKernels failed, kernels size before sort: " << old_kernels.size()
                  << ", kernels size after sort: " << kernels->size();
    return RET_ERROR;
  }
  return RET_OK;
}

void LiteKernelUtil::InitIOKernels(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    // clean io kernels
    kernel->set_in_kernels({});
    kernel->set_out_kernels({});
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

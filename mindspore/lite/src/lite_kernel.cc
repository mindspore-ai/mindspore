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
#include <set>
#include "src/tensor.h"
#include "src/common/utils.h"
#include "src/runtime/infer_manager.h"
#include "src/common/version_manager.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
#ifdef SUPPORT_TRAIN
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
#endif
bool LiteKernel::IsReady(const std::vector<lite::Tensor *> &scope_tensors) {
  return std::all_of(this->in_tensors().begin(), this->in_tensors().end(), [&](lite::Tensor *in_tensor) {
    if (IsContain(scope_tensors, in_tensor)) {
      return in_tensor->IsReady();
    } else {
      return true;
    }
  });
}

void LiteKernel::InitOutTensorInitRefCount() {
  for (auto *tensor : this->out_tensors_) {
    size_t init_ref_count = 0;
    for (auto *post_kernel : this->out_kernels_) {
      init_ref_count +=
        std::count_if(post_kernel->in_tensors_.begin(), post_kernel->in_tensors_.end(),
                      [&tensor](const lite::Tensor *post_kernel_in_tensor) { return post_kernel_in_tensor == tensor; });
    }
    tensor->set_init_ref_count(init_ref_count);
  }
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->set_ref_count(tensor->ref_count() - 1);
    if (0 >= tensor->ref_count()) {
      tensor->FreeData();
    }
  }
  return 0;
}

int LiteKernel::FreeInWorkTensor() const {
  for (auto &in_tensor : this->in_tensors_) {
    MS_ASSERT(in_tensor != nullptr);
    if (in_tensor->root_tensor() == in_tensor) {
      continue;
    }
    in_tensor->DecRefCount();
  }
  return RET_OK;
}

int LiteKernel::PreProcess() {
  if (!InferShapeDone()) {
    op_parameter_->infer_flag_ = true;
    auto ret = lite::KernelInferShape(in_tensors_, &out_tensors_, op_parameter_);
    if (ret != 0) {
      op_parameter_->infer_flag_ = false;
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  for (auto *output : this->out_tensors()) {
    MS_ASSERT(output != nullptr);
    if (desc_.data_type == kNumberTypeFloat16 && output->data_type() == kNumberTypeFloat32) {
      output->set_data_type(kNumberTypeFloat16);
    }
    if (output->ElementsNum() >= MAX_MALLOC_SIZE / static_cast<int>(sizeof(int64_t))) {
      MS_LOG(ERROR) << "The size of output tensor is too big";
      return RET_ERROR;
    }
    auto ret = output->MallocData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MallocData failed";
      return ret;
    }
  }
  return RET_OK;
}

int LiteKernel::PostProcess() {
#ifdef SUPPORT_TRAIN
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
#else
  for (auto *output : this->out_tensors()) {
    MS_ASSERT(output != nullptr);
    output->ResetRefCount();
  }
  return FreeInWorkTensor();
#endif
}

int LiteKernel::Run(const KernelCallBack &before, const KernelCallBack &after) {
  if (before != nullptr) {
    if (!before(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
                {this->name_, this->type_str()})) {
      MS_LOG(WARNING) << "run kernel before_callback failed, name: " << this->name_;
    }
  }
  // Support ZeroShape
  size_t zero_shape_num = 0;
  for (auto tensor : this->out_tensors_) {
    for (size_t i = 0; i < tensor->shape().size(); i++) {
      if (tensor->shape()[i] == 0) {
        zero_shape_num++;
        break;
      }
    }
  }
  if (zero_shape_num != this->out_tensors_.size()) {
    auto ret = Run();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << this->name_;
      return ret;
    }
  }
  if (after != nullptr) {
    if (!after(TensorVectorCast(this->in_tensors_), TensorVectorCast(this->out_tensors_),
               {this->name_, this->type_str()})) {
      MS_LOG(WARNING) << "run kernel after_callback failed, name: " << this->name_;
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

void LiteKernel::FindInoutKernels(const std::vector<kernel::LiteKernel *> &scope_kernels) {
  // clean io kernels
  this->in_kernels_.clear();
  this->out_kernels_.clear();
  // find io kernels
  for (auto *scope_kernel : scope_kernels) {
    if (scope_kernel == this) {
      continue;
    }
    for (auto *tensor : this->in_tensors_) {
      if (lite::IsContain(scope_kernel->out_tensors(), tensor)) {
        if (!lite::IsContain(this->in_kernels(), scope_kernel)) {
          this->AddInKernel(scope_kernel);
        }
      }
    }
    for (auto *tensor : this->out_tensors_) {
      if (lite::IsContain(scope_kernel->in_tensors(), tensor)) {
        if (!lite::IsContain(this->out_kernels(), scope_kernel)) {
          this->AddOutKernel(scope_kernel);
        }
      }
    }
  }
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphInputNodes(const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<kernel::LiteKernel *> input_nodes;
  for (const auto &kernel : kernels) {
    // if kernel has no pre-kernel, kernel is a graph input, it must be a subgraph input
    if (kernel->in_kernels().empty() && !kernel->in_tensors().empty()) {
      input_nodes.insert(kernel);
      continue;
    }
    auto all_input_tensors = kernel->in_tensors();
    // remove all const tensor from input tensors
    for (auto iter = all_input_tensors.begin(); iter != all_input_tensors.end();) {
      if ((*iter)->IsConst()) {
        iter = all_input_tensors.erase(iter);
      } else {
        iter++;
      }
    }
    for (const auto &kernel_in_subgraph : kernels) {
      // remove input tensors from kernel in subgraph
      for (const auto *tensor : kernel_in_subgraph->out_tensors()) {
        auto ret = std::find(all_input_tensors.begin(), all_input_tensors.end(), tensor);
        if (ret != all_input_tensors.end()) {
          all_input_tensors.erase(ret);
        }
      }
    }
    // if some input tensor is not from kernel in subgraph
    if (!all_input_tensors.empty()) {
      input_nodes.insert(kernel);
    }
  }
  std::vector<kernel::LiteKernel *> result;
  result.insert(result.end(), input_nodes.begin(), input_nodes.end());
  return result;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphOutputNodes(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<kernel::LiteKernel *> output_nodes;
  // if kernel has no post-kernel, kernel is a graph output, it must be a subgraph output
  for (const auto &kernel : kernels) {
    if (kernel->is_model_output() || (kernel->out_kernels().empty() && !kernel->out_tensors().empty())) {
      output_nodes.insert(kernel);
      continue;
    }
    for (const auto &output : kernel->out_kernels()) {
      auto out_kernel_in_graph = std::find(kernels.begin(), kernels.end(), output);
      if (out_kernel_in_graph == kernels.end()) {
        output_nodes.insert(kernel);
        break;
      }
    }
  }
  std::vector<kernel::LiteKernel *> result;
  result.insert(result.end(), output_nodes.begin(), output_nodes.end());
  return result;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<lite::Tensor *> input_tensors;
  std::vector<kernel::LiteKernel *> input_nodes = SubgraphInputNodes(kernels);
  for (const auto &input_node : input_nodes) {
    auto &in_node_in_kernels = input_node->in_kernels();
    auto &in_node_in_tensors = input_node->in_tensors();
    for (auto &in_node_in_tensor : in_node_in_tensors) {
      if (in_node_in_tensor->IsGraphInput()) {
        input_tensors.insert(in_node_in_tensor);
      }
    }
    for (auto in_node_in_kernel : in_node_in_kernels) {
      auto iter = std::find(kernels.begin(), kernels.end(), in_node_in_kernel);
      if (iter != kernels.end()) {
        continue;
      }
      auto &outer_in_kernel_out_tensors = in_node_in_kernel->out_tensors();
      for (auto in_node_in_tensor : in_node_in_tensors) {
        auto outer_in_kernel_out_tensors_iter =
          std::find(outer_in_kernel_out_tensors.begin(), outer_in_kernel_out_tensors.end(), in_node_in_tensor);
        if (outer_in_kernel_out_tensors_iter != outer_in_kernel_out_tensors.end()) {
          input_tensors.insert(in_node_in_tensor);
        }
      }
    }
  }
  std::vector<lite::Tensor *> result;
  result.insert(result.end(), input_tensors.begin(), input_tensors.end());
  return result;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<lite::Tensor *> output_tensors;
  std::vector<kernel::LiteKernel *> output_nodes = SubgraphOutputNodes(kernels);
  for (const auto &output_kernel : output_nodes) {
    auto &outer_out_kernels = output_kernel->out_kernels();
    auto &out_kernel_out_tensors = output_kernel->out_tensors();
    for (auto out_kernel_out_tensor : out_kernel_out_tensors) {
      if (out_kernel_out_tensor->IsGraphOutput()) {
        output_tensors.insert(out_kernel_out_tensor);
      }
    }
    if (!outer_out_kernels.empty()) {
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
            output_tensors.insert(out_kernel_out_tensor);
          }
        }
      }
    }
  }
  std::vector<lite::Tensor *> result;
  result.insert(result.end(), output_tensors.begin(), output_tensors.end());
  return result;
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

void LiteKernelUtil::InitTensorInitRefCount(const std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    kernel->InitOutTensorInitRefCount();
  }
}

int LiteKernelUtil::SetInput(LiteKernel &kernelMod, const std::vector<lite::Tensor *> &inputs) { return -1; }
}  // namespace mindspore::kernel

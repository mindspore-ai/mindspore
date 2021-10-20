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

#include "src/lite_kernel_util.h"
#include <queue>
#include <unordered_map>
#include <set>
#include "src/sub_graph_kernel.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

std::set<lite::Tensor *> LiteKernelUtil::AllOutTensor(const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<lite::Tensor *> all_out_tensors{};
  for (const auto &kernel_in_subgraph : kernels) {
    for (auto *tensor : kernel_in_subgraph->out_tensors()) {
      all_out_tensors.insert(tensor);
    }
  }
  return all_out_tensors;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphInputNodes(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> input_nodes;
  std::set<lite::Tensor *> all_out_tensors = AllOutTensor(kernels);
  for (const auto &kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    bool kernel_is_input = false;
    auto all_input_tensors = kernel->in_tensors();
    for (auto input : kernel->in_tensors()) {
      if (input->IsConst()) {
        continue;
      }
      if (all_out_tensors.find(input) != all_out_tensors.end()) {
        continue;
      }
      kernel_is_input = true;
      break;
    }
    if (kernel_is_input && !lite::IsContain(input_nodes, kernel)) {
      input_nodes.push_back(kernel);
    }
  }
  return input_nodes;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphOutputNodes(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::set<kernel::LiteKernel *> all_kernels{};
  for (const auto &kernel : kernels) {
    all_kernels.insert(kernel);
  }
  std::vector<kernel::LiteKernel *> output_nodes;
  // if kernel has no post-kernel, kernel is a graph output, it must be a subgraph output
  for (const auto &kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    if (kernel->is_model_output() || (kernel->out_kernels().empty() && !kernel->out_tensors().empty())) {
      if (!lite::IsContain(output_nodes, kernel)) {
        output_nodes.push_back(kernel);
      }
      continue;
    }
    if (std::any_of(kernel->out_kernels().begin(), kernel->out_kernels().end(),
                    [&all_kernels](kernel::LiteKernel *tmp) { return all_kernels.find(tmp) == all_kernels.end(); }) &&
        !lite::IsContain(output_nodes, kernel)) {
      output_nodes.push_back(kernel);
    }
  }
  return output_nodes;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> input_tensors;
  std::vector<kernel::LiteKernel *> input_nodes = SubgraphInputNodes(kernels);
  for (const auto &input_node : input_nodes) {
    auto &in_node_in_kernels = input_node->in_kernels();
    auto &in_node_in_tensors = input_node->in_tensors();
    for (auto &in_node_in_tensor : in_node_in_tensors) {
      if (in_node_in_tensor->IsGraphInput()) {
        if (!lite::IsContain(input_tensors, in_node_in_tensor)) {
          input_tensors.push_back(in_node_in_tensor);
        }
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
          if (!lite::IsContain(input_tensors, in_node_in_tensor)) {
            input_tensors.push_back(in_node_in_tensor);
          }
        }
      }
    }
  }
  return input_tensors;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> output_tensors;
  std::vector<kernel::LiteKernel *> output_nodes = SubgraphOutputNodes(kernels);
  for (const auto &output_kernel : output_nodes) {
    auto &outer_out_kernels = output_kernel->out_kernels();
    auto &out_kernel_out_tensors = output_kernel->out_tensors();
    for (auto out_kernel_out_tensor : out_kernel_out_tensors) {
      if (out_kernel_out_tensor->IsGraphOutput()) {
        if (!lite::IsContain(output_tensors, out_kernel_out_tensor)) {
          output_tensors.push_back(out_kernel_out_tensor);
        }
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
            if (!lite::IsContain(output_tensors, out_kernel_out_tensor)) {
              output_tensors.push_back(out_kernel_out_tensor);
            }
          }
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

void LiteKernelUtil::InitTensorInitRefCount(const std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    kernel->InitOutTensorInitRefCount(&kernels);
  }
}

int LiteKernelUtil::SetInput(const LiteKernel &kernelMod, const std::vector<lite::Tensor *> &inputs) { return -1; }

#ifndef CONTROLFLOW_TENSORLIST_CLIP
bool LiteKernelUtil::IsSwitchCall(kernel::LiteKernel *kernel) {
#ifndef DELEGATE_CLIP
  if (kernel->desc().arch == kernel::kDelegate) {
    return false;
  }
#endif
  auto *subgraph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
  if (subgraph_kernel == nullptr) {
    return false;
  }
  for (auto &node : subgraph_kernel->nodes()) {
    if (node->type() == schema::PrimitiveType_Switch &&
        InputsContainsSpecificNode(node, schema::PrimitiveType_PartialFusion) && node->out_kernels().size() == 1 &&
        node->out_kernels().front()->type() == schema::PrimitiveType_Call) {
      return true;
    }
  }

  return false;
}
#endif

kernel::LiteKernel *LiteKernelUtil::GetInputsSpecificNode(const kernel::LiteKernel *kernel,
                                                          const schema::PrimitiveType &primitive_type) {
  for (auto input : kernel->in_kernels()) {
    if (input->type() == primitive_type) {
      return input;
    }
  }
  return nullptr;
}

bool LiteKernelUtil::InputsContainsSpecificNode(const kernel::LiteKernel *kernel,
                                                const schema::PrimitiveType &primitive_type) {
  if (GetInputsSpecificNode(kernel, primitive_type)) {
    return true;
  }
  return false;
}

void LiteKernelUtil::FindAllInoutKernels(const std::vector<kernel::LiteKernel *> &kernels) {
  std::unordered_map<lite::Tensor *, kernel::LiteKernel *> tensor_pre_kernel;
  std::unordered_map<lite::Tensor *, std::vector<kernel::LiteKernel *>> tensor_post_kernels;
  for (auto *kernel : kernels) {
    for (auto *tensor : kernel->out_tensors()) {
      tensor_pre_kernel[tensor] = kernel;
    }
    for (auto *tensor : kernel->in_tensors()) {
      (tensor_post_kernels[tensor]).push_back(kernel);
    }
  }

  for (auto *kernel : kernels) {
    kernel->set_in_kernels({});
    for (auto *tensor : kernel->in_tensors()) {
      auto iter = tensor_pre_kernel.find(tensor);
      if (iter != tensor_pre_kernel.end() && kernel != iter->second) {
        kernel->AddInKernel(iter->second);
      }
    }
    kernel->set_out_kernels({});
    for (auto *tensor : kernel->out_tensors()) {
      auto iter = tensor_post_kernels.find(tensor);
      if (iter != tensor_post_kernels.end()) {
        for (auto *find_kernel : iter->second) {
          if (kernel == find_kernel) {
            continue;
          }
          kernel->AddOutKernel(find_kernel);
        }
      }
    }
  }
}

}  // namespace mindspore::kernel

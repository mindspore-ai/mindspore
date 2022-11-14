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

#include "src/litert/kernel_exec_util.h"
#include <utility>
#include <queue>
#include <unordered_map>
#include <set>
#include "src/litert/sub_graph_kernel.h"
#include "nnacl/call_parameter.h"
#if GPU_OPENCL
#include "src/litert/kernel/opencl/opencl_subgraph.h"
#include "src/litert/kernel/gpu/opencl/opencl_runtime.h"
#endif
#include "src/control_flow/control_subgraph_creator.h"
#include "src/litert/kernel/cpu/base/partial_fusion.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

std::set<lite::Tensor *> KernelExecUtil::AllOutTensor(const std::vector<KernelExec *> &kernels) {
  std::set<lite::Tensor *> all_out_tensors{};
  for (const auto &kernel_in_subgraph : kernels) {
    for (auto *tensor : kernel_in_subgraph->out_tensors()) {
      (void)all_out_tensors.insert(tensor);
    }
  }
  return all_out_tensors;
}

std::vector<KernelExec *> KernelExecUtil::SubgraphInputNodes(const std::vector<KernelExec *> &kernels) {
  std::vector<KernelExec *> input_nodes;
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

std::vector<KernelExec *> KernelExecUtil::SubgraphOutputNodes(const std::vector<KernelExec *> &kernels) {
  std::set<KernelExec *> all_kernels{};
  for (const auto &kernel : kernels) {
    (void)all_kernels.insert(kernel);
  }
  std::vector<KernelExec *> output_nodes;
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
                    [&all_kernels](KernelExec *tmp) { return all_kernels.find(tmp) == all_kernels.end(); }) &&
        !lite::IsContain(output_nodes, kernel)) {
      output_nodes.push_back(kernel);
    }
  }
  return output_nodes;
}

std::vector<lite::Tensor *> KernelExecUtil::SubgraphInputTensors(const std::vector<KernelExec *> &kernels) {
  std::vector<lite::Tensor *> input_tensors;
  std::vector<KernelExec *> input_nodes = SubgraphInputNodes(kernels);
  for (const auto &input_node : input_nodes) {
    auto &in_node_in_kernels = input_node->in_kernels();
    auto &in_node_in_tensors = input_node->in_tensors();
    for (auto &in_node_in_tensor : in_node_in_tensors) {
      if (in_node_in_tensor->IsGraphInput() || (in_node_in_kernels.empty() && !in_node_in_tensor->IsConst())) {
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

std::vector<lite::Tensor *> KernelExecUtil::SubgraphOutputTensors(const std::vector<KernelExec *> &kernels) {
  std::vector<lite::Tensor *> output_tensors;
  std::vector<KernelExec *> output_nodes = SubgraphOutputNodes(kernels);
  for (const auto &output_kernel : output_nodes) {
    auto &outer_out_kernels = output_kernel->out_kernels();
    auto &out_kernel_out_tensors = output_kernel->out_tensors();
    for (auto out_kernel_out_tensor : out_kernel_out_tensors) {
      if ((out_kernel_out_tensor->IsGraphOutput() || outer_out_kernels.empty()) &&
          !lite::IsContain(output_tensors, out_kernel_out_tensor)) {
        output_tensors.push_back(out_kernel_out_tensor);
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
          if ((outer_out_kernel_in_tensors_iter != outer_out_kernel_in_tensors.end()) &&
              !lite::IsContain(output_tensors, out_kernel_out_tensor)) {
            output_tensors.push_back(out_kernel_out_tensor);
          }
        }
      }
    }
  }
  return output_tensors;
}

int KernelExecUtil::TopologicalSortKernels(std::vector<KernelExec *> *kernels) {
  auto old_kernels = *kernels;
  kernels->clear();
  std::queue<KernelExec *> kernel_queue;
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
      if (lite::IsContain(*kernels, const_cast<KernelExec *>(next_kernel))) {
        MS_LOG(ERROR) << "TopologicalSortKernels failed, loop exist";
        return RET_ERROR;
      }
      if (std::all_of(in_kernels.begin(), in_kernels.end(), [&](const KernelExec *in_kernel) {
            return lite::IsContain(*kernels, const_cast<KernelExec *>(in_kernel));
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

void KernelExecUtil::InitTensorInitRefCount(const std::vector<KernelExec *> &kernels) {
  for (auto *kernel : kernels) {
    kernel->InitOutTensorInitRefCount(&kernels);
  }
}

KernelExec *KernelExecUtil::GetInputsSpecificNode(const KernelExec *kernel,
                                                  const schema::PrimitiveType &primitive_type) {
  for (auto input : kernel->in_kernels()) {
    if (input->type() == primitive_type) {
      return input;
    }
  }
  return nullptr;
}

bool KernelExecUtil::InputsContainsSpecificNode(const KernelExec *kernel, const schema::PrimitiveType &primitive_type) {
  if (GetInputsSpecificNode(kernel, primitive_type)) {
    return true;
  }
  return false;
}

void KernelExecUtil::FindAllInoutKernels(const std::vector<KernelExec *> &kernels) {
  std::unordered_map<lite::Tensor *, KernelExec *> tensor_pre_kernel;
  std::unordered_map<lite::Tensor *, std::vector<KernelExec *>> tensor_post_kernels;
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

void KernelExecUtil::FindAllInoutKernelsInSubgraphKernel(const std::vector<KernelExec *> &kernels) {
  std::vector<KernelExec *> all_kernels;
  for (auto kernel : kernels) {
    if (kernel->desc().arch == kDelegate) {
      all_kernels.push_back(kernel);
      continue;
    }
    auto sub_graph = reinterpret_cast<SubGraphKernel *>(kernel);
    MS_ASSERT(sub_graph != nullptr);
    auto kernel_in_subgraph = sub_graph->nodes();
    (void)all_kernels.insert(all_kernels.end(), kernel_in_subgraph.begin(), kernel_in_subgraph.end());
  }

  KernelExecUtil::FindAllInoutKernels(all_kernels);
}

KernelExec *KernelExecUtil::FindInKernelForInTensor(const KernelExec *kernel, lite::Tensor *tensor) {
  for (auto in_kernel : kernel->in_kernels()) {
    if (lite::IsContain(in_kernel->out_tensors(), tensor)) {
      return in_kernel;
    }
  }
  return nullptr;
}

std::vector<KernelExec *> KernelExecUtil::FindOutKernelsForOutTensor(const KernelExec *kernel, lite::Tensor *tensor) {
  std::vector<KernelExec *> out_kernels;
  for (auto out_kernel : kernel->out_kernels()) {
    if (lite::IsContain(out_kernel->in_tensors(), tensor)) {
      out_kernels.push_back(out_kernel);
    }
  }
  return out_kernels;
}

int KernelExecUtil::SetKernelTensorDataType(const kernel::KernelExec *kernel) {
  CHECK_NULL_RETURN(kernel);
  if (kernel->desc().arch != kernel::KERNEL_ARCH::kCPU) {
    return RET_OK;
  }
  if (kernel->desc().data_type == kNumberTypeFloat16) {
    for (auto tensor : kernel->out_tensors()) {
      if (tensor->data_type() == kNumberTypeFloat32) {
        tensor->set_data_type(kNumberTypeFloat16);
      }
    }
  } else if (kernel->desc().data_type == kNumberTypeFloat32) {
    for (auto tensor : kernel->in_tensors()) {
      if (!tensor->IsConst() && tensor->data_type() == kNumberTypeFloat16) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
    for (auto tensor : kernel->out_tensors()) {
      if (tensor->data_type() == kNumberTypeFloat16 && kernel->type() != schema::PrimitiveType_Cast) {
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
  }
  return RET_OK;
}

bool KernelExecUtil::IsOutputSubGraph(const KernelExec *subgraph_kernel) {
  return !subgraph_kernel->out_tensors().empty() &&
         std::all_of(subgraph_kernel->out_tensors().begin(), subgraph_kernel->out_tensors().end(),
                     [](lite::Tensor *tensor) { return tensor->IsGraphOutput(); });
}

namespace {
SubGraphKernel *CreateCustomSubGraph(std::vector<KernelExec *> &&input_kernels,
                                     std::vector<KernelExec *> &&output_kernels,
                                     const std::vector<KernelExec *> &kernels, Kernel *kernel) {
  auto sub_kernel = new (std::nothrow) CustomSubGraph(input_kernels, output_kernels, kernels, kernel);
  if (sub_kernel == nullptr) {
    MS_LOG(ERROR) << "create custom subgraph failed!";
    return nullptr;
  }
  return sub_kernel;
}
}  // namespace

SubGraphKernel *KernelExecUtil::CreateSubGraphKernel(const std::vector<KernelExec *> &kernels,
                                                     const std::vector<lite::Tensor *> *in_tensors,
                                                     const std::vector<lite::Tensor *> *out_tensors, SubGraphType type,
                                                     const lite::InnerContext &context, int schema_version) {
  std::vector<lite::Tensor *> input_tensors;
  std::vector<lite::Tensor *> output_tensors;
  if (in_tensors != nullptr) {
    input_tensors = *in_tensors;
  } else {
    input_tensors = SubgraphInputTensors(kernels);
  }
  if (out_tensors != nullptr) {
    output_tensors = *out_tensors;
  } else {
    output_tensors = SubgraphOutputTensors(kernels);
  }
  auto lite_kernel = new (std::nothrow) LiteKernel(nullptr, input_tensors, output_tensors, &context);
  if (lite_kernel == nullptr) {
    return nullptr;
  }
  std::vector<KernelExec *> input_kernels = SubgraphInputNodes(kernels);
  std::vector<KernelExec *> output_kernels = SubgraphOutputNodes(kernels);
  SubGraphKernel *sub_graph = nullptr;
  switch (type) {
    case kCpuFP32SubGraph: {
      sub_graph = new (std::nothrow) CpuFp32SubGraph(input_kernels, output_kernels, kernels, lite_kernel);
    } break;
    case kCpuFP16SubGraph: {
#ifdef ENABLE_FP16
      sub_graph = new (std::nothrow) CpuFp16SubGraph(input_kernels, output_kernels, kernels, lite_kernel);
      for (auto out_tensor : output_tensors) {
        if (out_tensor->data_type() == kNumberTypeFloat32) {
          out_tensor->set_data_type(kNumberTypeFloat16);
        }
      }
#endif
    } break;
    case kGpuFp32SubGraph:
    case kGpuFp16SubGraph: {
#if GPU_OPENCL
      sub_graph = new (std::nothrow) OpenCLSubGraph(input_kernels, output_kernels, kernels, lite_kernel);
#endif
    } break;
    case kCustomSubGraph: {
      sub_graph = CreateCustomSubGraph(std::move(input_kernels), std::move(output_kernels), kernels, lite_kernel);
    } break;
    case kEntranceSubGraph:
    case kExitSubGraph: {
      sub_graph = lite::CreateControlSubgraph(type, lite_kernel);
    } break;
    default: {
      MS_LOG(ERROR) << "not support subgraph type: " << type;
      delete lite_kernel;
      return nullptr;
    }
  }
  if (sub_graph == nullptr) {
    delete lite_kernel;
    MS_LOG(ERROR) << "create subgraph type " << type << "failed.";
    return nullptr;
  }
  sub_graph->set_context(&context);
  sub_graph->SetSchemaVersion(schema_version);
  return sub_graph;
}

int KernelExecUtil::ReplaceSubGraphNodesInTensor(KernelExec *kernel, const lite::Tensor *old_tensor,
                                                 lite::Tensor *new_tensor) {
  int ref_count = 0;
  /* set op input for calculate */
  if (kernel->desc().arch == kDelegate) {
    ref_count++;
  } else {
    auto subgraph_kernel = reinterpret_cast<SubGraphKernel *>(kernel);
    if (subgraph_kernel == nullptr) {
      MS_LOG(ERROR) << "cast to subgraph kernel failed.";
      return RET_ERROR;
    }
    for (auto in_node : reinterpret_cast<SubGraphKernel *>(kernel)->in_nodes()) {
      for (size_t node_in_index = 0; node_in_index < in_node->in_tensors().size(); node_in_index++) {
        if (old_tensor == in_node->in_tensors()[node_in_index]) {
          in_node->set_in_tensor(new_tensor, node_in_index);
          ref_count++;
        }
      }
    }
  }
  new_tensor->set_init_ref_count(ref_count);
  return RET_OK;
}

int KernelExecUtil::ReplaceSubGraphNodesOutTensor(KernelExec *kernel, const lite::Tensor *old_tensor,
                                                  lite::Tensor *new_tensor) {
  int ref_count = 0;
  /* set op output for calculate */
  if (kernel->desc().arch == kDelegate) {
    ref_count++;
  } else {
    auto subgraph_kernel = reinterpret_cast<SubGraphKernel *>(kernel);
    if (subgraph_kernel == nullptr) {
      MS_LOG(ERROR) << "cast to subgraph kernel failed.";
      return RET_ERROR;
    }
    for (auto out_node : reinterpret_cast<SubGraphKernel *>(kernel)->out_nodes()) {
      for (size_t node_out_index = 0; node_out_index < out_node->out_tensors().size(); node_out_index++) {
        if (old_tensor == out_node->out_tensors()[node_out_index]) {
          out_node->set_out_tensor(new_tensor, node_out_index);
          ref_count++;
        }
      }
    }
  }
  new_tensor->set_init_ref_count(ref_count);
  return RET_OK;
}

SubGraphKernel *KernelExecUtil::BelongToWhichSubGraph(const std::vector<KernelExec *> &subgraphs, KernelExec *kernel) {
  for (auto &item : subgraphs) {
    if (item->subgraph_type() == kernel::kNotSubGraph) {
      continue;
    }
    auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(item);
    if (subgraph == nullptr) {
      continue;
    }
    if (std::any_of(subgraph->nodes().begin(), subgraph->nodes().end(),
                    [&kernel](const KernelExec *node) { return node == kernel; })) {
      return subgraph;
    }
  }
  return nullptr;
}

#ifndef CONTROLFLOW_TENSORLIST_CLIP
bool KernelExecUtil::IsSwitchTypeCall(KernelExec *kernel) {
  if (kernel->desc().arch == kDelegate) {
    return false;
  }
  auto *subgraph_kernel = reinterpret_cast<SubGraphKernel *>(kernel);
  if (subgraph_kernel == nullptr) {
    return false;
  }
  for (auto &node : subgraph_kernel->nodes()) {
    if ((node->type() == schema::PrimitiveType_Switch || node->type() == schema::PrimitiveType_SwitchLayer) &&
        InputsContainsSpecificNode(node, schema::PrimitiveType_PartialFusion) && node->out_kernels().size() == 1 &&
        node->out_kernels().front()->type() == schema::PrimitiveType_Call) {
      return true;
    }
  }

  return false;
}

bool KernelExecUtil::IsNonTailCall(const KernelExec *node) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is nullptr";
    return false;
  }
  auto parameter = reinterpret_cast<CallParameter *>(node->op_parameter());
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "Parameter is nullptr";
    return false;
  }
  return node->type() == schema::PrimitiveType_Call && !(parameter->is_tail_call);
}

bool KernelExecUtil::IsTailCall(const KernelExec *node) {
  return node->type() == schema::PrimitiveType_Call &&
         (reinterpret_cast<CallParameter *>(node->op_parameter())->is_tail_call);
}

bool KernelExecUtil::IsNonTailCallSubGraph(KernelExec *kernel) {
  auto subgraph_kernel = reinterpret_cast<SubGraphKernel *>(kernel);
  if (subgraph_kernel == nullptr) {
    return false;
  }
  auto nodes = subgraph_kernel->nodes();
  return std::any_of(nodes.begin(), nodes.end(),
                     [](const KernelExec *node) { return KernelExecUtil::IsNonTailCall(node); });
}

bool KernelExecUtil::IsTailCallSubGraph(KernelExec *kernel) {
  auto subgraph_kernel = reinterpret_cast<SubGraphKernel *>(kernel);
  if (subgraph_kernel == nullptr) {
    return false;
  }
  if (IsNonTailCallSubGraph(subgraph_kernel)) {
    return false;
  }
  auto output_nodes = subgraph_kernel->out_nodes();
  if (std::any_of(output_nodes.begin(), output_nodes.end(), [](const KernelExec *node) { return IsTailCall(node); })) {
    return true;
  }
  return false;
}

std::vector<KernelExec *> KernelExecUtil::GetCallInputPartials(const KernelExec *call_node) {
  if (call_node->type() != schema::PrimitiveType_Call) {
    MS_LOG(ERROR) << "input node is not call node.";
    return {};
  }
  auto call_inputs = call_node->in_kernels();
  if (call_inputs.size() != 1) {
    MS_LOG(ERROR) << "call inputs size is: " << call_inputs.size() << ", not is 1.";
    return {};
  }

  std::vector<KernelExec *> partial_nodes{};
  auto call_input_node = call_inputs.front();
  switch (call_input_node->type()) {
    case schema::PrimitiveType_PartialFusion: {
      partial_nodes.push_back(call_input_node);
      break;
    }
    case schema::PrimitiveType_Switch:
    case schema::PrimitiveType_SwitchLayer: {
      auto switch_type_node = call_input_node;
      for (auto item : switch_type_node->in_kernels()) {
        if (item->type() == schema::PrimitiveType_PartialFusion) {
          partial_nodes.push_back(item);
        }
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "not support call input type is: " << call_input_node->type();
      return {};
    }
  }
  return partial_nodes;
}

std::vector<KernelExec *> KernelExecUtil::GetCallInputPartialsCorrespondingOutputSubgraph(KernelExec *call_node) {
  auto partial_nodes = GetCallInputPartials(call_node);
  std::vector<KernelExec *> all_subgraphs{};
  for (auto partial_node : partial_nodes) {
    auto partial_kernel = reinterpret_cast<PartialFusionKernel *>(partial_node->kernel());
    if (partial_kernel == nullptr) {
      MS_LOG(ERROR) << "cast to partial kernel failed.";
      return all_subgraphs;
    }
    // only get the output subgraph, the last subgraph is the output subgraph.
    auto partial_subgraphs = partial_kernel->subgraph_kernels();
    all_subgraphs.push_back(partial_subgraphs.back());
    // exit graph's input graph also need set same output tensor init refcount.
    if (partial_subgraphs.size() > 1 && partial_subgraphs.back()->subgraph_type() == kExitSubGraph) {
      auto last_index = partial_subgraphs.size() - 1;
      all_subgraphs.push_back(partial_subgraphs[last_index - 1]);
    }
  }
  return all_subgraphs;
}

KernelExec *KernelExecUtil::GetPartialOutputCall(const KernelExec *partial_node) {
  if (partial_node->type() != schema::PrimitiveType_PartialFusion) {
    MS_LOG(ERROR) << "input node is not partial node.";
    return nullptr;
  }
  auto partial_outputs = partial_node->out_kernels();
  if (partial_outputs.size() != 1) {
    MS_LOG(ERROR) << "partial outputs size is: " << partial_outputs.size() << ", not is 1.";
    return nullptr;
  }

  KernelExec *call_node = nullptr;
  auto partial_output_node = partial_outputs.front();
  switch (partial_output_node->type()) {
    case schema::PrimitiveType_Call: {
      call_node = partial_output_node;
      break;
    }
    case schema::PrimitiveType_Switch:
    case schema::PrimitiveType_SwitchLayer: {
      auto switch_type_node = partial_output_node;
      auto switch_outputs = switch_type_node->out_kernels();
      if (switch_outputs.size() != 1) {
        MS_LOG(ERROR) << "switch outputs size is: " << switch_outputs.size() << ", not is 1.";
        return nullptr;
      }
      if (switch_outputs.front()->type() == schema::PrimitiveType_Call) {
        call_node = switch_outputs.front();
      } else {
        MS_LOG(ERROR) << "graph is not right, switch output is not call node.";
        return nullptr;
      }
      break;
    }
    default: {
      MS_LOG(ERROR) << "not support partial output type is: " << partial_output_node->type();
      return nullptr;
    }
  }
  return call_node;
}

#else

bool KernelExecUtil::IsSwitchTypeCall(KernelExec *kernel) { return false; }

bool KernelExecUtil::IsNonTailCall(const KernelExec *node) { return false; }

bool KernelExecUtil::IsTailCall(const KernelExec *node) { return false; }

bool KernelExecUtil::IsNonTailCallSubGraph(KernelExec *kernel) { return false; }

bool KernelExecUtil::IsTailCallSubGraph(KernelExec *kernel) { return false; }

std::vector<KernelExec *> KernelExecUtil::GetCallInputPartials(const KernelExec *call_node) { return {}; }

std::vector<KernelExec *> KernelExecUtil::GetCallInputPartialsCorrespondingOutputSubgraph(KernelExec *call_node) {
  return {};
}

KernelExec *KernelExecUtil::GetPartialOutputCall(const KernelExec *partial_node) { return nullptr; }

#endif
}  // namespace mindspore::kernel

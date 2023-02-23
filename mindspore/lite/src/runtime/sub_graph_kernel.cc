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

#include "src/runtime/sub_graph_kernel.h"
#include <algorithm>
#include <queue>
#include "src/tensor.h"
#include "src/tensorlist.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/cpu/fp16/fp16_op_handler.h"
#endif
#include "src/common/version_manager.h"
#include "src/runtime/infer_manager.h"
#include "src/common/tensor_util.h"
#include "src/common/utils.h"
#include "src/common/prim_inner.h"
#include "src/runtime/kernel_exec_util.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_ERR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

std::string SubGraphKernel::ToString() const {
  std::ostringstream oss;
  oss << "===============================================" << std::endl
      << "Subgraph type : " << this->subgraph_type_ << std::endl;
  oss << this->in_tensors().size() << " Subgraph inputTensors:" << std::endl;
  for (auto tensor : in_tensors()) {
    oss << tensor->ToString() << std::endl;
  }
  oss << std::endl << this->out_tensors().size() << " Subgraph outputTensors:" << std::endl;
  for (auto tensor : out_tensors()) {
    oss << tensor->ToString() << std::endl;
  }
  oss << std::endl << this->in_nodes_.size() << " Subgraph input nodes:" << std::endl;
  for (auto kernel : this->in_nodes_) {
    oss << "***********************************************" << std::endl;
    oss << kernel->ToString() << std::endl;
  }
  oss << std::endl << this->out_nodes_.size() << " Subgraph output nodes:" << std::endl;
  for (auto kernel : this->out_nodes_) {
    oss << "***********************************************" << std::endl;
    oss << kernel->ToString() << std::endl;
  }
  oss << std::endl << nodes_.size() << " nodes in subgraph:" << std::endl;
  for (auto kernel : this->nodes_) {
    oss << "***********************************************" << std::endl;
    oss << kernel->ToString() << std::endl;
  }
  return oss.str();
}

int SubGraphKernel::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  if (this->executor_ == nullptr) {
    MS_LOG(ERROR) << "executor is nullptr";
    return RET_ERROR;
  }
  auto ret = executor_->Run(this->in_tensors(), this->out_tensors(), this->nodes_, before, after);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run sub graph failed: " << ret;
    return ret;
  }

  return lite::RET_OK;
}

int SubGraphKernel::ReSize() {
  for (auto kernel : nodes_) {
    MS_CHECK_FALSE_MSG(kernel == nullptr, RET_ERROR, "input kernel is nullptr.");
    MS_CHECK_FALSE_MSG(kernel->subgraph_type() != kernel::kNotSubGraph, RET_ERROR,
                       "all nodes in should be kernel in subgraph kernels");
    std::vector<lite::Tensor *> inputs = kernel->in_tensors();
    std::vector<lite::Tensor *> outputs = kernel->out_tensors();
    for (auto &output : outputs) {
      output->FreeData();
    }
    auto ret = lite::KernelInferShape(inputs, outputs, kernel->kernel()->primitive(), kernel->Context()->GetProviders(),
                                      schema_version_, kernel->kernel());
    if (ret == lite::RET_NOT_SUPPORT) {
      auto parameter = kernel->op_parameter();
      if (parameter == nullptr) {
        MS_LOG(ERROR) << "kernel(" << kernel->name() << ")'s op_parameter is nullptr!";
        return RET_ERROR;
      }
      // replace with custom op in the future.
      if (parameter->type_ == static_cast<int>(PrimType::PrimType_Inner_Identity)) {
        ret = kernel->ReSize();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "kernel " << kernel->name() << " resize fail!ret = " << ret;
          return ret;
        }
        continue;
      }
      ret = lite::KernelInferShape(inputs, outputs, parameter, context_->allocator);
    }
    if (ret == RET_INFER_INVALID) {
      MS_LOG(DEBUG) << "InferShape shouldn't be done before runtime, type:"
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(kernel->type()))
                    << "flag set to false.";
    } else if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, type: "
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(kernel->type()));
      return RET_INFER_ERR;
    }
    if (ret == RET_OK) {
      ret = kernel->ReSize();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "kernel " << kernel->name() << " resize fail!ret = " << ret;
        return ret;
      }
    }
  }
  return RET_OK;
}

int SubGraphKernel::MallocNodesOutputSpace() {
  for (auto node : nodes_) {
    MS_CHECK_FALSE_MSG(node == nullptr, RET_ERROR, "input kernel is nullptr.");
    MS_CHECK_FALSE_MSG(node->subgraph_type() != kernel::kNotSubGraph, RET_ERROR,
                       "all nodes in should be kernel in subgraph kernels");
    std::vector<lite::Tensor *> outputs = node->out_tensors();
    for (auto &output : outputs) {
      auto ret = lite::MallocTensorData(output);
      if (ret != RET_OK) {
        return ret;
      }
    }
  }
  return RET_OK;
}

int SubGraphKernel::MallocSubgraphInputs() {
  for (auto input : in_tensors()) {
    auto ret = lite::MallocTensorData(input);
    if (ret != RET_OK) {
      return ret;
    }
  }
  return RET_OK;
}

void SubGraphKernel::InitInputTensorInitRefCount() {
  for (auto &input : this->in_tensors()) {
    int input_init_refcount = input->init_ref_count();
    for (auto *node : nodes_) {
      input_init_refcount += std::count_if(node->in_tensors().begin(), node->in_tensors().end(),
                                           [&input](const lite::Tensor *item) { return item == input; });
    }
    input->set_init_ref_count(input_init_refcount);
  }
}

void SubGraphKernel::InitOutTensorInitRefCount(const std::vector<KernelExec *> *mask_kernels) {
  for (auto *node : nodes_) {
    node->InitOutTensorInitRefCount(mask_kernels);
  }
  for (auto &output : this->out_tensors()) {
    if (output->init_ref_count() == 0) {  // true only when output is also an input and only exist in control-flow model
      output->set_init_ref_count(1);
    }
  }
}

int SubGraphKernel::TopologicalSortNodes() {
  in_nodes_ = kernel::KernelExecUtil::SubgraphInputNodes(nodes_);
  auto old_nodes = nodes_;
  nodes_.clear();
  std::queue<KernelExec *> kernel_queue;
  for (auto kernel : in_nodes_) {
    if (std::all_of(kernel->in_kernels().begin(), kernel->in_kernels().end(),
                    [&](KernelExec *in_kernel) { return (!lite::IsContain(old_nodes, in_kernel)); })) {
      kernel_queue.push(kernel);
    }
  }

  while (!kernel_queue.empty()) {
    auto cur_kernel = kernel_queue.front();
    nodes_.emplace_back(cur_kernel);
    kernel_queue.pop();
    CHECK_NULL_RETURN(cur_kernel);
    auto next_kernels = cur_kernel->out_kernels();
    for (auto next_kernel : next_kernels) {
      if (!lite::IsContain(old_nodes, next_kernel)) {
        continue;
      }
      if (lite::IsContain(nodes_, const_cast<KernelExec *>(next_kernel))) {
        MS_LOG(ERROR) << "TopologicalSortKernels failed, loop exist";
        return RET_ERROR;
      }
      auto in_kernels = next_kernel->in_kernels();
      if (std::all_of(in_kernels.begin(), in_kernels.end(), [&](KernelExec *in_kernel) {
            return lite::IsContain(nodes_, in_kernel) || (!lite::IsContain(old_nodes, in_kernel));
          })) {
        kernel_queue.push(next_kernel);
      }
    }
  }
  if (nodes_.size() != old_nodes.size()) {
    MS_LOG(ERROR) << "TopologicalSortKernels failed, kernels size before sort: " << old_nodes.size()
                  << ", kernels size after sort: " << nodes_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

void SubGraphKernel::InsertInEdge(KernelExec *kernel, KernelExec *replace_kernel, const int &tensor_index) {
  // replace_kernel is a kernel with ont input tensor and output tensor
  auto in_kernel = KernelExecUtil::FindInKernelForInTensor(kernel, kernel->in_tensors().at(tensor_index));
  if (in_kernel != nullptr) {
    in_kernel->RemoveOutKernel(kernel);  // Assume there is only one tensor between in_kernel and kernel.
    in_kernel->AddOutKernel(replace_kernel);
    kernel->RemoveInKernel(in_kernel);
    replace_kernel->AddInKernel(in_kernel);
  }
  replace_kernel->AddOutKernel(kernel);
  kernel->AddInKernel(replace_kernel);
  kernel->set_in_tensor(replace_kernel->out_tensors().at(0), tensor_index);

  nodes_.push_back(replace_kernel);
}

void SubGraphKernel::InsertOutEdge(KernelExec *kernel, KernelExec *replace_kernel, const int &tensor_index) {
  // replace_kernel is a kernel with ont input tensor and output tensor
  auto out_kernels = KernelExecUtil::FindOutKernelsForOutTensor(kernel, kernel->out_tensors().at(tensor_index));
  for (const auto &post_kernel : out_kernels) {
    post_kernel->RemoveInKernel(kernel);  // Assume there is only one tensor between kernel and post_kernel.
    post_kernel->AddInKernel(replace_kernel);
    kernel->RemoveOutKernel(post_kernel);
    replace_kernel->AddOutKernel(post_kernel);
  }
  replace_kernel->AddInKernel(kernel);
  kernel->AddOutKernel(replace_kernel);
  kernel->set_out_tensor(replace_kernel->in_tensors().at(0), tensor_index);

  nodes_.push_back(replace_kernel);
}

// in_kernel -> in_post_kernel -> out_pre_kernel -> out_kernels.
// remove in_post_kernel and out_pre_kernel, link in_kernel and out_kernels.
// in_post_kernel and out_pre_kernel can be the same kernel sometimes.
int SubGraphKernel::UpdateInOutKernels(KernelExec *in_kernel, std::vector<KernelExec *> out_kernels,
                                       KernelExec *in_post_kernel, KernelExec *out_pre_kernel) {
  for (const auto &out_kernel : out_kernels) {
    out_kernel->RemoveInKernel(out_pre_kernel);
    out_pre_kernel->RemoveOutKernel(out_kernel);
    if (in_kernel != nullptr) {
      out_kernel->AddInKernel(in_kernel);
      in_kernel->AddOutKernel(out_kernel);
    }
  }

  if (in_post_kernel != out_pre_kernel) {
    in_post_kernel->RemoveOutKernel(out_pre_kernel);
    out_pre_kernel->RemoveInKernel(in_post_kernel);
  }

  if (in_post_kernel->out_kernels().empty() && in_kernel != nullptr && !lite::IsContain(out_nodes_, in_post_kernel)) {
    in_kernel->RemoveOutKernel(in_post_kernel);
    in_post_kernel->RemoveInKernel(in_kernel);
  }

  // update subgraph input node
  if (lite::IsContain(in_nodes_, in_post_kernel)) {
    for (const auto &out_kernel : out_kernels) {
      in_nodes_.push_back(out_kernel);
    }
    if (in_post_kernel->out_kernels().empty() && !lite::IsContain(out_nodes_, in_post_kernel)) {
      lite::VectorErase(&in_nodes_, in_post_kernel);
    }
  }

  // update subgraph output node
  if (lite::IsContain(out_nodes_, out_pre_kernel) && in_kernel != nullptr) {
    out_nodes_.push_back(in_kernel);
    if (out_pre_kernel->in_kernels().empty() && !lite::IsContain(in_nodes_, out_pre_kernel)) {
      lite::VectorErase(&out_nodes_, out_pre_kernel);
    }
  }
  return RET_OK;
}

// Update tensor according to the subgraph.
// Because the model input must be subgraph input, and the model output must be subgraph output.
int SubGraphKernel::UpdateInOutTensors(KernelExec *in_kernel, std::vector<KernelExec *> out_kernels,
                                       lite::Tensor *in_tensor, lite::Tensor *out_tensor, bool keep_input) {
  auto reserve_input = (keep_input && !lite::IsContain(out_tensors(), out_tensor)) ||
                       (!keep_input && lite::IsContain(in_tensors(), in_tensor));
  if (reserve_input) {
    for (const auto &post_kernel : out_kernels) {
      CHECK_NULL_RETURN(post_kernel);
      auto index = post_kernel->FindInTensorIndex(out_tensor);
      post_kernel->set_in_tensor(in_tensor, index);
    }
  } else {
    CHECK_NULL_RETURN(in_kernel);
    auto index = in_kernel->FindOutTensorIndex(in_tensor);
    in_kernel->set_out_tensor(out_tensor, index);

    for (const auto &out_kernel : in_kernel->out_kernels()) {
      if (lite::IsContain(out_kernel->in_tensors(), in_tensor)) {
        auto input_index = out_kernel->FindInTensorIndex(in_tensor);
        out_kernel->set_in_tensor(out_tensor, input_index);
      }
    }
  }
  return RET_OK;
}

// Remove a single way kernel.
// Before removing, pre_kernel -> in_tensor -> kernel -> out_tensor -> post_kernel.
// Keep_input is true, reserve the input tensor: pre_kernel -> in_tensor -> post_kernel.
// Keep_input is false, reserve the output tensor: pre_kernel -> out_tensor -> post_kernel.
int SubGraphKernel::DeleteSingleWayNode(KernelExec *kernel, bool keep_input) {
  if (lite::IsContain(in_nodes_, kernel) && lite::IsContain(out_nodes_, kernel)) {
    MS_LOG(INFO) << "A single kernel subgraph can't delete this kernel.";
    return RET_OK;
  }
  auto in_tensor = kernel->in_tensors().at(0);
  auto out_tensor = kernel->out_tensors().at(0);
  auto in_kernel = KernelExecUtil::FindInKernelForInTensor(kernel, in_tensor);
  auto out_kernels = KernelExecUtil::FindOutKernelsForOutTensor(kernel, out_tensor);
  if (in_kernel == nullptr && out_kernels.empty()) {
    MS_LOG(INFO) << "A single kernel model can't delete this kernel.";
    return RET_OK;
  }

  // update kernel link
  auto ret = UpdateInOutKernels(in_kernel, out_kernels, kernel, kernel);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update kernel link failed when removing kernel " << kernel->name();
    return RET_ERROR;
  }

  // update tensor link
  ret = UpdateInOutTensors(in_kernel, out_kernels, in_tensor, out_tensor, keep_input);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update tensor failed when removing kernel " << kernel->name();
    return RET_ERROR;
  }
  DropNode(kernel);
  delete kernel;
  return RET_OK;
}

void SubGraphKernel::DropNode(KernelExec *node) {
  lite::VectorErase(&nodes_, node);
  lite::VectorErase(&in_nodes_, node);
  lite::VectorErase(&out_nodes_, node);
}

int SubGraphKernel::SubGraphSplitByOperator(KernelsArray *kernels_array) {
  kernels_array->units.clear();
  auto graph_input = this->in_tensors();
  std::vector<KernelExec *> nodes_tmp = nodes_;
  size_t kernels_num = nodes_tmp.size();
  for (size_t kernel_index = 0; kernel_index < kernels_num; kernel_index++) {
    auto kernel = nodes_tmp[kernel_index];
    if (kernel == nullptr) {
      continue;
    }
    MS_CHECK_TRUE_MSG(kernel->subgraph_type() == kernel::kNotSubGraph, RET_ERROR, "node cannot be a subgraph.");
    kernels_array->units.push_back({});
    size_t now_index = kernels_array->units.size() - 1;
    kernels_array->units.at(now_index).kernels.push_back(kernel);
    for (auto in_kernel : kernel->in_kernels()) {
      for (size_t i = 0; i < now_index; i++) {
        if (lite::IsContain(kernels_array->units.at(i).kernels, in_kernel)) {
          kernels_array->units.at(now_index).input_indexs.push_back(i);
          kernels_array->units.at(i).output_indexs.push_back(now_index);
        }
      }
    }
    bool is_graph_input = true;
    for (auto &in_tensor : kernel->in_tensors()) {
      if (!(lite::IsContain(graph_input, in_tensor) || in_tensor->IsGraphInput() || in_tensor->IsConst())) {
        is_graph_input = false;
      }
    }
    if (is_graph_input) {
      if (kernel->in_kernels().size() != 0) {
        MS_LOG(ERROR) << "graph input node in_kernels num invalid!";
        return RET_ERROR;
      }
      kernels_array->graph_input.push_back(now_index);
    } else if (kernel->in_kernels().size() == 0) {
      MS_LOG(ERROR) << "graph input node invalid!";
      return RET_ERROR;
    }
    MS_ASSERT(std::find_if(kernel->in_kernels().begin(), kernel->in_kernels().end(), [kernel](KernelExec *in_kernel) {
                return !lite::IsContain(in_kernel->out_kernels(), kernel);
              }) == kernel->in_kernels().end());
    MS_ASSERT(
      std::find_if(kernel->out_kernels().begin(), kernel->out_kernels().end(), [kernel](KernelExec *out_kernel) {
        return !lite::IsContain(out_kernel->in_kernels(), kernel);
      }) == kernel->out_kernels().end());
    while ((kernel->out_kernels().size() == 1) && (kernel->out_kernels().front()->in_kernels().size() == 1)) {
      kernel = kernel->out_kernels().front();
      size_t i;
      for (i = kernel_index + 1; i < kernels_num; i++) {
        if (nodes_tmp[i] == kernel) {
          break;
        }
      }
      if (i < kernels_num) {
        nodes_tmp[i] = nullptr;
      } else {
        MS_LOG(ERROR) << "graph structure invalid!";
        return RET_ERROR;
      }
      kernels_array->units.at(now_index).kernels.push_back(kernel);
    }
  }
  return RET_OK;
}

int CustomSubGraph::Prepare() {
  auto ret = SubGraphKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  if (nodes_.size() < 1) {
    return RET_OK;
  }
  auto provider = nodes_[0]->desc().provider;
  auto context = this->Context();
  AllocatorPtr allocator = context->allocator;
  auto iter = std::find_if(context->device_list_.begin(), context->device_list_.end(),
                           [&provider](const auto &dev) { return dev.provider_ == provider; });
  if (iter != context->device_list_.end()) {
    allocator = iter->allocator_;
  }

  for (size_t i = 0; i < nodes_.size() - 1; ++i) {
    auto node = nodes_[i];
    for (auto tensor : node->out_tensors()) {
      MS_ASSERT(tensor != nullptr);
      if (tensor->allocator() == nullptr) {
        tensor->set_allocator(allocator);
      }
    }
  }

  auto node = nodes_[nodes_.size() - 1];
  for (auto tensor : node->out_tensors()) {
    MS_ASSERT(tensor != nullptr);
    if (tensor->allocator() == nullptr) {
      tensor->set_allocator(context->allocator);
    }
  }
  return RET_OK;
}

int CustomSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  for (auto kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
  }

  return RET_OK;
}

int CpuSubGraph::Prepare() {
  auto ret = SubGraphKernel::Prepare();
  if (ret != RET_OK) {
    return ret;
  }
  for (auto node : nodes_) {
    for (auto tensor : node->out_tensors()) {
      MS_ASSERT(tensor != nullptr);
      if (tensor->allocator() == nullptr) {
        tensor->set_allocator(this->Context()->allocator);
      }
    }
  }
  for (auto &out : this->out_tensors()) {
    if (out->allocator() == nullptr) {
      out->set_allocator(this->Context()->allocator);
    }
  }
  return RET_OK;
}

int CpuSubGraph::Execute(const KernelCallBack &before, const KernelCallBack &after) {
  MS_ASSERT(this->Context()->allocator.get() != nullptr);

  for (auto *kernel : nodes_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

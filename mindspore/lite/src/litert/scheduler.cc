/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/scheduler.h"
#include <map>
#include <unordered_set>
#include <queue>
#include <string>
#include <vector>
#include <algorithm>
#include "src/tensorlist.h"
#include "nnacl/partial_fusion_parameter.h"
#include "include/errorcode.h"
#include "src/common/graph_util.h"
#include "src/common/utils.h"
#include "src/litert/kernel_registry.h"
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
#include "include/registry/register_kernel.h"
#endif
#include "src/litert/kernel_exec_util.h"
#include "src/litert/sub_graph_kernel.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/common/version_manager.h"
#include "src/common/prim_util.h"
#include "src/litert/lite_model.h"
#include "src/common/tensor_util.h"
#include "src/common/context_util.h"
#include "src/litert/infer_manager.h"
#include "src/litert/runtime_pass.h"
#include "src/litert/pass/format_pass/format_pass.h"
#if !defined(AUTO_PARALLEL_CLIP) || !defined(RUNTIME_PASS_CLIP)
#include "src/litert/sub_graph_split.h"
#include "src/litert/pass/online_fusion/online_fusion_pass_registry.h"
#endif
#include "src/litert/weight_decoder.h"
#include "src/litert/kernel/cpu/fp16/fp16_op_handler.h"
#include "nnacl/nnacl_common.h"
#if GPU_OPENCL
#include "src/litert/kernel/opencl/opencl_subgraph.h"
#include "src/litert/kernel/gpu/opencl/opencl_runtime.h"
#endif
#include "include/registry/register_kernel_interface.h"
#include "extendrt/mindir_loader/abstract_base_model.h"
#include "src/litert/pack_weight_manager.h"
#if defined(PARALLEL_INFERENCE) && defined(ENABLE_MINDRT)
#include "thread/parallel_thread_pool_manager.h"
#endif

using AbstractBaseModel = mindspore::infer::AbstractBaseModel;

namespace mindspore::lite {
namespace {
constexpr int kMainSubGraphIndex = 0;
}  // namespace

namespace {
// support_fp16: current device and package support float16
int CastKernelWeight(const kernel::SubGraphType &belong_subgraph_type, const kernel::KernelExec *kernel,
                     bool support_fp16) {
  MS_ASSERT(kernel != nullptr);
  MS_ASSERT(kernel->subgraph_type() == kernel::kNotSubGraph);
  if (belong_subgraph_type != kernel::kCpuFP32SubGraph && belong_subgraph_type != kernel::kCpuFP16SubGraph) {
    return RET_OK;
  }
  for (auto *tensor : kernel->in_tensors()) {
    MS_ASSERT(tensor != nullptr);
    // only cast const tensor
    // tensorlist not support fp16 now
    if (!tensor->IsConst() || tensor->data_type() == kObjectTypeTensorType) {
      continue;
    }
    // only support fp32->fp16 or fp16->fp32
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      continue;
    }
    if (tensor->data_type() == kNumberTypeFloat32 && belong_subgraph_type == kernel::kCpuFP16SubGraph) {
      auto ret = CastConstTensorData(tensor, kNumberTypeFloat16, support_fp16);
      if (ret != RET_OK) {
        MS_LOG(DEBUG) << "Cast const tensor from fp32 to fp16 failed, tensor name : " << tensor->tensor_name();
        return ret;
      }
    } else if (tensor->data_type() == kNumberTypeFloat16 && belong_subgraph_type == kernel::kCpuFP32SubGraph) {
      auto ret = CastConstTensorData(tensor, kNumberTypeFloat32, support_fp16);
      if (ret != RET_OK) {
        MS_LOG(DEBUG) << "Cast const tensor from fp16 to fp32 failed, tensor name : " << tensor->tensor_name();
        return ret;
      }
    } else {
      MS_LOG(DEBUG) << "No need to cast";
    }
  }
  return RET_OK;
}

int CopyConstTensorData(const std::vector<Tensor *> &tensors, int op_type) {
  // packed kernels such as conv don't need to copy because weight will be packed in kernel
  if (lite::PackWeightManager::GetInstance()->IsCopyTensor(op_type)) {
    return RET_OK;
  }

  for (auto *tensor : tensors) {
    // only copy non-copied const tensor
    if (!tensor->IsConst() && tensor->data() != nullptr) {
      MS_LOG(ERROR) << "Illegitimate tensor : " << tensor->tensor_name();
      return RET_ERROR;
    }
    if (!tensor->IsConst() || tensor->own_data()) {
      continue;
    }
    if (tensor->data_type() == kObjectTypeTensorType) {
      // tensorlist's data is nullptr since ConvertTensors
      // we never set or malloc data of tensorlist but malloc tensors in tensorlist
      MS_ASSERT(tensor->data() == nullptr);
    } else {
      auto copy_tensor = Tensor::CopyTensor(*tensor, true);
      if (copy_tensor == nullptr) {
        MS_LOG(ERROR) << "Copy tensor failed";
        return RET_ERROR;
      }
      tensor->FreeData();
      tensor->set_data(copy_tensor->data());
      tensor->set_own_data(true);
      copy_tensor->set_data(nullptr);
      delete (copy_tensor);
    }
  }
  return RET_OK;
}
}  // namespace

// support_fp16: current device and package support float16
int Scheduler::HandleBuildinCpuKernelWeight(const kernel::SubGraphType &belong_subgraph_type,
                                            const kernel::KernelExec *kernel) {
  MS_ASSERT(kernel != nullptr);
  MS_ASSERT(kernel->subgraph_type() == kernel::kNotSubGraph);
  if (is_train_session_ || kernel->type() == schema::PrimitiveType_Custom ||
      kernel->desc().provider != kernel::kBuiltin) {
    return RET_OK;
  }
  auto ret = CastKernelWeight(belong_subgraph_type, kernel, context_->device_and_pkg_support_fp16_);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "CastKernelWeight failed: " << ret;
    return RET_NOT_SUPPORT;
  }
  if (!(reinterpret_cast<LiteModel *>(src_model_)->keep_model_buf())) {
    // we don't need to restore tensor for copy data
    MS_CHECK_TRUE_RET(kernel->op_parameter() != nullptr, RET_ERROR);
    ret = CopyConstTensorData(kernel->in_tensors(), kernel->op_parameter()->type_);
    if (ret != RET_OK) {
      MS_LOG(DEBUG) << "CopyConstTensorsData failed: " << ret;
      return RET_NOT_SUPPORT;
    }
  }
  return RET_OK;
}

int Scheduler::InitKernels(std::vector<kernel::KernelExec *> &&dst_kernels) {
  if (is_train_session_) {
    return RET_OK;
  }
  for (auto kernel : dst_kernels) {
    // delegate graph kernel
    if (kernel->desc().arch == kernel::kDelegate) {
      continue;
    }
    auto subgraph_type = kernel->subgraph_type();
    if (subgraph_type == kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "construct subgraph failed.";
      return RET_ERROR;
    }
    auto subgraph_nodes = reinterpret_cast<kernel::SubGraphKernel *>(kernel)->nodes();
    for (auto node : subgraph_nodes) {
      for (auto *tensor : node->out_tensors()) {
        if (tensor->IsConst()) {
          MS_CHECK_TRUE_MSG(node->op_parameter() != nullptr, RET_NULL_PTR, "node's op_parameter is invalid.");
          if (node->op_parameter()->type_ == ::PrimType::PrimType_Inner_ShapeFusion) {
            continue;
          }
          MS_LOG(ERROR) << "Illegitimate kernel output tensor : " << tensor->tensor_name();
          continue;
        }
      }
      auto ret = HandleBuildinCpuKernelWeight(subgraph_type, node);
      if (ret != RET_OK) {
        return ret;
      }
    }
#if GPU_OPENCL
    if (kernel->desc().arch == kernel::kGPU) {
      if (this->GetEnableGLTexture() == true && (kernel == dst_kernels.front() || kernel == dst_kernels.back() - 1)) {
        kernel->SetOpenGLTextureEnable(true);
        MS_LOG(INFO) << "Set OpenGLSharingMem for subgraph success!" << std::endl;
      }
      auto ret = reinterpret_cast<kernel::OpenCLSubGraph *>(kernel)->RunPass();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "OpenCLSubGraph RunPass failed.";
        return ret;
      }
    }
#endif
  }
  return RET_OK;
}

int Scheduler::SchedulePreProcess() {
  schema_version_ = reinterpret_cast<LiteModel *>(src_model_)->GetSchemaVersion();

#if !defined(AUTO_PARALLEL_CLIP) || !defined(RUNTIME_PASS_CLIP)
  auto search_sub_graph =
    SearchSubGraph(context_, src_model_, src_tensors_, &op_parameters_, &graph_output_node_indexes_);
#endif

#if !defined(RUNTIME_PASS_CLIP)
  OnlineFusionRegistry::GetInstance()->DoOnlineFusionPass(&search_sub_graph);
#endif

  this->graph_output_node_indexes_ = GetGraphOutputNodes(src_model_);

  if (src_model_->model_type_ != ModelType_MSLite) {
    // call abstract model infer interface
    *is_infershape_ = RET_OK;
  } else {
    *is_infershape_ = InferSubGraphShape(kMainSubGraphIndex);
  }
  if (*is_infershape_ != RET_OK && *is_infershape_ != RET_INFER_INVALID) {
    MS_LOG(ERROR) << "op infer shape failed.";
    return *is_infershape_;
  }

  if (context_->enable_parallel_) {
#ifndef AUTO_PARALLEL_CLIP
    if (*is_infershape_ != RET_INFER_INVALID) {
      search_sub_graph.SubGraphSplit();
    }
#else
    MS_LOG(ERROR) << unsupport_auto_parallel_log;
    return RET_NOT_SUPPORT;
#endif
  }
  return RET_OK;
}

int Scheduler::CheckCpuValid(const std::vector<kernel::KernelExec *> *dst_kernels) const {
  if (context_->IsDeviceTypeEnabled(DT_CPU)) {
    return RET_OK;
  }
  for (auto kernel : *dst_kernels) {
    if (kernel->desc().arch == kernel::KERNEL_ARCH::kCPU) {
      MS_LOG(ERROR) << "kernel: " << kernel->name() << " only support in CPU.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Scheduler::ConstructSubGraphs(std::vector<kernel::KernelExec *> *dst_kernels) {
  if (*is_control_flow_) {
    return ConstructControlFlowMainGraph(dst_kernels);
  }

  auto src_kernel = *dst_kernels;
  dst_kernels->clear();
  std::map<const kernel::KernelExec *, bool> is_kernel_finish;
  return ConstructNormalSubGraphs(src_kernel, dst_kernels, &is_kernel_finish);
}

STATUS Scheduler::DelQuantDTypeCastKernel(std::vector<kernel::KernelExec *> *kernels) {
  for (auto iter = (*kernels).begin(); iter != (*kernels).end();) {
    auto cur_kernel = *iter;
    if (cur_kernel->subgraph_type() != kernel::kNotSubGraph) {
      auto sub_inner_graph = reinterpret_cast<kernel::SubGraphKernel *>(cur_kernel);
      auto &subgraph_nodes = sub_inner_graph->nodes();
      if (DelQuantDTypeCastKernel(&subgraph_nodes) != RET_OK) {
        MS_LOG(ERROR) << "DeleteRedundantTrans failed in subgraph.";
        return RET_ERROR;
      }
    }
    if (cur_kernel->type() != schema::PrimitiveType_QuantDTypeCast) {
      iter++;
      continue;
    }
    auto &post_kernels = cur_kernel->out_kernels();
    auto &pre_kernels = cur_kernel->in_kernels();
    if (cur_kernel->in_tensors().size() != 1) {
      MS_LOG(ERROR) << cur_kernel->name() << " input size error."
                    << " cur_kernel in tensors size:" << cur_kernel->in_tensors().size();
      return RET_ERROR;
    }
    bool graph_input = pre_kernels.empty();
    if (!graph_input) {
      // modify post kernel input to new kernel and new tensor
      for (auto post_kernel : post_kernels) {
        auto post_in_kernels = post_kernel->in_kernels();
        auto post_input_iter = std::find(post_in_kernels.begin(), post_in_kernels.end(), cur_kernel);
        *post_input_iter = pre_kernels[0];
        post_kernel->set_in_tensor(cur_kernel->in_tensors()[0], post_input_iter - post_in_kernels.begin());
        post_kernel->set_in_kernels(post_in_kernels);
      }
      auto pre_out_kernels = pre_kernels[0]->out_kernels();
      auto pre_out_iter = std::find(pre_out_kernels.begin(), pre_out_kernels.end(), cur_kernel);
      if (pre_out_iter != pre_out_kernels.end()) {
        pre_out_kernels.erase(pre_out_iter);
        pre_out_kernels.insert(pre_out_iter, post_kernels.begin(), post_kernels.end());
        pre_kernels[0]->set_out_kernels(pre_kernels);
      }
    } else {
      for (auto post_kernel : post_kernels) {
        auto post_in_kernels = post_kernel->in_kernels();
        auto post_input_iter = std::find(post_in_kernels.begin(), post_in_kernels.end(), cur_kernel);
        *post_input_iter = {};
        post_kernel->set_in_tensor(cur_kernel->in_tensors()[0], post_input_iter - post_in_kernels.begin());
        post_kernel->set_in_kernels(post_in_kernels);
      }
    }

    // update data type
    for (auto tensor : cur_kernel->in_tensors()) {
      tensor->set_data_type(kNumberTypeFloat32);
    }
    for (auto tensor : cur_kernel->out_tensors()) {
      tensor->set_data_type(kNumberTypeFloat32);
    }

    // update model output kernel & tensor
    if (cur_kernel->is_model_output()) {
      pre_kernels[0]->set_is_model_output(true);
      cur_kernel->in_tensors()[0]->set_category(Category::GRAPH_OUTPUT);
      pre_kernels[0]->set_out_kernels({});
      // If the current kernel is the output kernel, use the current output tensor as the output tensor of the previous
      // node.
      auto pre_out_tensors = pre_kernels[0]->out_tensors();
      auto tensor_iter = std::find(pre_out_tensors.begin(), pre_out_tensors.end(), cur_kernel->in_tensors()[0]);
      if (tensor_iter != pre_kernels[0]->out_tensors().end()) {
        *tensor_iter = cur_kernel->out_tensors()[0];
      }
    }

    // delete cur kernel
    iter = kernels->erase(iter);
    MS_LOG(DEBUG) << "Delete kernel: " << cur_kernel->name();
    delete cur_kernel;
  }
  return RET_OK;
}

int Scheduler::Schedule(std::vector<kernel::KernelExec *> *dst_kernels) {
  int check_input_ret = CheckInputParam(dst_kernels);
  if (check_input_ret != RET_OK) {
    MS_LOG(ERROR) << "CheckInputParam failed! ret: " << check_input_ret;
    return check_input_ret;
  }

  shape_fusion_pass_ =
    std::make_shared<ShapeFusionPass>(context_, reinterpret_cast<LiteModel *>(src_model_), src_tensors_);
  MS_CHECK_TRUE_RET(shape_fusion_pass_ != nullptr, RET_ERROR);
  int ret = SchedulePreProcess();
  if (ret != RET_OK) {
    return ret;
  }

  if (*is_control_flow_) {
    control_flow_scheduler_ = std::make_shared<ControlFlowScheduler>(context_, ms_context_, src_tensors_);
    MS_CHECK_TRUE_MSG(control_flow_scheduler_ != nullptr, RET_ERROR, "new control scheduler failed.");
  }

  ret = ScheduleGraphToKernels(dst_kernels);
  FreeOpParameters();
  op_parameters_.clear();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule graph to kernels failed.";
    return ret;
  }
  if (context_->float_mode) {
    kernel::KernelExecUtil::FindAllInoutKernels(*dst_kernels);
    ret = DelQuantDTypeCastKernel(dst_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Delete quant_dtype_cast kernel failed.";
      return ret;
    }
  }
  shape_fusion_pass_->StoreStateAndReset();

  ret = InitDelegateKernels(dst_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Repalce delegate kernels failed.";
    return ret;
  }

  ret = CheckCpuValid(dst_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "kernels invalid in set devices.";
    return ret;
  }

  kernel::KernelExecUtil::FindAllInoutKernels(*dst_kernels);

  ret = ConstructSubGraphs(dst_kernels);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstructSubGraphs failed.";
    return ret;
  }

  if (*is_control_flow_) {
    control_flow_scheduler_->SetSubgraphForPartialNode(&partial_kernel_subgraph_index_map_,
                                                       &subgraph_index_subgraph_kernel_map_);
    ret = control_flow_scheduler_->Schedule(dst_kernels);
    MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "control flow schedule failed.");
  }

  auto status = RuntimePass(dst_kernels, src_tensors_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "runtime pass failed.";
    return RET_ERROR;
  }

  ret = InitKernels(std::move(*dst_kernels));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitKernels failed.";
    return ret;
  }
  shape_fusion_pass_->RestoreState();
  if (IsPrintDebug()) {
    MS_LOG(DEBUG) << "schedule kernels success.";
    for (auto subgraph : *dst_kernels) {
      MS_LOG(DEBUG) << "[subgraph] : " << subgraph->name() << ",  type:" << subgraph->subgraph_type();
      if (subgraph->desc().arch == kernel::KERNEL_ARCH::kDelegate) {
        continue;
      }
      std::vector<kernel ::KernelExec *> kernel_list = reinterpret_cast<kernel::SubGraphKernel *>(subgraph)->nodes();
      for (auto kernel : kernel_list) {
        MS_LOG(DEBUG) << "kernel: [" << kernel->name() << "] "
                      << "TypeId(" << kernel->desc().data_type << "); "
                      << "OpType(" << PrimitiveCurVersionTypeName(kernel->desc().type) << "); "
                      << "format(" << kernel->desc().format << "); "
                      << "arch(" << kernel->desc().arch << ")";
      }
    }
  }
  return RET_OK;
}

int Scheduler::CheckInputParam(const std::vector<kernel::KernelExec *> *dst_kernels) const {
  if (dst_kernels == nullptr) {
    return RET_ERROR;
  }
  if (src_model_ == nullptr) {
    MS_LOG(ERROR) << "Input model is nullptr";
    return RET_PARAM_INVALID;
  }
  if (src_model_->graph_.sub_graphs_.empty()) {
    MS_LOG(ERROR) << "Model should have a subgraph at least";
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

#ifndef DELEGATE_CLIP
int Scheduler::ReplaceDelegateKernels(std::vector<kernel::KernelExec *> *dst_kernels) {
  std::vector<kernel::Kernel *> kernels;
  for (size_t i = 0; i < dst_kernels->size(); i++) {
    kernels.push_back((*dst_kernels)[i]->kernel());
  }

  ms_inputs_ = LiteTensorsToMSTensors(*inputs_);
  ms_outputs_ = LiteTensorsToMSTensors(*outputs_);
  auto schema_version = static_cast<SchemaVersion>(schema_version_);
  DelegateModel<schema::Primitive> *model =
    new (std::nothrow) DelegateModel<schema::Primitive>(&kernels, ms_inputs_, ms_outputs_, primitives_, schema_version);
  if (model == nullptr) {
    MS_LOG(ERROR) << "New delegate model failed.";
    return RET_NULL_PTR;
  }
  auto ret = delegate_->Build(model);
  if (ret != mindspore::kSuccess) {
    delete model;
    MS_LOG(ERROR) << "Delegate prepare kernels failed.";
    return RET_ERROR;
  }

  auto src_kernels = *dst_kernels;
  dst_kernels->clear();
  std::map<const kernel::KernelExec *, bool> delegate_support;
  for (auto kernel : src_kernels) {
    delegate_support[kernel] = true;
  }
  for (auto kernel : kernels) {
    size_t index = 0;
    for (; index < src_kernels.size(); index++) {
      if (kernel == src_kernels[index]->kernel()) {
        // Kernels that the delegate does not support keep the original backend
        dst_kernels->push_back(src_kernels[index]);
        delegate_support[src_kernels[index]] = false;
        break;
      }
    }
    if (index == src_kernels.size()) {
      // New liteKernel to save delegate subgraph
      std::shared_ptr<kernel::Kernel> shared_kernel(kernel);
      auto kernel_exec = new (std::nothrow) kernel::KernelExec(shared_kernel);
      if (kernel_exec == nullptr) {
        delete model;
        MS_LOG(ERROR) << "New KernelExec for delegate subgraph failed.";
        return RET_NULL_PTR;
      }
      auto delegate_type = kNumberTypeFloat32;
      for (auto &input : kernel->inputs()) {
        if (static_cast<TypeId>(input.DataType()) == kNumberTypeFloat16) {
          delegate_type = kNumberTypeFloat16;
          break;
        }
      }
      kernel::KernelKey delegate_desc{kernel::kDelegate, delegate_type, NHWC, schema::PrimitiveType_NONE, "", ""};
      kernel_exec->set_desc(delegate_desc);
      dst_kernels->push_back(kernel_exec);
    }
  }
  // Release the cpu kernel that has been replace by delegate subgraph, as well as their tensor data
  for (auto kernel : src_kernels) {
    if (delegate_support[kernel] == true) {
      auto inputs = kernel->in_tensors();
      for (auto *tensor : inputs) {
        MS_ASSERT(tensor != nullptr);
        if (tensor->IsConst()) {
          tensor->FreeData();
        }
      }
      delete kernel;
    }
  }
  delete model;
  return RET_OK;
}

int Scheduler::InitDelegateKernels(std::vector<kernel::KernelExec *> *dst_kernels) {
  /* no delegate valid */
  if (delegate_ == nullptr) {
    return RET_OK;
  }

  /* set delegate spin count */
  context_->thread_pool_->SetSpinCountMinValue();

  /* external delegate */
  if (delegate_device_type_ == -1) {
    auto ret = ReplaceDelegateKernels(dst_kernels);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "external delegate init failed.";
      return ret;
    }
  }

  /* Inner delegate  :  check Priority */
  std::vector<kernel::KernelExec *> src_kernels = *dst_kernels;
  dst_kernels->clear();

  while (!src_kernels.empty()) {
    std::vector<kernel::KernelExec *> tmp_kernels;
    kernel::KernelExec *remain_kernel = nullptr;

    /* Loop for inner delegate npu and TensorRT subgraph */
    while (!src_kernels.empty()) {
      auto kernel = src_kernels.front();
      VectorErase(&src_kernels, kernel);
      bool priority_ret =
        DeviceTypePriority(context_, delegate_device_type_, KernelArchToDeviceType(kernel->desc().arch));
      if (priority_ret == true) {
        tmp_kernels.push_back(kernel);
      } else {
        remain_kernel = kernel;
        break;
      }
    }

    /* start current NPU-kernels replace */
    if (tmp_kernels.empty()) {
      if (remain_kernel != nullptr) {
        dst_kernels->push_back(remain_kernel);
        remain_kernel = nullptr;
      }
      continue;
    }
    auto ret = ReplaceDelegateKernels(&tmp_kernels);
    if (ret != RET_OK) {
      dst_kernels->insert(dst_kernels->end(), src_kernels.begin(), src_kernels.end());
      dst_kernels->insert(dst_kernels->end(), tmp_kernels.begin(), tmp_kernels.end());
      if (remain_kernel != nullptr) {
        dst_kernels->push_back(remain_kernel);
      }
      MS_LOG(ERROR) << "Inner delegate replace delegate kernels failed.";
      return ret;
    }

    dst_kernels->insert(dst_kernels->end(), tmp_kernels.begin(), tmp_kernels.end());
    tmp_kernels.clear();
    if (remain_kernel != nullptr) {
      dst_kernels->push_back(remain_kernel);
      remain_kernel = nullptr;
    }
  }

  return RET_OK;
}
#endif

void Scheduler::FindNodeInoutTensors(const lite::LiteGraph::Node &node, std::vector<Tensor *> *inputs,
                                     std::vector<Tensor *> *outputs) {
  MS_ASSERT(inputs != nullptr);
  MS_ASSERT(outputs != nullptr);
  auto in_size = node.input_indices_.size();
  inputs->reserve(in_size);
  for (size_t j = 0; j < in_size; ++j) {
    inputs->emplace_back(src_tensors_->at(node.input_indices_[j]));
  }
  auto out_size = node.output_indices_.size();
  outputs->reserve(out_size);
  for (size_t j = 0; j < out_size; ++j) {
    outputs->emplace_back(src_tensors_->at(node.output_indices_[j]));
  }
}

int Scheduler::InferNodeShape(const lite::LiteGraph::Node *node) {
  MS_ASSERT(node != nullptr);
  auto primitive = node->primitive_;
  MS_ASSERT(primitive != nullptr);
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  FindNodeInoutTensors(*node, &inputs, &outputs);
  auto ret = KernelInferShape(inputs, outputs, node->primitive_, context_->GetProviders(), schema_version_);
  if (ret != RET_NOT_SUPPORT) {
    *infer_along_running_ = false;
    return ret;
  }

  auto parame_gen = PopulateRegistry::GetInstance()->GetParameterCreator(
    GetPrimitiveType(node->primitive_, schema_version_), schema_version_);
  if (parame_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr.";
    FreeOpParameters();
    return RET_NULL_PTR;
  }
  auto parameter = parame_gen(primitive);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << GetPrimitiveTypeName(primitive, schema_version_);
    FreeOpParameters();
    return RET_ERROR;
  }

  parameter->quant_type_ = node->quant_type_;
  parameter->thread_num_ = context_->thread_num_;
  if (node->output_indices_.empty()) {
    MS_LOG(ERROR) << "The output size is invalid";
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    return RET_ERROR;
  }
  if (op_parameters_.find(node->output_indices_.at(0)) != op_parameters_.end()) {
    if (parameter->destroy_func_ != nullptr) {
      parameter->destroy_func_(parameter);
    }
    free(parameter);
    parameter = op_parameters_[node->output_indices_.at(0)];
  } else {
    op_parameters_[node->output_indices_.at(0)] = parameter;
  }

  if (IsCallNode(primitive, schema_version_)) {
    return InferCallShape(node);
  }
  ret = KernelInferShape(inputs, outputs, parameter, context_->allocator);
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    FreeOpParameters();
    return RET_ERROR;
  }
  for (auto &output : outputs) {
    output->set_shape_changed(false);
  }
  if (*is_control_flow_) {
    for (auto &output : outputs) {
      output->set_shape({-1});
    }
    return RET_INFER_INVALID;
  }

  if (ret == RET_OK) {
    for (auto &output : outputs) {
      if (static_cast<size_t>(output->ElementsNum()) >= GetMaxMallocSize() / sizeof(int64_t)) {
        MS_LOG(ERROR) << "The size of output tensor is too big";
        FreeOpParameters();
        return RET_ERROR;
      }
    }
  } else if (ret != RET_INFER_INVALID) {
    FreeOpParameters();
    return RET_ERROR;
  }
  return ret;
}

void Scheduler::FreeOpParameters() {
  for (auto &param : op_parameters_) {
    if (param.second != nullptr) {
      if (param.second->destroy_func_ != nullptr) {
        param.second->destroy_func_(param.second);
      }
      free(param.second);
      param.second = nullptr;
    }
  }
}

int Scheduler::RestoreSubGraphInput(const lite::LiteGraph::Node *partial_node) {
  auto subgraph_index = GetPartialGraphIndex(partial_node->primitive_, schema_version_);
  MS_CHECK_TRUE_MSG(subgraph_index >= 0, RET_NULL_PTR, "subgraph index is negative.");
  auto subgraph = src_model_->graph_.sub_graphs_.at(subgraph_index);
  for (size_t i = 0; i < subgraph->input_indices_.size(); ++i) {
    auto &subgraph_input = src_tensors_->at(subgraph->input_indices_[i]);
    subgraph_input->set_data(nullptr);
  }
  return RET_OK;
}

void CopyCommonTensor(Tensor *dst_tensor, Tensor *src_tensor) {
  dst_tensor->set_data_type(src_tensor->data_type());
  dst_tensor->set_shape(src_tensor->shape());
  dst_tensor->set_format(src_tensor->format());
  dst_tensor->set_data(src_tensor->data());
}

int Scheduler::CopyPartialShapeToSubGraph(const lite::LiteGraph::Node *partial_node) {
  auto subgraph_index = GetPartialGraphIndex(partial_node->primitive_, schema_version_);
  MS_CHECK_TRUE_MSG(subgraph_index >= 0, RET_NULL_PTR, "subgraph index is negative.");
  auto subgraph = src_model_->graph_.sub_graphs_.at(subgraph_index);
  if (subgraph->input_indices_.size() != partial_node->input_indices_.size()) {
    MS_LOG(ERROR) << "partial node " << partial_node->name_ << " inputs size: " << partial_node->input_indices_.size()
                  << " vs "
                  << " subgraph input size: " << subgraph->input_indices_.size();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < partial_node->input_indices_.size(); ++i) {
    auto &subgraph_input = src_tensors_->at(subgraph->input_indices_[i]);
    auto &partial_input = src_tensors_->at(partial_node->input_indices_[i]);
    if (partial_input->data_type() == kObjectTypeTensorType) {
      return RET_INFER_INVALID;
    }
    CopyCommonTensor(subgraph_input, partial_input);
  }

  return RET_OK;
}

int Scheduler::InferPartialShape(const lite::LiteGraph::Node *node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(node != nullptr);
  if (!IsPartialNode(node->primitive_, schema_version_)) {
    MS_LOG(ERROR) << "Node is not a partial";
    return RET_PARAM_INVALID;
  }
  CopyPartialShapeToSubGraph(node);
  int subgraph_index = GetPartialGraphIndex(node->primitive_, schema_version_);
  auto ret = InferSubGraphShape(subgraph_index);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "infer subgraph: " << subgraph_index << " failed, ret:" << ret;
  }
  RestoreSubGraphInput(node);
  return ret;
}

LiteGraph::Node *Scheduler::NodeInputIsPartial(const lite::LiteGraph::Node *node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(node != nullptr);
  for (auto &iter : src_model_->graph_.all_nodes_) {
    if (iter->output_indices_ == node->input_indices_) {
      if (IsPartialNode(iter->primitive_, schema_version_)) {
        return iter;
      } else {
        return nullptr;
      }
    }
  }
  return nullptr;
}

int Scheduler::InferCallShape(const lite::LiteGraph::Node *node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(node != nullptr);
  if (!IsCallNode(node->primitive_, schema_version_)) {
    MS_LOG(ERROR) << "Node is not a call cnode";
    return RET_PARAM_INVALID;
  }

  auto partial_input = NodeInputIsPartial(node);
  if (partial_input) {
    return InferPartialShape(partial_input);
  }
  auto switch_input = NodeInputIsSwitchType(node);
  if (switch_input) {
    *is_control_flow_ = true;
    return InferSwitchShape(switch_input);
  }

  MS_LOG(ERROR) << "call input is not partial and also not switch.";
  return RET_ERROR;
}

int Scheduler::InferSubGraphShape(size_t subgraph_index) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(!src_model_->graph_.sub_graphs_.empty());
  MS_ASSERT(src_model_->graph_.sub_graphs_.size() > subgraph_index);
  if (find(infer_subgraph_index_.begin(), infer_subgraph_index_.end(), subgraph_index) != infer_subgraph_index_.end()) {
    MS_LOG(ERROR) << "The subgraph has been infer shape, subgraph index: " << subgraph_index;
    return RET_INFER_INVALID;
  }
  infer_subgraph_index_.push_back(subgraph_index);
  auto subgraph = src_model_->graph_.sub_graphs_.at(subgraph_index);
  int subgraph_infershape_ret = RET_OK;
  for (auto node_index : subgraph->node_indices_) {
    auto node = src_model_->graph_.all_nodes_[node_index];
    MS_ASSERT(node != nullptr);
    auto *primitive = node->primitive_;
    if (primitive == nullptr) {
      MS_LOG(ERROR) << "Op " << node->name_ << " should exist in model!";
      return RET_ERROR;
    }
    if (node->node_type_ == schema::PrimitiveType_Shape) {
      // convert shape to built-in shape
      MS_CHECK_TRUE_RET(node->input_indices_.size() == 1, RET_ERROR);
      shape_fusion_pass_->Run(node, subgraph_index);
    }
    auto ret = InferNodeShape(node);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape interrupted, name: " << node->name_
                   << ", type: " << GetPrimitiveTypeName(primitive, schema_version_) << ", set infer flag to false.";
      subgraph_infershape_ret = RET_INFER_INVALID;
    } else if (ret != RET_OK) {
      FreeOpParameters();
      MS_LOG(ERROR) << "InferShape failed, name: " << node->name_
                    << ", type: " << GetPrimitiveTypeName(primitive, schema_version_);
      return RET_INFER_ERR;
    }
  }
  return subgraph_infershape_ret;
}

namespace {
// support_fp16: current device and package support float16
int CastAndRestoreConstTensorData(Tensor *tensor, std::map<Tensor *, Tensor *> *restored_origin_tensors,
                                  TypeId dst_data_type, bool support_fp16) {
  MS_ASSERT(tensor != nullptr);
  MS_ASSERT(tensor->IsConst());
  MS_ASSERT(tensor->data_type() == kNumberTypeFloat32 || tensor->data_type() == kNumberTypeFloat16);
  MS_ASSERT(dst_data_type == kNumberTypeFloat32 || dst_data_type == kNumberTypeFloat16);
  if (tensor->data_type() == dst_data_type) {
    return RET_OK;
  }
  auto origin_data = tensor->data();
  MS_ASSERT(origin_data != nullptr);
  auto restore_tensor = Tensor::CopyTensor(*tensor, false);
  if (restore_tensor == nullptr) {
    return RET_NULL_PTR;
  }
  restore_tensor->set_data(origin_data);
  restore_tensor->set_own_data(tensor->own_data());
  tensor->set_data(nullptr);
  tensor->set_data_type(dst_data_type);
  auto ret = tensor->MallocData();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "malloc data failed";
    return ret;
  }
  auto new_tensor_data = tensor->data();
  MS_ASSERT(new_tensor_data != nullptr);
  if (dst_data_type == kNumberTypeFloat32) {
    Float16ToFloat32_fp16_handler(origin_data, new_tensor_data, tensor->ElementsNum(), support_fp16);
  } else {  // dst_data_type == kNumberTypeFloat16
    Float32ToFloat16_fp16_handler(origin_data, new_tensor_data, tensor->ElementsNum(), support_fp16);
  }
  if (restored_origin_tensors->find(tensor) != restored_origin_tensors->end()) {
    MS_LOG(ERROR) << "Tensor " << tensor->tensor_name() << " is already be stored";
    delete restore_tensor;
    return RET_ERROR;
  }
  (*restored_origin_tensors)[tensor] = restore_tensor;
  return RET_OK;
}

// support_fp16: current device and package support float16
int CastConstTensorsData(const std::vector<Tensor *> &tensors, std::map<Tensor *, Tensor *> *restored_origin_tensors,
                         TypeId dst_data_type, bool support_fp16) {
  MS_ASSERT(restored_origin_tensors != nullptr);
  if (dst_data_type != kNumberTypeFloat32 && dst_data_type != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Only support fp32 or fp16 as dst_data_type.";
    return RET_PARAM_INVALID;
  }
  for (auto *tensor : tensors) {
    MS_ASSERT(tensor != nullptr);
    // only cast const tensor
    // tensorlist not support fp16 now
    if (!tensor->IsConst() || tensor->data_type() == kObjectTypeTensorType) {
      continue;
    }
    // only support fp32->fp16 or fp16->fp32
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      continue;
    }
    if (tensor->data_type() == kNumberTypeFloat32 && dst_data_type == kNumberTypeFloat16) {
      auto ret = CastAndRestoreConstTensorData(tensor, restored_origin_tensors, kNumberTypeFloat16, support_fp16);
      if (ret != RET_OK) {
        MS_LOG(DEBUG) << "Cast const tensor from fp32 to fp16 failed, tensor name : " << tensor->tensor_name();
        return ret;
      }
    } else if (tensor->data_type() == kNumberTypeFloat16 && dst_data_type == kNumberTypeFloat32) {
      auto ret = CastAndRestoreConstTensorData(tensor, restored_origin_tensors, kNumberTypeFloat32, support_fp16);
      if (ret != RET_OK) {
        MS_LOG(DEBUG) << "Cast const tensor from fp16 to fp32 failed, tensor name : " << tensor->tensor_name();
        return ret;
      }
    } else {
      MS_LOG(DEBUG) << "No need to cast from " << tensor->data_type() << " to " << dst_data_type;
    }
  }
  return RET_OK;
}

inline void FreeRestoreTensors(std::map<Tensor *, Tensor *> *restored_origin_tensors) {
  MS_ASSERT(restored_origin_tensors != nullptr);
  for (auto &restored_origin_tensor : *restored_origin_tensors) {
    restored_origin_tensor.second->set_data(nullptr);
    delete (restored_origin_tensor.second);
    restored_origin_tensor.second = nullptr;
  }
  restored_origin_tensors->clear();
}

inline void RestoreTensorData(std::map<Tensor *, Tensor *> *restored_origin_tensors) {
  MS_ASSERT(restored_origin_tensors != nullptr);
  for (auto &restored_origin_tensor : *restored_origin_tensors) {
    auto *origin_tensor = restored_origin_tensor.first;
    auto *restored_tensor = restored_origin_tensor.second;
    MS_ASSERT(origin_tensor != nullptr);
    MS_ASSERT(restored_tensor != nullptr);
    origin_tensor->FreeData();
    origin_tensor->set_data_type(restored_tensor->data_type());
    origin_tensor->set_data(restored_tensor->data());
    origin_tensor->set_own_data(restored_tensor->own_data());
  }
  FreeRestoreTensors(restored_origin_tensors);
}
}  // namespace

void Scheduler::ResetByExecutionPlan(std::string node_name, TypeId *data_type) {
  if (execution_plan_ == nullptr) {
    return;
  }
  auto iter = execution_plan_->find(node_name);
  if (iter != execution_plan_->end()) {
    *data_type = iter->second;
  }
  return;
}

int Scheduler::FindCpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                             OpParameter *op_parameter, const kernel::KernelKey &desc, TypeId kernel_data_type,
                             kernel::KernelExec **kernel) {
  MS_CHECK_TRUE_MSG(op_parameter != nullptr, RET_ERROR, "op parameter is nullptr.");
  auto op_type = op_parameter->type_;
  if (!KernelRegistry::GetInstance()->SupportKernel(desc)) {
    return RET_NOT_SUPPORT;
  }
  kernel::KernelKey cpu_desc = desc;
  if (kernel_data_type == kNumberTypeFloat16) {
    if (!context_->IsCpuFloat16Enabled() ||
        (cpu_desc.data_type != kNumberTypeFloat32 && cpu_desc.data_type != kNumberTypeFloat16)) {
      return RET_NOT_SUPPORT;
    }
    cpu_desc.data_type = kNumberTypeFloat16;
  }
  auto ret = WeightDecoder::DequantNode(op_parameter, in_tensors, kernel_data_type, src_model_->graph_.version_,
                                        context_->float_mode);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "Dequant input tensors failed: " << ret;
    return RET_NOT_SUPPORT;
  }
  std::map<Tensor *, Tensor *> restored_origin_tensors;

  if (is_train_session_) {
    ret = CastConstTensorsData(in_tensors, &restored_origin_tensors, kernel_data_type,
                               context_->device_and_pkg_support_fp16_);
    if (ret != RET_OK) {
      MS_LOG(DEBUG) << "CastConstTensorsData failed: " << ret;
      return RET_NOT_SUPPORT;
    }
  }

#if defined(PARALLEL_INFERENCE) && defined(ENABLE_MINDRT)
  // reset op task num, The number of operator segmentation tasks is not necessarily equal to the number of threads
  int thread_num_limit = ParallelThreadPoolManager::GetInstance()->GetTaskNum(config_info_);
  if (thread_num_limit != -1 && IsSharedThreadPoolOp(op_type)) {
    op_parameter->thread_num_ = thread_num_limit;
  }
#endif

  ret = KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context_, ms_context_, cpu_desc,
                                                     op_parameter, kernel);
  if (ret == RET_OK) {
    MS_LOG(DEBUG) << "Get TypeId(expect = " << kernel_data_type << ", real = " << cpu_desc.data_type
                  << ") op success: " << PrimitiveCurVersionTypeName(op_type);
    if (is_train_session_) {
      (*kernel)->Prepare();
      RestoreTensorData(&restored_origin_tensors);
    }
  }
  return ret;
}

#ifdef GPU_OPENCL
int Scheduler::FindGpuKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                             OpParameter *op_parameter, const kernel::KernelKey &desc, kernel::KernelExec **kernel,
                             TypeId prefer_data_type) {
  MS_ASSERT(op_parameter != nullptr);
  MS_ASSERT(kernel != nullptr);
  if (!context_->IsDeviceTypeEnabled(DT_GPU)) {
    return RET_NOT_SUPPORT;
  }

  // support more data type like int32
  kernel::KernelKey gpu_desc{kernel::KERNEL_ARCH::kGPU, desc.data_type, NHWC, desc.type};
  if (desc.data_type == kNumberTypeFloat32 && context_->IsGpuFloat16Enabled()) {
    gpu_desc.data_type = kNumberTypeFloat16;
  }
  if (prefer_data_type == kNumberTypeFloat16 || prefer_data_type == kNumberTypeFloat32) {
    gpu_desc.data_type = prefer_data_type;
  }
  // weight dequant
  auto ret = WeightDecoder::DequantNode(op_parameter, in_tensors, kNumberTypeFloat32, src_model_->graph_.version_,
                                        context_->float_mode);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "Dequant input tensors failed: " << ret;
    return RET_NOT_SUPPORT;
  }
  // we don't need to restore tensor for copy data
  ret = CopyConstTensorData(in_tensors, op_parameter->type_);
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "CopyConstTensorsData failed: " << ret;
    return RET_NOT_SUPPORT;
  }
  ret = KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context_, ms_context_, gpu_desc,
                                                     op_parameter, kernel);
  if (ret == RET_OK) {
    MS_LOG(DEBUG) << "Get gpu op success: " << PrimitiveCurVersionTypeName(gpu_desc.type);
  } else {
    MS_LOG(DEBUG) << "Get gpu op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(gpu_desc.type);
  }
  return ret;
}
#endif

int Scheduler::FindProviderKernel(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                  const LiteGraph::Node *node, TypeId data_type, kernel::KernelExec **kernel) {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  MS_ASSERT(kernel != nullptr);
  int ret = RET_NOT_SUPPORT;
  auto prim_type = GetPrimitiveType(node->primitive_, schema_version_);
  if (prim_type == schema::PrimitiveType_Custom) {
    for (auto &&device : context_->device_list_) {
      if (!device.provider_.empty() && !device.provider_device_.empty()) {
        kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type,       NHWC, prim_type,
                               device.provider_device_,   device.provider_};
        ret = KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context_, ms_context_, desc,
                                                           nullptr, kernel, node->primitive_);
        if (ret == RET_OK && *kernel != nullptr) {
          return ret;
        }
      }
    }

    kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, NHWC, prim_type, "", ""};
    ret = KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context_, ms_context_, desc, nullptr,
                                                       kernel, node->primitive_);
    if (ret == RET_OK && *kernel != nullptr) {
      return ret;
    }
    return RET_NOT_SUPPORT;
  }
  if (!context_->IsProviderEnabled()) {
    return ret;
  }
  if (schema_version_ == SCHEMA_V0) {
    return ret;
  }
  for (auto &&device : context_->device_list_) {
    if (!device.provider_.empty()) {
      kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type,       NHWC, prim_type,
                             device.provider_device_,   device.provider_};
      ret = KernelRegistry::GetInstance()->GetKernelExec(in_tensors, out_tensors, context_, ms_context_, desc, nullptr,
                                                         kernel, node->primitive_);
      if (ret == RET_OK && *kernel != nullptr) {
        return ret;
      }
    }
  }
#endif
  return RET_NOT_SUPPORT;
}

kernel::KernelExec *Scheduler::FindBackendKernel(const std::vector<Tensor *> &in_tensors,
                                                 const std::vector<Tensor *> &out_tensors, const LiteGraph::Node *node,
                                                 TypeId prefer_data_type) {
  MS_ASSERT(node != nullptr);
  // why we need this
  TypeId data_type;
  if (node->quant_type_ == schema::QuantType_QUANT_WEIGHT) {
    if (in_tensors.front()->data_type() == kNumberTypeBool) {
      data_type = kNumberTypeBool;
    } else {
      data_type = kNumberTypeFloat32;
    }
  } else {
    data_type = GetFirstFp32Fp16OrInt8Type(in_tensors);
    if (data_type == kTypeUnknown) {
      MS_LOG(ERROR) << "GetFirstFp32Fp16OrInt8Type is unknown.";
      return nullptr;
    }
  }
  if (context_->float_mode) {
    for (auto tensor : out_tensors) {
      if (!tensor->quant_params().empty() &&
          (tensor->data_type() == kNumberTypeInt8 || tensor->data_type() == kNumberTypeUInt8)) {
        data_type = kNumberTypeFloat32;
        tensor->set_data_type(kNumberTypeFloat32);
      }
    }
  }
  kernel::KernelExec *kernel = nullptr;
  auto status = FindProviderKernel(in_tensors, out_tensors, node, data_type, &kernel);
  if (status == RET_OK && kernel != nullptr) {
    return kernel;
  }
  MS_ASSERT(!node->output_indices_.empty());
  OpParameter *op_parameter = op_parameters_[node->output_indices_.at(0)];
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Can not find OpParameter!type: " << GetPrimitiveTypeName(node->primitive_, schema_version_);
    return nullptr;
  }

#if (defined GPU_OPENCL) || (defined ENABLE_FP16)
  int kernel_thread_count = op_parameter->thread_num_;
#endif
  op_parameter->is_train_session_ = is_train_session_;
  kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, data_type, NHWC, op_parameter->type_};

#ifdef GPU_OPENCL
  bool gpu_priority = DeviceTypePriority(context_, DT_GPU, DT_CPU);
  bool use_gpu_kernel = node->device_type_ == DT_GPU || node->device_type_ == kDefaultDeviceType;
  if (gpu_priority && use_gpu_kernel) {
    status = FindGpuKernel(in_tensors, out_tensors, op_parameter, desc, &kernel, prefer_data_type);
    if (status == RET_OK) {
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get gpu op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(desc.type) << " "
                    << node->name_;
      if (status == RET_ERROR) {
        op_parameters_.erase(node->output_indices_.at(0));
        auto ret = InferNodeShape(node);
        if (ret == RET_INFER_INVALID || ret == RET_OK) {
          op_parameter = op_parameters_[node->output_indices_.at(0)];
          op_parameter->thread_num_ = kernel_thread_count;
        } else {
          MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
          return nullptr;
        }
      }
    }
  }
#endif
#ifdef ENABLE_FP16
  if ((prefer_data_type == kNumberTypeFloat16 || prefer_data_type == kTypeUnknown) &&
      ((is_train_session_ == false) || (sched_cb_ && sched_cb_->SchedFp16Kernel(node)))) {
    status = FindCpuKernel(in_tensors, out_tensors, op_parameter, desc, kNumberTypeFloat16, &kernel);
    if (status == RET_OK) {
      return kernel;
    } else {
      MS_LOG(DEBUG) << "Get fp16 op failed, scheduler to cpu: " << PrimitiveCurVersionTypeName(desc.type) << " "
                    << node->name_;
      if (status == RET_ERROR) {
        op_parameters_.erase(node->output_indices_.at(0));
        auto ret = InferNodeShape(node);
        if (ret == RET_INFER_INVALID || ret == RET_OK) {
          op_parameter = op_parameters_[node->output_indices_.at(0)];
          op_parameter->thread_num_ = kernel_thread_count;
        } else {
          MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
          return nullptr;
        }
      }
    }
  }
#endif
  if (data_type == kNumberTypeFloat16) {
    MS_LOG(DEBUG) << "Get fp16 op failed, back to fp32 op.";
    desc.data_type = kNumberTypeFloat32;
  }
  status = FindCpuKernel(in_tensors, out_tensors, op_parameter, desc, kNumberTypeFloat32, &kernel);
  if (status == RET_OK) {
    return kernel;
  } else if (status == RET_ERROR) {
    op_parameters_.erase(node->output_indices_.at(0));
    auto ret = InferNodeShape(node);
    if (!(ret == RET_INFER_INVALID || ret == RET_OK)) {
      MS_LOG(ERROR) << "Try repeat infer fail: " << node->name_;
    }
  }
#ifdef OP_INT8_CLIP
  if (desc.data_type == kNumberTypeInt8) {
    MS_LOG(ERROR) << unsupport_int8_log;
  }
#endif
  return nullptr;
}

namespace {
kernel::SubGraphType GetKernelSubGraphType(const kernel::KernelExec *kernel, const InnerContext &context,
                                           bool is_controlflow = false) {
  if (kernel == nullptr) {
    return kernel::kNotSubGraph;
  }

  auto desc = kernel->desc();
  if (desc.arch == kernel::KERNEL_ARCH::kGPU) {
    if (desc.data_type == kNumberTypeFloat16) {
      return kernel::kGpuFp16SubGraph;
    } else {
      return kernel::kGpuFp32SubGraph;
    }
  } else if (desc.arch == kernel::KERNEL_ARCH::kNPU) {
    return kernel::kNpuSubGraph;
  } else if (desc.arch == kernel::KERNEL_ARCH::kAPU) {
    return kernel::kApuSubGraph;
  } else if (desc.arch == kernel::KERNEL_ARCH::kCPU) {
    if (desc.data_type == kNumberTypeFloat16) {
      return kernel::kCpuFP16SubGraph;
    } else {
      return kernel::kCpuFP32SubGraph;
    }
  } else if (desc.arch == kernel::KERNEL_ARCH::kCustom) {
    return kernel::kCustomSubGraph;
  }
  return kernel::kNotSubGraph;
}
}  // namespace

kernel::KernelExec *Scheduler::SchedulePartialToKernel(const lite::LiteGraph::Node *src_node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(src_node != nullptr);
  auto *primitive = src_node->primitive_;
  MS_ASSERT(primitive != nullptr);
  if (!IsPartialNode(primitive, schema_version_)) {
    return nullptr;
  }
  auto subgraph_index = GetPartialGraphIndex(src_node->primitive_, schema_version_);
  if (SubGraphHasScheduled(subgraph_index)) {
    MS_LOG(INFO) << "Subgraph has been scheduled.";
    return {};
  } else {
    SubGraphMarkScheduled(subgraph_index);
  }
  auto subgraph_kernel = SchedulePartialToSubGraphKernel(subgraph_index);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "SchedulePartialToSubGraphKernel failed, subgraph_index: " << subgraph_index;
    return {};
  }
  subgraph_kernel->set_name("subgraph_" + std::to_string(subgraph_index));
  return subgraph_kernel;
}

#ifdef ENABLE_FP16
int Scheduler::SubGraphPreferDataType(const int &subgraph_index, TypeId *prefer_data_type) {
  if (!context_->IsCpuFloat16Enabled() || context_->GetDelegateMode() == kNNAPI) {
    *prefer_data_type = kNumberTypeFloat32;
    return RET_OK;
  }

  auto subgraph = src_model_->graph_.sub_graphs_.at(subgraph_index);
  for (auto node_index : subgraph->node_indices_) {
    auto node = src_model_->graph_.all_nodes_[node_index];
    MS_ASSERT(node != nullptr);
    MS_ASSERT(!node->output_indices_.empty());
    OpParameter *op_parameter = op_parameters_[node->output_indices_.at(0)];
    if (op_parameter == nullptr) {
      MS_LOG(ERROR) << "Can not find OpParameter!type: " << GetPrimitiveTypeName(node->primitive_, schema_version_);
      return RET_ERROR;
    }
    kernel::KernelKey desc{kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat16, NHWC, op_parameter->type_};
    if (!KernelRegistry::GetInstance()->SupportKernel(desc)) {
      *prefer_data_type = kNumberTypeFloat32;
      return RET_OK;
    }

    std::vector<Tensor *> inputs;
    std::vector<Tensor *> outputs;
    FindNodeInoutTensors(*node, &inputs, &outputs);
    if (node->quant_type_ == schema::QuantType_QUANT_WEIGHT) {
      *prefer_data_type = kNumberTypeFloat32;
      return RET_OK;
    }
    TypeId data_type = GetFirstFp32Fp16OrInt8Type(inputs);
    if (data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16) {
      *prefer_data_type = kNumberTypeFloat32;
      return RET_OK;
    }
  }
  *prefer_data_type = kNumberTypeFloat16;
  return RET_OK;
}
#endif

std::vector<kernel::KernelExec *> Scheduler::ScheduleMainSubGraphToKernels() {
  std::vector<kernel::KernelExec *> kernels;
  std::vector<lite::Tensor *> in_tensors;
  std::vector<lite::Tensor *> out_tensors;
  TypeId prefer_data_type = context_->GetDelegateMode() == kNNAPI ? kNumberTypeFloat32 : kTypeUnknown;
  auto ret = ScheduleSubGraphToKernels(kMainSubGraphIndex, &kernels, &in_tensors, &out_tensors, prefer_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule subgraph failed, index: " << kMainSubGraphIndex;
    for (auto *kernel : kernels) {
      delete kernel;
      kernel = nullptr;
    }
    return {};
  }
  return kernels;
}

kernel::KernelExec *Scheduler::SchedulePartialToSubGraphKernel(const int &subgraph_index) {
  TypeId prefer_data_type = kTypeUnknown;
#ifdef ENABLE_FP16
  if (SubGraphPreferDataType(subgraph_index, &prefer_data_type) != RET_OK) {
    MS_LOG(ERROR) << "SubGraphPreferDataType failed, subgraph index: " << subgraph_index;
    return nullptr;
  }
#endif
  std::vector<kernel::KernelExec *> kernels;
  std::vector<lite::Tensor *> in_tensors;
  std::vector<lite::Tensor *> out_tensors;
  auto ret = ScheduleSubGraphToKernels(subgraph_index, &kernels, &in_tensors, &out_tensors, prefer_data_type);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule subgraph failed, index: " << subgraph_index;
    return nullptr;
  }
  kernel::KernelExecUtil::FindAllInoutKernels(kernels);
  kernel::SubGraphType cur_sub_graph_type = kernel::kCpuFP32SubGraph;
  if (!kernels.empty()) {
    cur_sub_graph_type = GetKernelSubGraphType(kernels.front(), *context_, true);
  }
  MS_LOG(INFO) << "cur_sub_graph_type: " << cur_sub_graph_type;
  auto subgraph_kernel = kernel::KernelExecUtil::CreateSubGraphKernel(kernels, &in_tensors, &out_tensors,
                                                                      cur_sub_graph_type, *context_, schema_version_);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "CreateSubGraphKernel failed, cur_sub_graph_type: " << cur_sub_graph_type;
    return nullptr;
  }
  return subgraph_kernel;
}

std::vector<kernel::KernelExec *> Scheduler::ScheduleSubGraphToSubGraphKernels(const int &subgraph_index) {
  if (subgraph_index == kMainSubGraphIndex) {
    return ScheduleMainSubGraphToKernels();
  }
  auto subgraph_kernel = SchedulePartialToSubGraphKernel(subgraph_index);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "SchedulePartialToSubGraphKernel failed, subgraph_index: " << subgraph_index;
    return {};
  }
  subgraph_kernel->set_name("subgraph_" + std::to_string(subgraph_index));
  subgraph_index_subgraph_kernel_map_[subgraph_index] = subgraph_kernel;
  return {subgraph_kernel};
}

kernel::KernelExec *Scheduler::ScheduleNodeToKernel(const lite::LiteGraph::Node *src_node, TypeId prefer_data_type) {
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  MS_ASSERT(src_node != nullptr);
  FindNodeInoutTensors(*src_node, &inputs, &outputs);

  ResetByExecutionPlan(src_node->name_, &prefer_data_type);

  mindspore::kernel::KernelExec *kernel = nullptr;
  if (src_model_->model_type_ != mindspore::lite::ModelType_MSLite) {
    auto abstract_model_ptr = reinterpret_cast<AbstractBaseModel *>(src_model_);
    if (abstract_model_ptr == nullptr) {
      MS_LOG(ERROR) << "src model is not abstract base model return nullptr.";
      return nullptr;
    }
    kernel = abstract_model_ptr->FindBackendKernel(inputs, outputs, src_node, context_, prefer_data_type);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "FindBackendKernel return nullptr, name: " << src_node->name_;
      return nullptr;
    }
  } else {
    kernel = this->FindBackendKernel(inputs, outputs, src_node, prefer_data_type);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "FindBackendKernel return nullptr, name: " << src_node->name_
                    << ", type: " << GetPrimitiveTypeName(src_node->primitive_, schema_version_);
      return nullptr;
    }
  }
  op_parameters_[src_node->output_indices_.at(0)] = nullptr;
  auto ret = kernel::KernelExecUtil::SetKernelTensorDataType(kernel);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set tensor data type for kernel " << kernel->name() << std::endl;
    delete kernel;
    return nullptr;
  }
  kernel->set_name(src_node->name_);
  if (kernel->kernel() != nullptr) {
    kernel->kernel()->SetConfig(config_info_);
  }
  return kernel;
}

bool Scheduler::IsControlFlowPattern(const lite::LiteGraph::Node &partial_node) {
  lite::LiteGraph::Node *partial_node_output = nullptr;
  for (auto output_index : partial_node.output_indices_) {
    for (auto &node : src_model_->graph_.all_nodes_) {
      if (IsContain(node->input_indices_, output_index)) {
        partial_node_output = node;
        break;
      }
    }
  }

  return partial_node_output != nullptr && (IsCallNode(partial_node_output->primitive_, schema_version_) ||
                                            IsSwitchNode(partial_node_output->primitive_, schema_version_) ||
                                            IsSwitchLayerNode(partial_node_output->primitive_, schema_version_));
}

int Scheduler::ScheduleGraphToKernels(std::vector<kernel::KernelExec *> *dst_kernels, TypeId prefer_data_type) {
  subgraphs_to_schedule_.push_back(kMainSubGraphIndex);
  while (!subgraphs_to_schedule_.empty()) {
    auto cur_subgraph_index = subgraphs_to_schedule_.front();
    subgraphs_to_schedule_.pop_front();
    auto kernels = ScheduleSubGraphToSubGraphKernels(cur_subgraph_index);
    if (kernels.empty()) {
      MS_LOG(ERROR) << "ScheduleSubGraphToSubGraphKernel failed";
      return RET_ERROR;
    }
    std::copy(kernels.begin(), kernels.end(), std::back_inserter(*dst_kernels));
  }
  return RET_OK;
}

int Scheduler::ScheduleSubGraphToKernels(size_t subgraph_index, std::vector<kernel::KernelExec *> *dst_kernels,
                                         std::vector<lite::Tensor *> *in_tensors,
                                         std::vector<lite::Tensor *> *out_tensors, TypeId prefer_data_type) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(!src_model_->graph_.sub_graphs_.empty());
  MS_ASSERT(src_model_->graph_.sub_graphs_.size() > subgraph_index);
  MS_ASSERT(dst_kernels != nullptr);
  MS_ASSERT(dst_kernels->empty());
  auto subgraph = src_model_->graph_.sub_graphs_.at(subgraph_index);
  auto ret = RET_OK;
  for (auto node_index : subgraph->node_indices_) {
    auto node = src_model_->graph_.all_nodes_[node_index];
    MS_ASSERT(node != nullptr);
    auto *primitive = node->primitive_;
    MS_ASSERT(primitive != nullptr);
    kernel::KernelExec *kernel = nullptr;

    if (src_model_->model_type_ == ModelType_MSLite && IsPartialNode(primitive, schema_version_)) {
      if (IsControlFlowPattern(*node)) {
        kernel = ScheduleNodeToKernel(node, prefer_data_type);
        auto partial_subgraph_index = GetPartialGraphIndex(primitive, schema_version_);
        MS_CHECK_TRUE_MSG(control_flow_scheduler_ != nullptr, RET_ERROR, "control flow scheduler is nullptr.");
        control_flow_scheduler_->RecordSubgraphCaller(partial_subgraph_index, kernel);
        if (SubGraphHasScheduled(partial_subgraph_index)) {
          partial_kernel_subgraph_index_map_[kernel] = static_cast<size_t>(partial_subgraph_index);
          MS_LOG(INFO) << "subgraph has scheduled. ";
        } else {
          SubGraphMarkScheduled(partial_subgraph_index);
          partial_kernel_subgraph_index_map_[kernel] = static_cast<size_t>(partial_subgraph_index);
          subgraphs_to_schedule_.push_back(partial_subgraph_index);
        }
      } else {
        MS_CHECK_TRUE_MSG(
          subgraph_index != static_cast<size_t>(GetPartialGraphIndex(node->primitive_, schema_version_)), RET_ERROR,
          "Unreasonable cycles exist in subgraph.");
        kernel = SchedulePartialToKernel(node);
      }
    } else {
      kernel = ScheduleNodeToKernel(node, prefer_data_type);
    }
    if (kernel == nullptr || ret != RET_OK) {
      MS_LOG(ERROR) << "schedule node return nullptr, name: " << node->name_
                    << ", type: " << GetPrimitiveTypeName(primitive, schema_version_);
      return RET_ERROR;
    }
    kernel->set_is_model_output(IsContain(graph_output_node_indexes_, size_t(node_index)));
    dst_kernels->emplace_back(kernel);
    primitives_.emplace(kernel->kernel(), static_cast<const schema::Primitive *>(primitive));
  }
  if (in_tensors != nullptr) {
    std::transform(subgraph->input_indices_.begin(), subgraph->input_indices_.end(), std::back_inserter(*in_tensors),
                   [&](const uint32_t index) { return this->src_tensors_->at(index); });
  }
  if (out_tensors != nullptr) {
    std::transform(subgraph->output_indices_.begin(), subgraph->output_indices_.end(), std::back_inserter(*out_tensors),
                   [&](const uint32_t index) { return this->src_tensors_->at(index); });
  }
  return RET_OK;
}

namespace {
bool KernelFitCurrentSubGraphCPUFp32(TypeId data_type) {
  return (data_type == kNumberTypeFloat32 || data_type == kNumberTypeFloat || data_type == kNumberTypeInt8 ||
          data_type == kNumberTypeInt || data_type == kNumberTypeInt32 || data_type == kNumberTypeInt64 ||
          data_type == kNumberTypeUInt8 || data_type == kNumberTypeBool);
}

bool KernelFitCurrentSubGraph(const kernel::SubGraphType subgraph_type, const kernel::KernelExec &kernel) {
  switch (subgraph_type) {
    case kernel::SubGraphType::kNotSubGraph:
    case kernel::SubGraphType::kApuSubGraph:
      return false;
    case kernel::SubGraphType::kGpuFp16SubGraph:
      if (kernel.desc().arch != kernel::KERNEL_ARCH::kGPU) {
        return false;
      }
      return (kernel.desc().data_type != kNumberTypeFloat32);
    case kernel::SubGraphType::kGpuFp32SubGraph:
      if (kernel.desc().arch != kernel::KERNEL_ARCH::kGPU) {
        return false;
      }
      return (kernel.desc().data_type != kNumberTypeFloat16);
    case kernel::SubGraphType::kNpuSubGraph:
      return kernel.desc().arch == kernel::KERNEL_ARCH::kNPU;
    case kernel::SubGraphType::kCpuFP16SubGraph: {
      auto desc = kernel.desc();
      if (desc.arch != kernel::KERNEL_ARCH::kCPU) {
        return false;
      }
#ifdef ENABLE_FP16
      if (desc.data_type == kNumberTypeInt8 || desc.data_type == kNumberTypeUInt8) {
        return true;
      }
#endif
      return (desc.data_type == kNumberTypeFloat16);
    }
    case kernel::SubGraphType::kCpuFP32SubGraph: {
      auto desc = kernel.desc();
      if (desc.arch != kernel::KERNEL_ARCH::kCPU) {
        return false;
      }
      return KernelFitCurrentSubGraphCPUFp32(desc.data_type);
    }
    default:
      return false;
  }
}

kernel::KernelExec *FindAllSubGraphKernels(const std::vector<kernel::KernelExec *> &sorted_kernels,
                                           const InnerContext &context, size_t *cur_index, int schema_version) {
  std::vector<kernel::KernelExec *> sub_kernels;
  sub_kernels.emplace_back(sorted_kernels[*cur_index]);
  auto cur_sub_graph_type = GetKernelSubGraphType(sorted_kernels[*cur_index], context);
  for (*cur_index = *cur_index + 1; *cur_index < sorted_kernels.size(); ++(*cur_index)) {
    auto cur_kernel = sorted_kernels[*cur_index];
    MS_ASSERT(GetKernelSubGraphType(cur_kernel, context) != kernel::kApuSubGraph);
    // already a subgraph or a delegate
    if (cur_kernel->desc().arch == kernel::kDelegate) {
      --(*cur_index);
      break;
    }
    if (cur_kernel->subgraph_type() != kernel::kNotSubGraph ||
        !KernelFitCurrentSubGraph(cur_sub_graph_type, *cur_kernel)) {
      --(*cur_index);
      break;
    }
    sub_kernels.emplace_back(cur_kernel);
  }
  return kernel::KernelExecUtil::CreateSubGraphKernel(sub_kernels, nullptr, nullptr, cur_sub_graph_type, context,
                                                      schema_version);
}
}  // namespace

int Scheduler::ConstructNormalSubGraphs(const std::vector<kernel::KernelExec *> &src_kernel,
                                        std::vector<kernel::KernelExec *> *dst_kernel,
                                        std::map<const kernel::KernelExec *, bool> *is_kernel_finish) {
  if (src_kernel.empty()) {
    return RET_OK;
  }

  // construct subgraph
  for (size_t index = 0; index < src_kernel.size(); index++) {
    auto cur_kernel = src_kernel[index];
    MS_ASSERT(cur_kernel != nullptr);
    // Not support APU now
    MS_ASSERT(GetKernelSubGraphType(cur_kernel, *context_) != kernel::kApuSubGraph);
    if (cur_kernel->desc().arch == kernel::kDelegate) {
      dst_kernel->emplace_back(cur_kernel);
      continue;
    }
    // already a subgraph or a delegate
    if (cur_kernel->subgraph_type() != kernel::kNotSubGraph) {
      dst_kernel->emplace_back(cur_kernel);
      continue;
    }
    auto subgraph = FindAllSubGraphKernels(src_kernel, *context_, &index, schema_version_);
    if (subgraph == nullptr) {
      MS_LOG(ERROR) << "Create SubGraphKernel failed";
      return RET_ERROR;
    }
    dst_kernel->emplace_back(subgraph);
  }
  for (auto *subgraph : *dst_kernel) {
    if (subgraph->desc().arch == kernel::kDelegate) {
      *infer_along_running_ = false;
      continue;
    }
    if (subgraph->subgraph_type() != kernel::kCpuFP32SubGraph &&
        subgraph->subgraph_type() != kernel::kCpuFP16SubGraph) {
      *infer_along_running_ = false;
    }
    auto subgraph_kernel = static_cast<kernel::SubGraphKernel *>(subgraph);
    if (subgraph_kernel == nullptr) {
      MS_LOG(ERROR) << "kernel: " << subgraph->name() << " not is subgraph kernel.";
      return RET_ERROR;
    }
    // this is for train session cpu fp16, should be removed in the future.
    auto ret = subgraph_kernel->SetFp16Attr();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Init SubGraph failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

TypeId Scheduler::GetFirstFp32Fp16OrInt8Type(const std::vector<Tensor *> &in_tensors) {
  for (const auto &tensor : in_tensors) {
    auto dtype = tensor->data_type();
    if (dtype == kObjectTypeTensorType) {
      return TensorListDataType(tensor);
    }
    std::unordered_set<TypeId> type_set = {kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeInt8,  kNumberTypeInt32,
                                           kNumberTypeBool,    kNumberTypeUInt8,   kObjectTypeString};
    if (type_set.find(dtype) != type_set.end()) {
      return dtype;
    }
  }
  if (in_tensors.empty()) {
    MS_LOG(ERROR) << "in tensor is empty.";
    return kTypeUnknown;
  }
  MS_ASSERT(!in_tensors.empty());
  return in_tensors[0]->data_type() == kObjectTypeTensorType ? kNumberTypeFloat32 : in_tensors[0]->data_type();
}

kernel::SubGraphType Scheduler::PartialSubGraphType(const std::vector<kernel::KernelExec *> &kernels) {
  if (std::any_of(kernels.begin(), kernels.end(),
                  [](const kernel::KernelExec *item) { return item->desc().data_type == kNumberTypeFloat16; })) {
    return kernel::kCpuFP16SubGraph;
  }
  return kernel::kCpuFP32SubGraph;
}

int Scheduler::InferSwitchShape(const lite::LiteGraph::Node *switch_node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(switch_node != nullptr);
  std::deque<lite::LiteGraph::Node *> partial_cnode_to_infer{};
  for (size_t i = 1; i < switch_node->input_indices_.size(); ++i) {
    auto branch_output_index = switch_node->input_indices_.at(i);
    for (auto &node : src_model_->graph_.all_nodes_) {
      if (IsContain(node->output_indices_, branch_output_index) && IsPartialNode(node->primitive_, schema_version_) &&
          partial_cnode_inferred_.find(node) == partial_cnode_inferred_.end()) {
        partial_cnode_inferred_.insert(node);
        partial_cnode_to_infer.push_back(node);
        break;
      }
    }
  }

  while (!partial_cnode_to_infer.empty()) {
    auto &node = partial_cnode_to_infer.front();
    partial_cnode_to_infer.pop_front();
    int ret = InferPartialShape(node);
    if (ret != RET_OK) {
      MS_LOG(WARNING) << "partial infer not ok, ret: " << ret;
    }
  }
  return RET_OK;
}

LiteGraph::Node *Scheduler::NodeInputIsSwitchType(const lite::LiteGraph::Node *node) {
  MS_ASSERT(src_model_ != nullptr);
  MS_ASSERT(node != nullptr);
  for (auto &iter : src_model_->graph_.all_nodes_) {
    if (iter->output_indices_ == node->input_indices_) {
      if (IsSwitchNode(iter->primitive_, schema_version_) || IsSwitchLayerNode(iter->primitive_, schema_version_)) {
        return iter;
      } else {
        return nullptr;
      }
    }
  }
  return nullptr;
}

bool Scheduler::SubGraphHasScheduled(const int &index) {
  return scheduled_subgraph_index_.find(index) != scheduled_subgraph_index_.end();
}

void Scheduler::SubGraphMarkScheduled(const int &index) { scheduled_subgraph_index_.insert(index); }

int Scheduler::ConstructControlFlowMainGraph(std::vector<kernel::KernelExec *> *kernels) {
  auto back_kernels = *kernels;
  kernels->clear();
  std::vector<kernel::KernelExec *> main_graph_kernels{};
  for (auto &kernel : back_kernels) {
    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      kernels->push_back(kernel);
    } else {
      main_graph_kernels.push_back(kernel);
    }
  }
  auto cur_subgraph_type = PartialSubGraphType(main_graph_kernels);
  auto subgraph_kernel = kernel::KernelExecUtil::CreateSubGraphKernel(main_graph_kernels, nullptr, nullptr,
                                                                      cur_subgraph_type, *context_, schema_version_);
  if (subgraph_kernel == nullptr) {
    MS_LOG(ERROR) << "create main graph for control flow model failed.";
    return RET_ERROR;
  }
  kernels->insert(kernels->begin(), subgraph_kernel);
  return RET_OK;
}

std::vector<kernel::KernelExec *> Scheduler::NonTailCallNodes() {
  std::vector<kernel::KernelExec *> ret{};
  if (*is_control_flow_) {
    ret = control_flow_scheduler_->GetNonTailCalls();
  }
  return ret;
}
}  // namespace mindspore::lite

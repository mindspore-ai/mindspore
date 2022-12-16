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

#include "src/litert/lite_session.h"
#include <set>
#include <vector>
#include <utility>
#include "src/litert/pack_weight_manager.h"
#include "src/litert/runtime_pass.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/scheduler.h"
#include "src/litert/inner_allocator.h"
#include "src/litert/executor.h"
#include "src/common/context_util.h"
#include "src/common/utils.h"
#include "src/common/graph_util.h"
#include "src/common/tensor_util.h"
#include "src/common/file_utils.h"
#include "src/litert/lite_model.h"
#include "src/litert/weight_decoder.h"
#include "src/litert/runtime_allocator.h"
#include "src/litert/kernel_exec_util.h"
#include "src/litert/cpu_info.h"
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
#include "src/registry/register_kernel_impl.h"
#endif
#ifdef ENABLE_MINDRT
#include "src/litert/mindrt_executor.h"
#endif
#if SUPPORT_NPU
#include "src/litert/delegate/npu/npu_delegate.h"
#endif
#if GPU_OPENCL
#include "src/litert/kernel/opencl/opencl_subgraph.h"
#endif
#if GPU_TENSORRT
#include "src/litert/delegate/tensorrt/tensorrt_delegate.h"
#endif
#if SUPPORT_NNAPI
#include "src/litert/delegate/nnapi/nnapi_delegate.h"
#endif
#ifdef ENABLE_COREML
#include "src/litert/delegate/coreml/coreml_delegate.h"
#endif
#include "src/litert/runtime_convert.h"
#include "extendrt/mindir_loader/model_loader.h"
#ifndef __ANDROID__
#include "kernel/ascend/plugin/ascend_kernel_plugin.h"
#endif
#include "thread/parallel_thread_pool_manager.h"

using AbstractBaseModel = mindspore::infer::AbstractBaseModel;

namespace mindspore {
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif
namespace lite {
namespace {
bool ExistCustomCpuKernel() {
#ifndef CUSTOM_KERNEL_REGISTRY_CLIP
  const std::string kArchCPU = "CPU";
  auto custom_kernel_creators = registry::RegistryKernelImpl::GetInstance()->GetCustomKernelCreators();
  for (const auto &custom_kernel_creator : custom_kernel_creators) {  // <provider, <arch, <type, CreateKernel*>>>
    if (custom_kernel_creator.second.empty()) {
      continue;
    }
    if (std::any_of(
          custom_kernel_creator.second.begin(), custom_kernel_creator.second.end(),
          [kArchCPU](const std::pair<std::string, std::unordered_map<std::string, registry::CreateKernel *>> &pair) {
            return pair.first == kArchCPU && !pair.second.empty();
          })) {
      return true;
    }
  }
#endif
  return false;
}
}  // namespace

LiteSession::LiteSession() {
#ifdef USE_GLOG
  mindspore::mindspore_log_init();
#endif
  this->is_running_.store(false);
}

int LiteSession::CheckTensorValid(lite::Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
  if (dst_tensor->data_type() == kObjectTypeTensorType) {
    return RET_OK;
  }
  if (dst_tensor->IsGraphInput() || dst_tensor->IsGraphOutput()) {
    return RET_OK;
  }
  if (dst_tensor->IsConst() == false && dst_tensor->data() != nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

void LiteSession::ConvertTensorsQuantParam(const schema::Tensor *src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  auto quant_params = src_tensor->quantParams();
  if (quant_params != nullptr) {
    for (size_t j = 0; j < quant_params->size(); j++) {
      auto quant_param = quant_params->Get(j);
      LiteQuantParam quant_arg{};
      if (quant_param == nullptr) {
        quant_arg.inited = false;
      } else {
        quant_arg.inited = true;
        quant_arg.bitNum = quant_param->numBits();
        quant_arg.scale = quant_param->scale();
        quant_arg.zeroPoint = quant_param->zeroPoint();
        quant_arg.var_corr = quant_param->varCorr();
        quant_arg.mean_corr = quant_param->meanCorr();
        quant_arg.roundType = quant_param->roundType();
        quant_arg.multiplier = quant_param->multiplier();
        quant_arg.dstDtype = quant_param->dstDtype();
        quant_arg.min = quant_param->min();
        quant_arg.max = quant_param->max();
      }
      dst_tensor->AddQuantParam(quant_arg);
    }
  }
  auto quant_clusters = src_tensor->quantClusters();
  if (quant_clusters != nullptr) {
    std::vector<float> clusters;
    for (size_t j = 0; j < quant_clusters->size(); j++) {
      clusters.push_back(quant_clusters->Get(j));
    }
    dst_tensor->set_quant_clusters(clusters);
  }
}

int LiteSession::ConvertTensorsData(const lite::LiteModel *model, size_t tensor_index, lite::Tensor *dst_tensor) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  auto src_tensor = model->GetSchemaTensor(tensor_index);
  if (src_tensor == nullptr || src_tensor->handler() == nullptr || src_tensor->data() == nullptr ||
      src_tensor->length() == 0) {
    MS_LOG(DEBUG) << "No valid data converted.";
    return RET_OK;
  }

  /* tensor list convert */
  if (dst_tensor->data_type() == kObjectTypeTensorType) {
    const int *src_data = reinterpret_cast<const int *>(src_tensor->data());
    return DecodeTensorLsit(dst_tensor, src_data, src_tensor->length());
  }

  /* normal tensor check */
  auto shape_info = dst_tensor->shape();
  if (shape_info.end() !=
      std::find_if(shape_info.begin(), shape_info.end(), [](const int shape) { return shape <= 0; })) {
    MS_LOG(ERROR) << "Invalid shape size, tensor name: " << src_tensor->handler()->name();
    return RET_ERROR;
  }

  int compress_type = src_tensor->handler()->weightQuantCompressType();
  int ret = RET_NO_CHANGE;
  if (compress_type != kFSEInfer) {
    ret = WeightDecoder::DecompressTensor(*src_tensor, dst_tensor);
  }
  if (ret == RET_NO_CHANGE) {
    if (dst_tensor->Size() == 0 || src_tensor->length() < dst_tensor->Size()) {
      MS_LOG(ERROR) << "Tensor data shape invalid";
      return RET_ERROR;
    }
    auto data_pair = src_tensor->ReleaseData();
    dst_tensor->set_data(data_pair.second);
    dst_tensor->set_own_data(data_pair.first);
  } else if (ret != RET_OK) {
    MS_LOG(ERROR) << "Decompress tensor data failed: " << ret;
    return ret;
  }
  return RET_OK;
}

lite::Tensor *LiteSession::ConvertTensor(const schema::Tensor &src_tensor) {
  int32_t data_type = src_tensor.dataType();
  if (data_type <= kTypeUnknown || data_type >= kMonadTypeEnd) {
    MS_LOG(ERROR) << "invalid data type. " << data_type;
    return nullptr;
  }
  auto src_category = TensorCategory(src_tensor);
  std::vector<int> shape;
  if (src_tensor.dims() == nullptr) {
    MS_LOG(DEBUG) << "Dims of src_tensor is nullptr";
  }
  if (src_tensor.dims() != nullptr) {
    if (src_tensor.dataType() == kObjectTypeString && src_tensor.data() != nullptr) {
      shape.push_back(src_tensor.data()->size());
    } else {
      for (size_t j = 0; j < src_tensor.dims()->size(); j++) {
        shape.push_back(src_tensor.dims()->data()[j]);
      }
    }
    if (std::any_of(shape.begin(), shape.end(), [](const int &element) { return element < 0 && element != -1; })) {
      MS_LOG(ERROR) << "Dims of src_tensor is unsupported";
      return nullptr;
    }
  }
  lite::Tensor *dst_tensor = nullptr;
  if (TypeId(data_type) == kObjectTypeTensorType) {
    MS_CHECK_TRUE_RET(src_tensor.data() != nullptr, nullptr);
    MS_CHECK_TRUE_RET(src_tensor.data()->size() > 0, nullptr);
    auto src_data = src_tensor.data()->data();
    dst_tensor = CreateTensorList(shape, src_category, src_data);
  } else {
    dst_tensor = new (std::nothrow)
      Tensor(TypeId(data_type), shape, static_cast<mindspore::Format>(src_tensor.format()), src_category);
  }
  if (src_tensor.name() != nullptr) {
    dst_tensor->set_tensor_name(src_tensor.name()->str());
  }
  auto compress_type = static_cast<CompressType>(src_tensor.weightQuantCompressType());
  if (compress_type == kFSEInfer) {
    dst_tensor->set_compress_type(static_cast<CompressType>(compress_type));
    dst_tensor->set_compressed_size(src_tensor.data()->size());
  }
  return dst_tensor;
}

int LiteSession::ConvertTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto lite_model = reinterpret_cast<const lite::LiteModel *>(model);
  uint32_t tensor_count = model->graph_.all_tensors_.size();
  auto model_input_indices = model->graph_.input_indices_;
  auto model_output_indices = model->graph_.output_indices_;

  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *src_tensor = model->graph_.all_tensors_[i];
    if (src_tensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return RET_NULL_PTR;
    }
    auto *dst_tensor = ConvertTensor(*src_tensor);
    if (dst_tensor == nullptr) {
      MS_LOG(ERROR) << "Convert new " << i << "th tensor failed!";
      return RET_NULL_PTR;
    }
    auto ret = ConvertTensorsData(lite_model, i, dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert data of " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }
    ConvertTensorsQuantParam(src_tensor, dst_tensor);
    if (IsContain(model_input_indices, i)) {
      dst_tensor->set_category(Category::GRAPH_INPUT);
    }
    if (IsContain(model_output_indices, i)) {
      // a tensor is as both input and output, would be treated as an input.
      if (!dst_tensor->IsGraphInput()) {
        dst_tensor->set_category(Category::GRAPH_OUTPUT);
      }
    }

    ret = CheckTensorValid(dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Check " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }

    this->tensors_.emplace_back(dst_tensor);
  }
  return RET_OK;
}

void LiteSession::InitGraphInputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto graph_in_size = model->graph_.input_indices_.size();
  for (size_t i = 0; i < graph_in_size; ++i) {
    auto in_tensor_idx = model->graph_.input_indices_[i];
    MS_ASSERT(in_tensor_idx < this->tensors_.size());
    auto *in_tensor = this->tensors_.at(in_tensor_idx);
    MS_ASSERT(in_tensor != nullptr);
    this->inputs_.emplace_back(in_tensor);
  }
}

void LiteSession::InitGraphInputMSTensors() {
  MS_ASSERT(this->input_vec_.empty());
  for (auto &input_tensor : this->inputs_) {
    MS_ASSERT(input_tensor != nullptr);
    this->input_vec_.emplace_back(input_tensor);
  }
}

void LiteSession::InitGraphOutputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->outputs_.empty());
  auto graph_out_size = model->graph_.output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    auto out_tensor_idx = model->graph_.output_indices_[i];
    MS_ASSERT(out_tensor_idx < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(out_tensor_idx);
    MS_ASSERT(out_tensor != nullptr);
    this->outputs_.emplace_back(out_tensor);
  }
}

void LiteSession::InitGraphInputMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->input_map_.empty());
  MS_ASSERT(this->input_shape_map_.empty());
  auto graph_input_node_indexes = GetGraphInputNodes(model);
  auto graph_in_size = model->graph_.input_indices_.size();
  for (auto in_node_index : graph_input_node_indexes) {
    auto in_node = model->graph_.all_nodes_[in_node_index];
    MS_ASSERT(in_node != nullptr);
    auto in_size = in_node->input_indices_.size();
    for (size_t i = 0; i < in_size; ++i) {
      if (this->input_map_.find(in_node->name_ + std::to_string(i)) != this->input_map_.end()) {
        MS_LOG(ERROR) << "cant find input " << in_node->name_ + std::to_string(i) << "at input_map_";
        return;
      }
      auto in_tensor_index = size_t(in_node->input_indices_[i]);
      bool is_graph_input = false;
      for (size_t j = 0; j < graph_in_size; ++j) {
        if (in_tensor_index == model->graph_.input_indices_[j]) {
          is_graph_input = true;
          break;
        }
      }
      if (!is_graph_input) {
        continue;
      }
      MS_ASSERT(in_tensor_index < this->tensors_.size());
      auto *in_tensor = this->tensors_.at(in_tensor_index);
      if (in_tensor == nullptr) {
        MS_LOG(ERROR) << "in_tensor is null!";
        return;
      }
      auto tensor_name = in_node->name_ + std::to_string(i);
      this->input_map_[tensor_name] = in_tensor;
      this->input_shape_map_[in_tensor] = in_tensor->shape();
      if (!in_tensor->tensor_name().empty()) {
        this->input_map_[in_tensor->tensor_name()] = in_tensor;
      }
    }
  }

  for (auto input_tensor : this->inputs_) {
    MS_ASSERT(input_tensor != nullptr);
    if (this->input_map_.find(input_tensor->tensor_name()) == this->input_map_.end()) {
      this->input_map_[input_tensor->tensor_name()] = input_tensor;
    }
    if (this->input_shape_map_.find(input_tensor) == this->input_shape_map_.end()) {
      this->input_shape_map_[input_tensor] = input_tensor->shape();
    }
  }
}

void LiteSession::InitGraphOutputNodeMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto graph_output_node_indexes = GetGraphOutputNodes(model);
  auto graph_out_size = model->graph_.output_indices_.size();
  for (auto out_node_index : graph_output_node_indexes) {
    auto out_node = model->graph_.all_nodes_[out_node_index];
    MS_ASSERT(out_node != nullptr);
    auto out_size = out_node->output_indices_.size();
    for (size_t i = 0; i < out_size; ++i) {
      auto out_tensor_index = out_node->output_indices_[i];
      bool is_graph_output = false;
      for (size_t j = 0; j < graph_out_size; ++j) {
        if (out_tensor_index == model->graph_.output_indices_[j]) {
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        continue;
      }
      MS_ASSERT(out_tensor_index < this->tensors_.size());
      auto *out_tensor = this->tensors_.at(out_tensor_index);
      if (out_tensor == nullptr) {
        MS_LOG(ERROR) << "out_tensor is null!";
        return;
      }
      this->output_node_map_[out_node->name_].emplace_back(out_tensor);
    }
  }
}

void LiteSession::InitGraphOutputTensorMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->output_tensor_map_.empty());
  auto graph_out_size = model->graph_.output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->graph_.output_indices_[i];
    MS_ASSERT(graph_out_index < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(graph_out_index);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "out_tensor is null!";
      return;
    }
    if (!out_tensor->tensor_name().empty()) {
      this->output_tensor_map_.insert(std::make_pair(out_tensor->tensor_name(), out_tensor));
      this->output_tensor_names_.emplace_back(out_tensor->tensor_name());
    } else {
      this->output_tensor_map_.insert(std::make_pair(std::to_string(graph_out_index), out_tensor));
      this->output_tensor_names_.emplace_back(std::to_string(graph_out_index));
    }
  }
}

void LiteSession::InitGraphInOutTensorsMap(const lite::Model *model) {
  InitGraphInputMSTensors();
  InitGraphInputMap(model);
  InitGraphOutputNodeMap(model);
  InitGraphOutputTensorMap(model);
}

int LiteSession::IsolateOutputTensor() {
  for (Tensor *src_tensor : outputs_) {
    if (src_tensor->IsGraphInput()) {
      continue;
    }
    Tensor *new_tensor = new (std::nothrow)
      Tensor(src_tensor->data_type(), src_tensor->shape(), src_tensor->format(), Category::GRAPH_OUTPUT);
    if (MS_UNLIKELY(new_tensor == nullptr)) {
      MS_LOG(ERROR) << "duplicate new output failed.";
      return RET_NULL_PTR;
    }
    new_tensor->set_allocator(src_tensor->allocator()); /* GPU use opencl allocator */
    new_tensor->set_tensor_name(src_tensor->tensor_name() + "_duplicate");
    for (LiteQuantParam quant : src_tensor->quant_params()) {
      new_tensor->AddQuantParam(quant);
    }
    new_tensor->set_init_ref_count(src_tensor->init_ref_count());

    /* src tensor set for graph calculate */
#ifdef ENABLE_FP16
    if (src_tensor->data_type() == kNumberTypeFloat16) {
      src_tensor->set_data_type(kNumberTypeFloat32);
    }
#endif
    src_tensor->set_ref_count(1);

    isolate_graph_output_map_.insert(std::make_pair(new_tensor, src_tensor));

    /* set new tensor for calculate */
    for (auto subgraph : kernels_) {
      /* subgraph input and output */
      auto in_size = subgraph->in_tensors().size();
      for (size_t i = 0; i < in_size; ++i) {
        if (subgraph->in_tensors()[i] == src_tensor) {
          subgraph->set_in_tensor(new_tensor, i);
        }
      }
      auto out_size = subgraph->out_tensors().size();
      for (size_t i = 0; i < out_size; ++i) {
        if (subgraph->out_tensors()[i] == src_tensor) {
          subgraph->set_out_tensor(new_tensor, i);
        }
      }
      if (subgraph->desc().arch == kernel::kDelegate) {
        continue;
      }
      /* node input and output */
      auto nodes = reinterpret_cast<kernel::SubGraphKernel *>(subgraph)->nodes();
      auto nodes_size = nodes.size();
      for (size_t i = 0; i < nodes_size; ++i) {
        auto node = nodes[i];
        out_size = node->out_tensors().size();
        for (size_t j = 0; j < out_size; ++j) {
          if (node->out_tensors()[j] == src_tensor) {
            node->set_out_tensor(new_tensor, j);
            break;
          }
        }
        in_size = node->in_tensors().size();
        for (size_t j = 0; j < in_size; ++j) {
          if (node->in_tensors()[j] == src_tensor) {
            node->set_in_tensor(new_tensor, j);
          }
        }
      }
    }
  }

  UpdateLinkInfoForIsolateOutput();
  return RET_OK;
}

void LiteSession::UpdateLinkInfoForIsolateOutput() {
  for (auto &item : isolate_graph_output_map_) {
    context_->ReplaceLinkInfoReceiverWithNewOne(item.first, item.second);
  }
  return;
}

void LiteSession::FreePackOpWeight(const std::vector<kernel::KernelExec *> &kernels) {
  // For reducing runtime RAM
  // free pack-op weight because pack-op will not access origin weight in runtime
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (!IsPackedOp(static_cast<int>(kernel->type()))) {
        continue;
      }
    } else {
      auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      FreePackOpWeight(subgraph->nodes());
    }
    auto inputs = kernel->in_tensors();
    for (auto *tensor : inputs) {
      MS_ASSERT(tensor != nullptr);
      if (!tensor->IsConst() || tensor->ref_count() >= 1) {
        continue;
      }
      tensor->FreeData();
    }
  }
}

void LiteSession::MarkSharedWeight(const std::vector<kernel::KernelExec *> &kernels) {
  // For reducing runtime RAM
  // free pack-op weight because pack-op will not access origin weight in runtime
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (IsPackedOp(static_cast<int>(kernel->type()))) {
        continue;
      }
    } else {
      auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      MarkSharedWeight(subgraph->nodes());
    }
    auto inputs = kernel->in_tensors();
    for (auto *tensor : inputs) {
      MS_ASSERT(tensor != nullptr);
      if (tensor->IsConst()) {
        tensor->IncRefCount();
      }
    }
  }
}

int LiteSession::CompileGraph(Model *model) {
  auto ret = PreCheck(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "schedule check failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  if (model->model_type_ != ModelType_MSLite) {
    ret = reinterpret_cast<AbstractBaseModel *>(model)->ConvertTensors(&this->tensors_);
  } else {
    // Convert to abstract base model interface
    ret = ConvertTensors(model);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertTensors failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  ret = lite::PackWeightManager::GetInstance()->StoreOriginTensorData(model, &tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StoreOriginTensorData failed.";
    return RET_ERROR;
  }
  InitGraphInputTensors(model);
  InitGraphOutputTensors(model);

  // scheduler kernels
  Scheduler scheduler(context_.get(), ms_context_, model, &tensors_, &inputs_, &outputs_, is_train_session_,
                      &is_infershape_, &is_control_flow_, &infer_along_running_, execution_plan_, delegate_,
                      delegate_device_type_);
  scheduler.SetupSchedulerCb(std::move(sched_cb_));
  scheduler.SetConfig(config_info_);
  ret = scheduler.Schedule(&kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  infer_along_running_ = infer_along_running_ && !is_control_flow_ && !is_train_session_;
  InitGraphInOutTensorsMap(model);

  non_tail_call_kernels_ = scheduler.NonTailCallNodes();

  ret = PrepareKernels(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  if (is_train_session_ || is_prepare_session_) {
    is_running_.store(false);
    return RET_OK;
  }

  ret = InitExecutor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitExecutor failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  MarkSharedWeight(kernels_);
  FreePackOpWeight(kernels_);

  ret = RuntimeAllocatorInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Runtime allocator init failed.";
    is_running_.store(false);
    return ret;
  }
  infer_along_running_ = infer_along_running_ && !is_control_flow_ && !is_train_session_;
  if (infer_along_running_) {
    this->context_->set_infer_checker(InferCheckerAll);
  }
  is_running_.store(false);
  return RET_OK;
}

bool LiteSession::IsIsolatedSubGraph(const kernel::KernelExec *kernel) {
  auto cur_in_tensors = kernel->in_tensors();
  for (auto cur_kernel : this->kernels_) {
    if (cur_kernel == kernel) {
      continue;
    }
    auto out_tensors = cur_kernel->out_tensors();
    for (auto tensor : cur_in_tensors) {
      if (IsContain(out_tensors, tensor)) {
        return false;
      }
    }
  }
  return true;
}

int LiteSession::SetAllocatorForDelegateKernels(const kernel::KernelExec *kernel) {
  if (kernel == nullptr) {
    return RET_NULL_PTR;
  }
  for (auto input : kernel->in_tensors()) {
    CHECK_NULL_RETURN(input);
    input->set_allocator(this->context_->allocator);
  }
  for (auto output : kernel->out_tensors()) {
    CHECK_NULL_RETURN(output);
    output->set_allocator(this->context_->allocator);
  }
  return RET_OK;
}

int LiteSession::PrepareKernels(const Model *model) {
  // find kernel's in_kernels and out_kernels in every subgraph
  kernel::KernelExecUtil::FindAllInoutKernelsInSubgraphKernel(this->kernels_);
  // find in_kernels and out_kernels between subgraph kernels
  kernel::KernelExecUtil::FindAllInoutKernels(this->kernels_);

  // init init_ref_count for subgraphs and kernels
  auto ret = SetTensorInitRefCount();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetTensorInitRefCount failed.";
    return ret;
  }

  for (auto kernel : this->kernels_) {
    if (kernel->desc().arch == kernel::kDelegate) {
      ret = SetAllocatorForDelegateKernels(kernel);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Prepare kernel " << kernel->name() << " failed: " << ret;
        return ret;
      }
    }

    if (!is_train_session_ && kernel->desc().arch != kernel::kDelegate && kernel->desc().arch != kernel::kGPU) {
      auto subgraph_kernel = static_cast<kernel::SubGraphKernel *>(kernel);
      if (subgraph_kernel == nullptr) {
        MS_LOG(ERROR) << "kernel: " << kernel->name() << " not is subgraph kernel.";
        return RET_ERROR;
      }
      for (auto &node : subgraph_kernel->nodes()) {
        ret = node->Prepare();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "node: " << node->name() << " prepare failed.";
          return ret;
        }
      }
    }
    ret = kernel->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Prepare kernel " << kernel->name() << " failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

int LiteSession::SetTensorInitRefCount() {
  for (auto *kernel : this->kernels_) {
    kernel->InitOutTensorInitRefCount();
    if (kernel->desc().arch == kernel::kDelegate) {
      continue;
    }
    if (IsIsolatedSubGraph(kernel)) {
      static_cast<kernel::SubGraphKernel *>(kernel)->InitInputTensorInitRefCount();
    }
  }

  if (!non_tail_call_kernels_.empty()) {
    return SetNonTaiCallSubgraphOutputInitRefCount();
  }
  return RET_OK;
}

int LiteSession::SetNonTaiCallSubgraphOutputInitRefCount() {
  for (auto call_kernel : non_tail_call_kernels_) {
    auto call_output = call_kernel->out_tensors();
    auto all_out_subgraphs = kernel::KernelExecUtil::GetCallInputPartialsCorrespondingOutputSubgraph(call_kernel);
    for (auto subgraph : all_out_subgraphs) {
      MS_CHECK_TRUE_MSG(subgraph->out_tensors().size() == call_output.size(), RET_ERROR,
                        "non tail call output size is not same as subgraph output.");
      std::set<Tensor *> subgraph_outputs_set{};
      for (size_t i = 0; i < subgraph->out_tensors().size(); ++i) {
        auto output = subgraph->out_tensors()[i];
        if (subgraph_outputs_set.find(output) == subgraph_outputs_set.end()) {
          output->set_init_ref_count(1);
          (void)subgraph_outputs_set.insert(output);
        } else {
          output->set_init_ref_count(output->init_ref_count() + 1);
        }
      }
    }
  }
  return RET_OK;
}

std::vector<mindspore::lite::Tensor *> LiteSession::GetInputs() const { return this->input_vec_; }

int LiteSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  ParallelThreadPoolManager::GetInstance()->ActivatePool(runner_id_, worker_id_);
  STATUS ret = CheckTensorsInvalid(inputs_);
  if (MS_UNLIKELY(ret != RET_OK)) {
    is_running_.store(false);
    MS_LOG(ERROR) << "CheckInputs failed.";
    return ret;
  }
  ret = CheckGraphInputShapes(inputs_, input_shape_map_);
  if (MS_UNLIKELY(ret != RET_OK)) {
    is_running_.store(false);
    MS_LOG(ERROR) << "Check graph input shapes failed.";
    return ret;
  }
  MS_ASSERT(this->context_ != nullptr);
  ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, before, after);
  if (MS_UNLIKELY(ret != RET_OK)) {
    MS_LOG(ERROR) << "RunGraph failed : " << ret;
  }
  if (infer_along_running_) {
    this->context_->set_infer_checker(InferCheckerInput);
    for (auto input : inputs_) {
      input->set_shape_changed(false);
    }
  }
  ParallelThreadPoolManager::GetInstance()->SetFreePool(runner_id_, worker_id_);
  is_running_.store(false);
  return ret;
}

int LiteSession::InitSharedThreadPool() {
  int workers_num = -1;
  int remaining_thread_num = -1;
  int thread_num_limit = -1;
  bool enable_shared_pool = false;
  if (config_info_ != nullptr) {
    auto runner_info_item = config_info_->find(kInnerModelParallelRunnerSection);
    if (runner_info_item != config_info_->end()) {
      auto item_runner = runner_info_item->second.find(kInnerRunnerIDKey);
      if (item_runner != runner_info_item->second.end()) {
        runner_id_ = runner_info_item->second.at(kInnerRunnerIDKey);
      }
      auto shared_pool_item = runner_info_item->second.find(kEnableSharedThreadPoolKey);
      if (shared_pool_item != runner_info_item->second.end() &&
          runner_info_item->second.at(kEnableSharedThreadPoolKey) == "true") {
        workers_num = std::atoi(runner_info_item->second.at(kInnerWorkerNumKey).c_str());
        remaining_thread_num = std::atoi(runner_info_item->second.at(kThreadNumRemainingPerWorkerKey).c_str());
        thread_num_limit = std::atoi(runner_info_item->second.at(kThreadNumLimitPerWorkerKey).c_str());
        worker_id_ = std::atoi(runner_info_item->second.at(kInnerModelIDKey).c_str());
        enable_shared_pool = true;
      }
    }
  }
  MS_LOG(INFO) << "runner id: " << runner_id_ << "  enable_shared_pool: " << enable_shared_pool
               << "  workers_num: " << workers_num << "  thread_num_limit: " << thread_num_limit
               << "  remaining_thread_num: " << remaining_thread_num;
  ParallelThreadPoolManager::GetInstance()->Init(enable_shared_pool, runner_id_, workers_num, remaining_thread_num,
                                                 thread_num_limit);
  return RET_OK;
}

int LiteSession::ContextInit(const std::shared_ptr<InnerContext> &context) {
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    return RET_NULL_PTR;
  }
  this->context_ = context;
  context_->SetBindRunnerId(runner_id_);
  auto ret = this->context_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    return ret;
  }

  ms_context_ = MSContextFromContext(context);
  if (ms_context_ == nullptr) {
    MS_LOG(ERROR) << "transfer context to ms context failed.";
    return RET_NULL_PTR;
  }

#ifdef MS_COMPILE_IOS
  context_->thread_pool_->SetMaxSpinCount(kDefaulLiteIosSpinCount);
  context_->thread_pool_->SetMinSpinCount(kDefaulLiteIosSpinCount);
#endif

  if (context_->inter_op_parallel_num_ > 1 && !runner_id_.empty() &&
      ParallelThreadPoolManager::GetInstance()->GetEnableSharedThreadPool(runner_id_)) {
    MS_LOG(INFO) << "Enable subgraph parallelism and enable thread pool sharing";
    ParallelThreadPoolManager::GetInstance()->BindPoolToRunner(context_->thread_pool_, config_info_);
  }

  return RET_OK;
}

int LiteSession::AscendInit(const std::shared_ptr<InnerContext> &context) {
#ifndef __ANDROID__
  if (!context->IsDeviceTypeEnabled(DT_ASCEND)) {
    MS_LOG(INFO) << "There is no Ascend device type.";
    return RET_OK;
  }
  return mindspore::AscendKernelPlugin::GetInstance().Register();
#else
  return RET_OK;
#endif
}

int LiteSession::CreateTensorRTDelegate() {
#if GPU_TENSORRT
  std::string cache_model_path;
  std::string serialize_path;
  size_t vocab_size = 0;
  size_t device_cache_size = 0;
  std::map<std::string, std::string> input_ranges;
  if (config_info_ != nullptr) {
    auto input_ranges_iter = config_info_->find(kGPUContextSection);
    if (input_ranges_iter != config_info_->end()) {
      input_ranges = input_ranges_iter->second;
    }
    auto ms_cache_iter = config_info_->find(kMSCacheSection);
    if (ms_cache_iter != config_info_->end()) {
      auto ms_cache = ms_cache_iter->second;
      auto model_path_iter = ms_cache.find(kMSCacheModelPathKey);
      if (model_path_iter != ms_cache.end()) {
        cache_model_path = model_path_iter->second;
      }

      auto vocab_size_iter = ms_cache.find(kMSCacheVocabSizeKey);
      if (vocab_size_iter != ms_cache.end()) {
        auto vocab_size_opt = GenericParseValue<size_t>(vocab_size_iter->second);
        if (!vocab_size_opt.IsNone()) {
          vocab_size = vocab_size_opt.Get();
        }
      }

      auto device_cache_size_iter = ms_cache.find(kMSCacheDeviceSizeKey);
      if (device_cache_size_iter != ms_cache.end()) {
        auto device_cache_size_opt = GenericParseValue<size_t>(device_cache_size_iter->second);
        if (!device_cache_size_opt.IsNone()) {
          device_cache_size = device_cache_size_opt.Get();
        }
      }

      auto serialize_path_iter = ms_cache.find(kMSCacheSerializePathKey);
      if (serialize_path_iter != ms_cache.end()) {
        serialize_path = serialize_path_iter->second;
      }
    }
  }

  delegate_ = std::make_shared<TensorRTDelegate>(ms_context_, cache_model_path, vocab_size, device_cache_size,
                                                 serialize_path, input_ranges);
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "New tensorrt delegate_ failed";
    return RET_ERROR;
  }
  delegate_device_type_ = DT_GPU;
  this->context_->delegate = delegate_;
#endif
  return RET_OK;
}

int LiteSession::CreateNPUDelegate() {
#if SUPPORT_NPU
  delegate_ = std::make_shared<NPUDelegate>(context_->GetDeviceInfo(DT_NPU).npu_device_info_);
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "New delegate_ failed";
    return RET_ERROR;
  }
  delegate_device_type_ = DT_NPU;
  this->context_->delegate = delegate_;
#endif
  return RET_OK;
}

int LiteSession::CreateNNAPIDelegate() {
#if SUPPORT_NNAPI
  bool enable_fp16 =
    context_->IsCpuFloat16Enabled() || context_->IsGpuFloat16Enabled() || context_->IsNpuFloat16Enabled();
  bool only_acc_device = !context_->IsDeviceTypeEnabled(DT_CPU) && !context_->IsDeviceTypeEnabled(DT_GPU) &&
                         context_->IsDeviceTypeEnabled(DT_NPU);
  bool disable_cpu = !context_->IsDeviceTypeEnabled(DT_CPU);
  auto providers = context_->GetProviders();
  std::vector<std::string> specified_devices(providers.begin(), providers.end());
  delegate_ = std::make_shared<NNAPIDelegate>(enable_fp16, only_acc_device, disable_cpu, specified_devices);
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "New delegate_ failed";
    return RET_ERROR;
  }
  this->context_->delegate = delegate_;
#endif
  return RET_OK;
}

int LiteSession::CreateCoreMLDelegate() {
#ifdef ENABLE_COREML
  delegate_ = std::make_shared<CoreMLDelegate>();
  if (delegate_ == nullptr) {
    MS_LOG(ERROR) << "New delegate_ failed";
    return RET_ERROR;
  }
  this->context_->delegate = delegate_;
#endif
  return RET_OK;
}

int LiteSession::DelegateInit() {
#ifndef DELEGATE_CLIP
  int ret = RET_OK;
  if (context_->delegate != nullptr) {
    delegate_ = context_->delegate;
    delegate_device_type_ = -1;
  } else if (context_->delegate_mode_ != kNoDelegate) {
    switch (context_->delegate_mode_) {
      case kNNAPI:
        ret = CreateNNAPIDelegate();
        break;
      case kCoreML:
        ret = CreateCoreMLDelegate();
        break;
      default:
        MS_LOG(ERROR) << "Unsupported built-in delegate mode: " << context_->delegate_mode_;
        return RET_ERROR;
    }
  } else {
    if (context_->IsDeviceTypeEnabled(DT_NPU)) {
      ret = CreateNPUDelegate();
    } else if (context_->IsDeviceTypeEnabled(DT_GPU)) {
      ret = CreateTensorRTDelegate();
    }
  }

  if (ret != RET_OK) {
    return ret;
  }
  if (delegate_ != nullptr) {
    auto delegate_ret = delegate_->Init();
    if (delegate_ret == mindspore::kLiteNotSupport) {
      MS_LOG(DEBUG) << "Delegate is unsupported";
      delegate_.reset();
      delegate_ = nullptr;
    } else if (delegate_ret == mindspore::kSuccess) {
      MS_LOG(INFO) << "Delegate init successfully";
    } else {
      MS_LOG(ERROR) << "Delegate init failed";
      return RET_ERROR;
    }
  }
#endif
  return RET_OK;
}

int LiteSession::Init(const std::shared_ptr<InnerContext> &context) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }

  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "Device not support isa";
    return RET_NOT_SUPPORT;
  }

  auto status = InitSharedThreadPool();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init Shared thread pool failed";
    is_running_.store(false);
    return status;
  }
  auto ret = ContextInit(context);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    is_running_.store(false);
    return ret;
  }

  ret = AscendInit(context);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Open Ascend kernel plugin failed";
    return ret;
  }

  ret = DelegateInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init delegate failed.";
    is_running_.store(false);
    return ret;
  }

  ret = InitGPURuntime();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init GPU runtime failed.";
    is_running_.store(false);
    return ret;
  }

  is_running_.store(false);
  return RET_OK;
}

void LiteSession::BindThread(bool if_bind) {
  // Abandoned code
  // Bind thread in executor
  return;
}

LiteSession::~LiteSession() {
  delegate_.reset();
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return;
  }
  for (auto *kernel : kernels_) {
    delete kernel;
    kernel = nullptr;
  }
  for (auto tensor : tensors_) {
    if (tensor == nullptr) {
      continue;
    }
    // Data of const tensor which doesn't own data will not freed.
    // Such as const data from meta_graph which will be freed when freeing meta_graph.
    if (tensor->IsConst() && !tensor->own_data()) {
      tensor->set_data(nullptr);
    }

    /* situation : user set graph-output-tensor data */
    if (tensor->IsGraphOutput() && tensor->allocator() == nullptr) {
      tensor->set_data(nullptr);
    }
    delete tensor;
    tensor = nullptr;
  }

  for (auto item : isolate_graph_output_map_) {
    auto isolate_output_tensor = item.first;
    isolate_output_tensor->set_data(nullptr);
    delete isolate_output_tensor;
    isolate_output_tensor = nullptr;
  }

  for (auto map : isolate_input_map_) {
    auto isolate_input_tensor = map.first;
    isolate_input_tensor->set_data(nullptr);
    delete isolate_input_tensor;
  }

  // Tensor * in input_map output_map are freed in tensors
  input_map_.clear();
  input_shape_map_.clear();
  output_node_map_.clear();
  output_tensor_map_.clear();
  input_vec_.clear();
  isolate_graph_output_map_.clear();

  delete this->executor_;
  this->executor_ = nullptr;
#if GPU_OPENCL
  delete opencl_runtime_wrapper_;
  opencl_runtime_wrapper_ = nullptr;
#endif
  delete ms_context_;
  ms_context_ = nullptr;
  ParallelThreadPoolManager::GetInstance()->ResetParallelThreadPoolManager(runner_id_);
  lite::PackWeightManager::GetInstance()->FreePackWeight(runner_id_, model_id_);
  if (model_ != nullptr && is_shared_weight_) {
    model_->buf = nullptr;
  }
  delete (model_);
  model_ = nullptr;
  is_running_.store(false);
}

mindspore::lite::Tensor *LiteSession::GetInputsByTensorName(const std::string &name) const {
  auto ret = input_map_.find(name);
  if (ret == input_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << name << " is not exist";
    return nullptr;
  }
  return ret->second;
}

std::vector<mindspore::lite::Tensor *> LiteSession::GetOutputsByNodeName(const std::string &node_name) const {
  auto ret = output_node_map_.find(node_name);
  if (ret == output_node_map_.end()) {
    MS_LOG(WARNING) << "Node  " << node_name << " is not an output node";
    std::vector<mindspore::lite::Tensor *> empty_ret;
    return empty_ret;
  }
  return ret->second;
}

std::vector<std::string> LiteSession::GetOutputTensorNames() const { return this->output_tensor_names_; }

mindspore::lite::Tensor *LiteSession::GetOutputByTensorName(const std::string &tensor_name) const {
  auto ret = output_tensor_map_.find(tensor_name);
  if (ret == output_tensor_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << tensor_name << " is not an output node";
    return nullptr;
  }
  return ret->second;
}

std::unordered_map<std::string, mindspore::lite::Tensor *> LiteSession::GetOutputs() const {
  return this->output_tensor_map_;
}

int LiteSession::UpdateInputShapeMap() {
  for (auto input : inputs_) {
    MS_CHECK_TRUE_MSG(input != nullptr, RET_ERROR, "graph input tensor is nullptr.");
    if (input_shape_map_.find(input) != input_shape_map_.end()) {
      input_shape_map_.at(input) = input->shape();
    } else {
      MS_LOG(ERROR) << "can't find " << input->tensor_name() << " in input_shape_map";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LiteSession::ResizeInputs(const std::vector<mindspore::lite::Tensor *> &inputs,
                              const std::vector<std::vector<int>> &dims) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "Inputs size " << inputs.size() << " is not equal to " << inputs_.size();
    return RET_PARAM_INVALID;
  }

  if (dims.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input dims size " << dims.size() << " is not equal to the inputs size " << inputs.size();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] != inputs_[i]) {
      MS_LOG(ERROR) << "Input[" << i << "] tensor is not equal to the inputs have been saved!";
      return RET_PARAM_INVALID;
    }
    inputs_[i]->FreeData();
    if (infer_along_running_ && !inputs_[i]->get_shape_changed()) {
      inputs_[i]->set_shape_changed(dims[i] != inputs_[i]->shape());
    }
    inputs_[i]->set_shape(dims[i]);
  }
  if (!is_train_session_) {
    executor_->Resize(inputs, dims);
  }
  return RET_OK;
}

void LiteSession::ResetInputsShape(const std::vector<std::vector<int>> &dims) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->FreeData();
    inputs_[i]->set_shape(dims[i]);
    inputs_[i]->set_shape_changed(false);
  }
}

int LiteSession::ReSizeKernels(const std::vector<kernel::KernelExec *> &kernels,
                               const std::unordered_map<Tensor *, Tensor *> &isolate_input_map) {
  for (auto kernel : kernels) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    auto ret = RET_OK;
    if (kernel->desc().arch == kernel::kDelegate) {
      ret = kernel->ReSize();
    } else {
      // resize subgraph inputs
      auto sub_graph_kernel = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      for (auto input : sub_graph_kernel->in_tensors()) {
        if (isolate_input_map.find(input) != isolate_input_map.end()) {
          input->set_shape(isolate_input_map.at(input)->shape());
        }
      }
      if (kernel->subgraph_type() == kernel::kGpuFp16SubGraph || kernel->subgraph_type() == kernel::kGpuFp32SubGraph) {
#if GPU_OPENCL
        auto sub_graph = reinterpret_cast<kernel::OpenCLSubGraph *>(kernel);
        ret = sub_graph->ReSize(false);
#endif
      } else {
        auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
        ret = sub_graph->ReSize();
      }
    }
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape is interrupted";
      continue;
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReSize node " << kernel->name() << " failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

void LiteSession::SynIsolateInOutputDataType() {
  for (auto &tensor_map : isolate_input_map_) {
    auto dst_tensor = tensor_map.second;
    auto src_tensor = tensor_map.first;

    src_tensor->set_data_type(dst_tensor->data_type());
  }

  for (auto &tensor_map : isolate_graph_output_map_) {
    auto dst_tensor = tensor_map.second;
    auto src_tensor = tensor_map.first;

    src_tensor->set_data_type(dst_tensor->data_type());
  }
}

int LiteSession::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                       std::map<std::string, unsigned int> *outputGLTexture) {
#if GPU_OPENCL
  if (!this->context_->GetDeviceInfo(DT_GPU).gpu_device_info_.enable_gl_texture_) {
    MS_LOG(ERROR) << "the context isn't set to support OpenGL texture";
    return RET_ERROR;
  }
  for (const auto &[name, GLTexture_id] : inputGLTexture) {
    auto iter = input_map_.find(name);
    if (iter == input_map_.end()) {
      MS_LOG(ERROR) << "the in tensor name " << name << "is not match any model input name";
      return RET_ERROR;
    }
    auto in_data = iter->second->MutableData();
    if (in_data == nullptr) {
      std::cout << "MallocData for input Tensor failed" << std::endl;
      return RET_ERROR;
    }
    memcpy(in_data, &GLTexture_id, sizeof(cl_GLuint));
    iter->second->set_data_type(kNumberTypeGLUInt);
  }
  for (auto [name, GLTexture_id] : *outputGLTexture) {
    auto iter = output_tensor_map_.find(name);
    if (iter == output_tensor_map_.end()) {
      MS_LOG(ERROR) << "the out tensor name " << name << "is not match any model output name";
      return RET_ERROR;
    }
    auto out_data = iter->second->MutableData();
    if (out_data == nullptr) {
      std::cout << "MallocData for input Tensor failed" << std::endl;
      return RET_ERROR;
    }
    memcpy(out_data, &GLTexture_id, sizeof(cl_GLuint));
    iter->second->set_data_type(kNumberTypeGLUInt);
  }

#ifdef ENABLE_MINDRT
  SynIsolateInOutputDataType();  // Synchronized input/output with isolate input/output data types
#endif

  if (this->kernels_.size() != 1) {
    MS_LOG(ERROR) << "Now only support one opencl subgraph if you want to input opengl texture";
    return RET_ERROR;
  }
  auto opencl_subgraph = reinterpret_cast<kernel::OpenCLSubGraph *>(kernels_.front());
  for (size_t i = 0; i < outputs_.size(); i++) {
    (opencl_subgraph)->set_out_tensor(outputs_[i], i);
  }
  for (auto node : opencl_subgraph->out_nodes()) {
    node->set_out_tensors(opencl_subgraph->out_tensors());
  }
#endif
  return RET_OK;
}

int LiteSession::Resize(const std::vector<mindspore::lite::Tensor *> &inputs,
                        const std::vector<std::vector<int>> &dims) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  std::vector<std::vector<int>> old_dims;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    old_dims.push_back(inputs_[i]->shape());
  }
  auto ret = ResizeInputs(inputs, dims);
  if (ret != RET_OK) {
    ResetInputsShape(old_dims);
    is_running_.store(false);
    return ret;
  }
  ret = UpdateInputShapeMap();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "update input shape map failed.";
    return RET_ERROR;
  }
  if (infer_along_running_) {
    is_running_.store(false);
    return ret;
  }

  ret = ReSizeKernels(kernels_, isolate_input_map_);
  if (ret != RET_OK) {
    ResetInputsShape(old_dims);
    auto resize_ret = ReSizeKernels(kernels_);
    if (resize_ret != RET_OK) {
      MS_LOG(ERROR) << "restore kernel size fail!ret: " << resize_ret;
    }
    is_running_.store(false);
    return ret;
  }

  if (RuntimeAllocatorInit() != RET_OK) {
    MS_LOG(ERROR) << "Runtime allocator in resize failed.";
    is_running_.store(false);
    return RET_ERROR;
  }

  auto status = GraphOptimizePass(&kernels_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GraphOptimizePass failed.";
    return RET_ERROR;
  }

  is_running_.store(false);
  return RET_OK;
}

int LiteSession::PreCheck(Model *model) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  if (model == nullptr) {
    MS_LOG(ERROR) << "The input model is nullptr.";
    return RET_PARAM_INVALID;
  }
  if (model->buf == nullptr) {
    MS_LOG(ERROR) << "The input model buf is nullptr.";
    return RET_PARAM_INVALID;
  }
  if (model->model_type_ != ModelType_MSLite) {
    // abstract base model
    if (!reinterpret_cast<AbstractBaseModel *>(model)->ModelVerify()) {
      MS_LOG(ERROR) << "wrong model input, please check";
      return RET_ERROR;
    }
  } else {
    // old routine, convert to abstract base model
    if (!reinterpret_cast<LiteModel *>(model)->ModelVerify()) {
      MS_LOG(ERROR) << "wrong model input, please check";
      return RET_ERROR;
    }
  }

#ifndef ENABLE_FP16
  if (context_->GetDeviceInfo(DT_CPU).cpu_device_info_.enable_float16_) {
    MS_LOG(WARNING) << unsupport_fp16_log;
  }
#endif
  return RET_OK;
}

int LiteSession::InitExecutor() {
  int ret;
#ifdef ENABLE_MINDRT
  ret = IsolateOutputTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Isolate output tensor failed.";
    return ret;
  }
  executor_ = new (std::nothrow) MindrtExecutor(&isolate_graph_output_map_, &isolate_input_map_);
#else
  executor_ = new (std::nothrow) Executor();
#endif
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "New Executor failed";
    return RET_ERROR;
  }

  ret = executor_->Prepare(kernels_, inputs_, outputs_, context_.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare executor failed: " << ret;
    return ret;
  }
  return RET_OK;
}

int LiteSession::RuntimeAllocatorValid() {
#ifdef ENABLE_ARM32
  MS_LOG(DEBUG) << "Not support runtime allocator in arm32.";
  return RET_ERROR;
#endif

#ifndef ENABLE_MINDRT
  MS_LOG(DEBUG) << "Not support runtime allocator in converter.";
  return RET_ERROR;
#endif

#ifdef BFC_MEMORY
  MS_LOG(DEBUG) << "Not support runtime allocator when BFC_MEMORY on.";
  return RET_ERROR;
#endif

  if ((context_->enable_parallel_ == true) || (context_->inter_op_parallel_num_ > 1)) {
    MS_LOG(DEBUG) << "Not support runtime allocator in subgraph parallel.";
    return RET_ERROR;
  }
  if (is_train_session_ == true) {
    MS_LOG(DEBUG) << "Not support runtime allocator in train session.";
    return RET_ERROR;
  }
  if (is_infershape_ != RET_OK) {
    MS_LOG(DEBUG) << "Not support runtime allocator in runtime-infershape.";
    return RET_ERROR;
  }
  if (kernels_.size() != 1) {
    MS_LOG(DEBUG) << "Not support runtime allocator in random subgraph sort";
    return RET_ERROR;
  }
#ifdef ENABLE_ARM64
  MS_LOG(DEBUG) << "support runtime allocator.";
  return RET_OK;
#endif
  return RET_ERROR;
}

void LiteSession::RuntimeAllocatorInitGraphOutput() {
  AllocatorPtr default_allocator = context_->allocator;
  for (auto graph_out : isolate_graph_output_map_) {
    auto cal_t = graph_out.first;
    auto out_t = graph_out.second;
    if (cal_t->allocator() != runtime_allocator_ || out_t->allocator() != default_allocator) {
      continue;
    }
    out_t->set_allocator(runtime_allocator_);
    if (cal_t->data_type() != out_t->data_type()) {
      runtime_allocator_->MallocTensorData(out_t);
    }
  }
  return;
}

void RuntimeAllocatorInitSubgraphInputs(const kernel::KernelExec *subgraph, const AllocatorPtr &default_allocator,
                                        const RuntimeAllocatorPtr &runtime_allocator,
                                        const std::unordered_map<Tensor *, Tensor *> &isolate_input_map,
                                        std::unordered_map<Tensor *, int> *tensor_ref_count,
                                        std::unordered_map<size_t, int> *data_ref_count) {
  MS_ASSERT(subgraph != nullptr && tensor_ref_count != nullptr && data_ref_count != nullptr);
  for (auto in_tensor : subgraph->in_tensors()) {
    auto iter = isolate_input_map.find(in_tensor);
    if (isolate_input_map.end() == iter) break;
    auto src_t = iter->second;

    if (src_t->data_type() == in_tensor->data_type()) {
      in_tensor->set_allocator(src_t->allocator());
      if (src_t->allocator() == runtime_allocator) {
        (*tensor_ref_count)[in_tensor] = in_tensor->init_ref_count();
        (*data_ref_count)[runtime_allocator->GetOffsetMap().at(src_t)] += in_tensor->init_ref_count();
        runtime_allocator->SetDataOffset(in_tensor, runtime_allocator->GetOffsetMap().at(src_t));
      }
    } else {
      if (in_tensor->allocator() == default_allocator) {
        in_tensor->set_allocator(runtime_allocator);
        runtime_allocator->MallocTensorData(in_tensor);
        (*tensor_ref_count)[in_tensor] = in_tensor->init_ref_count();
        (*data_ref_count)[runtime_allocator->GetOffsetMap().at(in_tensor)] = in_tensor->init_ref_count();
      }
    }

    if (src_t->allocator() != runtime_allocator) {
      continue;
    }

    (*tensor_ref_count)[src_t]--;
    (*data_ref_count)[runtime_allocator->GetOffsetMap().at(src_t)]--;

    if ((*tensor_ref_count)[src_t] <= 0) {
      if ((*data_ref_count)[runtime_allocator->GetOffsetMap().at(src_t)] <= 0) {
        runtime_allocator->FreeTensorData(src_t);
      }
    }
  }
}

void LiteSession::RuntimeAllocatorInitSubgraph() {
  AllocatorPtr default_allocator = context_->allocator;
  std::unordered_map<lite::Tensor *, int> tensor_ref_count;
  std::unordered_map<size_t, int> data_ref_count;

  for (auto subgraph : kernels_) {
    if (subgraph->desc().arch != kernel::KERNEL_ARCH::kCPU) {
      continue;
    }

    RuntimeAllocatorInitSubgraphInputs(subgraph, default_allocator, runtime_allocator_, isolate_input_map_,
                                       &tensor_ref_count, &data_ref_count);

    auto kernel_list = reinterpret_cast<kernel::SubGraphKernel *>(subgraph)->nodes();
    for (auto kernel : kernel_list) {
      /* malloc for output */
      for (auto tensor : kernel->out_tensors()) {
        if (tensor->allocator() != default_allocator || tensor->IsConst()) {
          continue;
        }
        tensor->set_allocator(runtime_allocator_);
        runtime_allocator_->MallocTensorData(tensor);
        tensor_ref_count[tensor] = tensor->init_ref_count();
        data_ref_count[runtime_allocator_->GetOffsetMap().at(tensor)] = tensor->init_ref_count();
      }

      /* free input after run */
      for (auto tensor : kernel->in_tensors()) {
        if (tensor->allocator() != runtime_allocator_) {
          continue;
        }
        tensor_ref_count[tensor]--;
        data_ref_count[runtime_allocator_->GetOffsetMap().at(tensor)]--;

        if (tensor_ref_count[tensor] <= 0 && tensor->allocator() == runtime_allocator_) {
          if (data_ref_count[runtime_allocator_->GetOffsetMap().at(tensor)] <= 0) {
            runtime_allocator_->FreeTensorData(tensor);
          }
        }
      }
    }
  }
  return;
}

int LiteSession::RuntimeAllocatorInit() {
  if (RuntimeAllocatorValid() != RET_OK) {
    return RET_OK;
  }
  if (ExistCustomCpuKernel()) {
    return RET_OK;
  }
  if (runtime_allocator_ == nullptr) {
    runtime_allocator_ = std::shared_ptr<RuntimeAllocator>(new (std::nothrow) RuntimeAllocator());
  } else {
    runtime_allocator_->Clear(context_->allocator);
  }
  if (runtime_allocator_ == nullptr) {
    MS_LOG(ERROR) << "RuntimeAllocator is null.";
    return RET_ERROR;
  }

  RuntimeAllocatorInitSubgraph();

  RuntimeAllocatorInitGraphOutput();

  auto ret = RuntimeAllocatorSetData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "using optimize allocator failed.";
    return ret;
  }
  return RET_OK;
}

int LiteSession::RuntimeAllocatorSetData() {
  void *data = runtime_allocator_->MallocOptData();
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc optimize data failed.";
    return RET_ERROR;
  }
  int8_t *int8_data = reinterpret_cast<int8_t *>(data);
  auto offset_map = runtime_allocator_->GetOffsetMap();

  for (auto &iter : offset_map) {
    auto tensor = iter.first;
    if (tensor->allocator() != runtime_allocator_) {
      return RET_ERROR;
    }
    tensor->set_data(int8_data + iter.second);
  }
  return RET_OK;
}

int LiteSession::InitGPURuntime() {
  if (context_->IsDeviceTypeEnabled(DT_CPU)) {
    CpuBindMode cpu_bind_mode = context_->GetDeviceInfo(DT_CPU).cpu_device_info_.cpu_bind_mode_;
    ThreadPool *thread_pool = this->context_->thread_pool_;
    if (thread_pool == nullptr) {
      MS_LOG(ERROR) << "thread pool is nullptr";
      is_running_.store(false);
      return RET_NULL_PTR;
    }
    thread_pool->SetProcessAffinity(static_cast<BindMode>(cpu_bind_mode));
  }
#if GPU_OPENCL
  if (this->context_->IsDeviceTypeEnabled(DT_GPU)) {
    opencl_runtime_wrapper_ = new (std::nothrow) opencl::OpenCLRuntimeInnerWrapper();
    if (opencl_runtime_wrapper_ == nullptr) {
      MS_LOG(ERROR) << "create OpenCLRuntimeInnerWrapper failed";
      return RET_ERROR;
    }
    const auto &gpu_device_info = this->context_->GetDeviceInfo(DT_GPU).gpu_device_info_;
    auto opencl_runtime = opencl_runtime_wrapper_->GetInstance();
    opencl_runtime->SetGLTextureEnable(gpu_device_info.enable_gl_texture_);
    opencl_runtime->SetGLContext(gpu_device_info.gl_context_);
    opencl_runtime->SetGLDisplay(gpu_device_info.gl_display_);
    if (opencl_runtime->Init() != RET_OK) {
      if (gpu_device_info.enable_gl_texture_) {
        MS_LOG(ERROR) << "Init OpenCL runtime failed, enable_gl_texture set true, only support GPU mode.";
        return RET_ERROR;
      }
      this->context_->device_list_ = {{DT_CPU, {gpu_device_info.enable_float16_, MID_CPU}}};
      MS_LOG(WARNING) << "Init OpenCL runtime failed, change to CPU mode.";
    } else {
      MS_LOG(INFO) << "Init OpenCL runtime success.";
    }

    opencl_runtime->SetFp16Enable(gpu_device_info.enable_float16_);

    /* check chip support shared memory */
    auto enable_arm_import_memory = opencl_runtime->isExtensionEnable(EXT_ARM_IMPORT_MEMORY_HOST);
    if (!enable_arm_import_memory) {
      MS_LOG(WARNING) << "GPU do not support shared memory!";
    }
  }
#endif
  // Setting the binding core will affect the opencl drive scheduling.
  if (context_->IsDeviceTypeEnabled(DT_CPU)) {
    ThreadPool *thread_pool = this->context_->thread_pool_;
    thread_pool->SetProcessAffinity(static_cast<BindMode>(NO_BIND));
  }
  return RET_OK;
}
}  // namespace lite

lite::LiteSession *lite::LiteSession::CreateSession(const std::shared_ptr<InnerContext> &context) {
  auto session = new (std::nothrow) lite::LiteSession();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    delete session;
    return nullptr;
  }
  return session;
}

lite::LiteSession *lite::LiteSession::CreateSession(const char *model_buf, size_t size,
                                                    const std::shared_ptr<InnerContext> &context) {
  auto *session = lite::LiteSession::CreateSession(context);
  if (session == nullptr) {
    MS_LOG(ERROR) << "Create session failed";
    return nullptr;
  }
  auto ret = reinterpret_cast<lite::LiteSession *>(session)->LoadModelAndCompileByBuf(
    model_buf, mindspore::ModelType::kMindIR_Lite, size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init session failed";
    delete session;
    return nullptr;
  }
  return session;
}

mindspore::ModelType lite::LiteSession::LoadModelByBuff(const char *model_buf, const size_t &buf_size, char **lite_buf,
                                                        size_t *size, mindspore::ModelType model_type) {
  if (model_type == mindspore::ModelType::kMindIR_Lite) {
    *size = buf_size;
    *lite_buf = const_cast<char *>(model_buf);
    return mindspore::ModelType::kMindIR_Lite;
  }

  if (model_type != mindspore::ModelType::kMindIR) {
    return mindspore::ModelType::kUnknownType;
  }

  flatbuffers::Verifier verify((const uint8_t *)model_buf, buf_size, INT32_MAX, INT32_MAX);
  auto version_verify = lite::LiteModel::VersionVerify(&verify);
  if (version_verify != SCHEMA_INVALID) {
    MS_LOG(DEBUG) << "The kMindIR type model buffer is valid mslite model buffer";
    *size = buf_size;
    *lite_buf = const_cast<char *>(model_buf);
    return mindspore::ModelType::kMindIR_Lite;
  }
  MS_LOG(WARNING) << "Invalid mslite model.";

#ifdef RUNTIME_CONVERT
  *lite_buf = RuntimeConvert(model_buf, buf_size, size, ms_context_);
#else
  MS_LOG(WARNING) << "Please enable runtime convert.";
#endif
#ifdef ENABLE_CLOUD_FUSION_INFERENCE
  *size = buf_size;
  *lite_buf = const_cast<char *>(model_buf);
#endif
  return mindspore::ModelType::kMindIR;
}

const char *lite::LiteSession::LoadModelByPath(const std::string &file, mindspore::ModelType model_type, size_t *size) {
  size_t buf_size;
  auto model_buf = lite::ReadFile(file.c_str(), &buf_size);
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "The model path is invalid";
    return model_buf;
  }

  char *lite_buf = nullptr;
  auto buf_model_type = LoadModelByBuff(model_buf, buf_size, &lite_buf, size, model_type);
  if (buf_model_type == mindspore::ModelType::kUnknownType || lite_buf == nullptr) {
    return nullptr;
  }

  return lite_buf;
}

std::string lite::LiteSession::ParseWeightPath() {
  std::string weight_path = "";
  if (config_info_ != nullptr) {
    auto ms_weight = config_info_->find(kConfigModelFileSection);
    if (ms_weight != config_info_->end()) {
      auto ms_weight_iter = ms_weight->second;
      if (ms_weight_iter.find(kConfigMindIRPathKey) != ms_weight_iter.end()) {
        weight_path = ms_weight_iter[kConfigMindIRPathKey];
      }
    }
  }
  return weight_path;
}

#ifdef ENABLE_LITE_HELPER
int lite::LiteSession::LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type,
                                                const size_t &buf_size,
                                                mindspore::infer::helper::InferHelpers *infer_helpers) {
#else
int lite::LiteSession::LoadModelAndCompileByBuf(const char *model_buf, mindspore::ModelType model_type,
                                                const size_t &buf_size) {
#endif
  auto status = lite::PackWeightManager::GetInstance()->InitPackWeightManager(model_buf, buf_size, &model_id_,
                                                                              &runner_id_, config_info_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "InitPackWeightByBuf failed.";
    return RET_ERROR;
  }
  auto new_model_buf =
    lite::PackWeightManager::GetInstance()->GetSharedModelBuf(model_buf, model_id_, config_info_, &is_shared_weight_);
  if (new_model_buf == nullptr) {
    MS_LOG(ERROR) << "get shared model buf is nullptr.";
    return RET_ERROR;
  }
  size_t lite_buf_size = 0;
  char *lite_buf = nullptr;
  auto buf_model_type = LoadModelByBuff(new_model_buf, buf_size, &lite_buf, &lite_buf_size, model_type);
  if (buf_model_type == mindspore::ModelType::kUnknownType || lite_buf == nullptr) {
    MS_LOG(ERROR) << "Invalid model_buf";
    return RET_ERROR;
  }
  auto weight_path = ParseWeightPath();
#ifdef ENABLE_LITE_HELPER
  auto *model = lite::ImportFromBuffer(lite_buf, lite_buf_size, true, model_type, weight_path, infer_helpers);
#else
  auto *model = lite::ImportFromBuffer(lite_buf, lite_buf_size, true, model_type, weight_path);
#endif
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return RET_ERROR;
  }
  (reinterpret_cast<lite::LiteModel *>(model))->set_keep_model_buf(keep_model_buf_);
  auto ret = CompileGraph(model);
  model->buf = nullptr;
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    delete model;
    return RET_ERROR;
  }
  set_model(model);
  return RET_OK;
}

int lite::LiteSession::LoadModelAndCompileByPath(const std::string &model_path, mindspore::ModelType model_type) {
  size_t model_size;
  auto model_buf = LoadModelByPath(model_path, model_type, &model_size);
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed";
    return RET_ERROR;
  }
  auto status = lite::PackWeightManager::GetInstance()->InitPackWeightManager(model_buf, model_size, &model_id_,
                                                                              &runner_id_, config_info_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "InitPackWeightByBuf failed.";
    return RET_ERROR;
  }
  auto new_model_buf =
    lite::PackWeightManager::GetInstance()->GetSharedModelBuf(model_buf, model_id_, config_info_, &is_shared_weight_);
  if (new_model_buf == nullptr) {
    MS_LOG(ERROR) << "get shared model buf is nullptr.";
    return RET_ERROR;
  }
  if (is_shared_weight_) {
    delete[] model_buf;
    model_buf = nullptr;
  }
  auto *model = lite::ImportFromBuffer(new_model_buf, model_size, true, model_type, model_path);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return RET_ERROR;
  }
  (reinterpret_cast<lite::LiteModel *>(model))->set_keep_model_buf(true);
  auto ret = CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    model->buf = nullptr;
    delete model;
    return RET_ERROR;
  }
  set_model(model);
  return RET_OK;
}
}  // namespace mindspore

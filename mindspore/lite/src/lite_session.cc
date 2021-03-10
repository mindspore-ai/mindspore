/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/lite_session.h"
#include <vector>
#include <utility>
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/scheduler.h"
#include "src/runtime/allocator.h"
#include "src/executor.h"
#include "src/common/utils.h"
#include "src/common/prim_util.h"
#include "src/common/graph_util.h"
#include "src/kernel_registry.h"
#include "src/lite_model.h"
#include "src/dequant.h"
#ifdef ENABLE_MINDRT
#include "src/mindrt_executor.h"
#endif
#if SUPPORT_NPU
#include "src/runtime/agent/npu/npu_manager.h"
#include "src/runtime/agent/npu/optimizer/npu_pass_manager.h"
#endif
#if GPU_OPENCL
#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#endif

namespace mindspore {
namespace lite {
// this method will not check whether tensor_idx is a weight tensor index, caller should ensure this.
static bool WeightTensorNeedCopy(const lite::Model *model, const uint32_t tensor_idx) {
#ifdef SUPPORT_TRAIN
  return false;
#endif

  MS_ASSERT(model != nullptr);
  auto post_node_idxes = GetLinkedPostNodeIdx(model, tensor_idx);
  return std::none_of(post_node_idxes.begin(), post_node_idxes.end(), [&](const size_t &post_node_idx) {
    auto node = model->all_nodes_[post_node_idx];
    MS_ASSERT(node != nullptr);
    return IsPackedOp(GetPrimitiveType(node->primitive_));
  });
}

LiteSession::LiteSession() { this->is_running_.store(false); }

void LiteSession::ConvertTensorsQuantParam(const schema::Tensor *src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  auto quant_params = src_tensor->quantParams();
  if (quant_params != nullptr) {
    for (size_t j = 0; j < quant_params->size(); j++) {
      QuantArg quant_arg{};
      quant_arg.bitNum = quant_params->Get(j)->numBits();
      quant_arg.scale = quant_params->Get(j)->scale();
      quant_arg.zeroPoint = quant_params->Get(j)->zeroPoint();
      quant_arg.var_corr = quant_params->Get(j)->varCorr();
      quant_arg.mean_corr = quant_params->Get(j)->meanCorr();
      quant_arg.inited = quant_params->Get(j)->inited();
      quant_arg.roundType = quant_params->Get(j)->roundType();
      quant_arg.multiplier = quant_params->Get(j)->multiplier();
      quant_arg.dstDtype = quant_params->Get(j)->dstDtype();
      dst_tensor->AddQuantParam(quant_arg);
    }
  }
  dst_tensor->set_enable_huffman_code(src_tensor->enableHuffmanCode());
  auto quant_clusters = src_tensor->quantClusters();
  if (quant_clusters != nullptr) {
    std::vector<float> clusters;
    for (size_t j = 0; j < quant_clusters->size(); j++) {
      clusters.push_back(quant_clusters->Get(j));
    }
    dst_tensor->set_quant_clusters(clusters);
  }
}

int LiteSession::ConvertTensorsData(const lite::Model *model, size_t tensor_index, const schema::Tensor *src_tensor,
                                    lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  auto NeedUnPack = [&src_tensor, &dst_tensor]() -> bool {
    auto data_type = src_tensor->dataType();
    int pack_size = src_tensor->data()->size();
    int org_size = dst_tensor->Size();
    return (pack_size != org_size) && (data_type == kNumberTypeInt8 || data_type == kNumberTypeInt16);
  };
  auto src_category = TensorCategory(src_tensor);
  if ((src_category == Tensor::Category::CONST_TENSOR || src_category == Tensor::Category::CONST_SCALAR) &&
      src_tensor->data() != nullptr && src_tensor->data()->size() > 0) {
    if (src_tensor->dataType() == kObjectTypeTensorType) {
      auto tensor_list = reinterpret_cast<TensorList *>(dst_tensor);
      if (src_tensor->data() == nullptr) {
        MS_LOG(ERROR) << "src_tensor->data() is nullptr";
        return RET_ERROR;
      }
      if (tensor_list->Decode(reinterpret_cast<const int *>(src_tensor->data()->data())) != RET_OK) {
        return RET_ERROR;
      }
    } else {
      if (WeightTensorNeedCopy(model, tensor_index)) {
        auto dst_data = dst_tensor->MutableData();
        if (dst_data == nullptr) {
          MS_LOG(ERROR) << "Data from tensor is nullptr";
          return RET_NULL_PTR;
        }
        if (NeedUnPack()) {
          auto ret = DequantUtil::UnPackToInt(src_tensor, dst_data);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "unpack to int failed.";
            return RET_NULL_PTR;
          }
        } else {
          memcpy(dst_data, src_tensor->data()->data(), dst_tensor->Size());
        }
        copyed_tensor_idxes_.emplace_back(tensor_index);
      } else {
        if (NeedUnPack()) {
          auto dst_data = dst_tensor->MutableData();
          if (dst_data == nullptr) {
            MS_LOG(ERROR) << "Data from tensor is nullptr";
            return RET_ERROR;
          }
          auto ret = DequantUtil::UnPackToInt(src_tensor, dst_data);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "unpack to int failed.";
            return RET_ERROR;
          }
          copyed_tensor_idxes_.emplace_back(tensor_index);
        } else {
          dst_tensor->set_data(const_cast<unsigned char *>(src_tensor->data()->data()));
        }
      }
    }
  }
  return RET_OK;
}

lite::Tensor *LiteSession::ConvertTensor(const schema::Tensor &src_tensor) {
  auto src_category = TensorCategory(&src_tensor);
  std::vector<int> shape;
  if (src_tensor.dims() == nullptr) {
    MS_LOG(DEBUG) << "Dims of src_tensor is nullptr";
  }
  if (src_tensor.dims() != nullptr && src_category == Tensor::Category::CONST_TENSOR) {
    if (src_tensor.dataType() == kObjectTypeString && src_tensor.data() != nullptr) {
      shape.push_back(src_tensor.data()->size());
    } else {
      for (size_t j = 0; j < src_tensor.dims()->size(); j++) {
        shape.push_back(src_tensor.dims()->data()[j]);
      }
    }
  }
  lite::Tensor *dst_tensor = nullptr;
  if (TypeId(src_tensor.dataType()) == kObjectTypeTensorType) {
    dst_tensor = new (std::nothrow) TensorList(shape, std::vector<int>(), src_category);
  } else {
    dst_tensor = new (std::nothrow) Tensor(TypeId(src_tensor.dataType()), shape, src_tensor.format(), src_category);
  }
  return dst_tensor;
}

int LiteSession::ConvertTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  copyed_tensor_idxes_.clear();
  uint32_t tensor_count = model->all_tensors_.size();
  MS_ASSERT(!model->sub_graphs_.empty());
  auto model_input_indices = model->sub_graphs_.front()->input_indices_;
  auto model_output_indices = model->sub_graphs_.front()->output_indices_;
  for (uint32_t i = 0; i < tensor_count; ++i) {
    auto *src_tensor = model->all_tensors_[i];
    if (src_tensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in model is nullptr";
      return RET_NULL_PTR;
    }
    auto *dst_tensor = ConvertTensor(*src_tensor);
    if (dst_tensor == nullptr) {
      MS_LOG(ERROR) << "Convert new " << i << "th tensor failed!";
      return RET_NULL_PTR;
    }
    auto ret = ConvertTensorsData(model, i, src_tensor, dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert data of " << i << "th tensor failed";
      delete (dst_tensor);
      return ret;
    }
    ConvertTensorsQuantParam(src_tensor, dst_tensor);
    if (IsContain(model_input_indices, i)) {
      dst_tensor->set_category(Tensor::GRAPH_INPUT);
    }
    if (IsContain(model_output_indices, i)) {
      dst_tensor->set_category(Tensor::GRAPH_OUTPUT);
    }
    if (src_tensor->name() != nullptr) {
      dst_tensor->set_tensor_name(src_tensor->name()->str());
    }
    this->tensors_.emplace_back(dst_tensor);
  }
  return RET_OK;
}

void LiteSession::InitGraphInputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(!(model->sub_graphs_.empty()));
  auto graph_in_size = model->sub_graphs_.front()->input_indices_.size();
  for (size_t i = 0; i < graph_in_size; ++i) {
    auto in_tensor_idx = model->sub_graphs_.front()->input_indices_[i];
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
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    auto out_tensor_idx = model->sub_graphs_.front()->output_indices_[i];
    MS_ASSERT(out_tensor_idx < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(out_tensor_idx);
    MS_ASSERT(out_tensor != nullptr);
    this->outputs_.emplace_back(out_tensor);
  }
}

void LiteSession::InitGraphInputMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(this->input_map_.empty());
  auto graph_input_node_indexes = GetGraphInputNodes(model);
  auto graph_in_size = model->sub_graphs_.front()->input_indices_.size();
  for (auto in_node_index : graph_input_node_indexes) {
    auto in_node = model->all_nodes_[in_node_index];
    MS_ASSERT(in_node != nullptr);
    auto in_size = in_node->input_indices_.size();
    for (size_t i = 0; i < in_size; ++i) {
      MS_ASSERT(this->input_map_.find(in_node->name_ + std::to_string(i)) == this->input_map_.end());
      auto in_tensor_index = size_t(in_node->input_indices_[i]);
      bool is_graph_input = false;
      for (size_t j = 0; j < graph_in_size; ++j) {
        if (in_tensor_index == model->sub_graphs_.front()->input_indices_[j]) {
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
      if (!in_tensor->tensor_name().empty()) {
        this->input_map_[in_tensor->tensor_name()] = in_tensor;
      }
    }
  }
}

void LiteSession::InitGraphOutputNodeMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  MS_ASSERT(!(model->sub_graphs_.empty()));
  auto graph_output_node_indexes = GetGraphOutputNodes(model);
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (auto out_node_index : graph_output_node_indexes) {
    auto out_node = model->all_nodes_[out_node_index];
    MS_ASSERT(out_node != nullptr);
    auto out_size = out_node->output_indices_.size();
    for (size_t i = 0; i < out_size; ++i) {
      auto out_tensor_index = out_node->output_indices_[i];
      bool is_graph_output = false;
      for (size_t j = 0; j < graph_out_size; ++j) {
        if (out_tensor_index == model->sub_graphs_.front()->output_indices_[j]) {
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
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->sub_graphs_.front()->output_indices_[i];
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

void LiteSession::AdjustModelOutputTensorInitRefCount(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto graph_out_size = model->sub_graphs_.front()->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->sub_graphs_.front()->output_indices_[i];
    MS_ASSERT(graph_out_index < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(graph_out_index);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "out_tensor is null!";
      return;
    }
    out_tensor->set_init_ref_count(out_tensor->init_ref_count() + 1);
  }
}

void LiteSession::InitGraphInOutTensors(const lite::Model *model) {
  InitGraphInputTensors(model);
  InitGraphInputMSTensors();
  InitGraphOutputTensors(model);
  InitGraphInputMap(model);
  InitGraphOutputNodeMap(model);
  InitGraphOutputTensorMap(model);
  for (auto *tensor : this->inputs_) {
    tensor->set_category(Tensor::Category::GRAPH_INPUT);
  }
}

int LiteSession::CompileGraph(Model *model) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  // model.MetaGraph ==> kernels
  if (model == nullptr) {
    MS_LOG(ERROR) << "The input model is nullptr.";
    is_running_.store(false);
    return RET_PARAM_INVALID;
  }
  if (model->buf == nullptr) {
    MS_LOG(ERROR) << "The input model buf is nullptr.";
    is_running_.store(false);
    return RET_PARAM_INVALID;
  }
  if (!reinterpret_cast<LiteModel *>(model)->ModelVerify()) {
    MS_LOG(ERROR) << "wrong model input, please check";
    is_running_.store(false);
    return RET_ERROR;
  }

  auto ret = ConvertTensors(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertTensors failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  // scheduler kernels
#if SUPPORT_NPU
  Scheduler scheduler(context_, model, &tensors_, npu_manager_, npu_pass_manager_);
#else
  Scheduler scheduler(context_, model, &tensors_);
#endif
  ret = scheduler.Schedule(&kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }
#if SUPPORT_NPU
  if (this->context_->IsNpuEnabled()) {
    MS_ASSERT(npu_manager_ != nullptr);
    if (npu_manager_->LoadOMModel() != RET_OK) {
      MS_LOG(ERROR) << "NPU client load model failed.";
      return RET_ERROR;
    }
  }
#endif
  InitGraphInOutTensors(model);

#ifdef ENABLE_MINDRT
  if (context_->IsCpuEnabled() && !context_->IsGpuEnabled() && !context_->IsNpuEnabled() && kernels_.size() == 1) {
    executor_ = new (std::nothrow) MindrtExecutor();
  } else {
    executor_ = new (std::nothrow) Executor();
  }
#else
  executor_ = new (std::nothrow) Executor();
#endif
  if (nullptr == executor_) {
    MS_LOG(ERROR) << "New Executor failed";
    is_running_.store(false);
    return RET_ERROR;
  }

  ret = executor_->Prepare(this->kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare executor failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  ret = PrepareKernels(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  is_running_.store(false);
  return RET_OK;
}

int LiteSession::PrepareKernels(Model *model) {
  std::vector<kernel::LiteKernel *> all_kernels;
  // find in_kernels and out_kernels for subgraphs
  for (auto kernel : this->kernels_) {
    kernel->FindInoutKernels(this->kernels_);
    auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    MS_ASSERT(sub_graph != nullptr);
    auto kernel_in_subgraph = sub_graph->nodes();
    all_kernels.insert(all_kernels.end(), kernel_in_subgraph.begin(), kernel_in_subgraph.end());
  }
  // find in_kernels and out_kernels for kernels
  for (auto *kernel : all_kernels) {
    kernel->FindInoutKernels(all_kernels);
  }
  // init init_ref_count for subgraphs and kernels
  for (auto *kernel : this->kernels_) {
    kernel->InitOutTensorInitRefCount();
  }
  AdjustModelOutputTensorInitRefCount(model);
  for (auto kernel : this->kernels_) {
    auto ret = kernel->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Prepare kernel " << kernel->name() << " failed: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetInputs() const { return this->input_vec_; }

int LiteSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  STATUS ret;
  MS_ASSERT(this->context_);
  if (before == nullptr && after == nullptr) {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get());
  } else {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get(), before, after);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunGraph failed : " << ret;
  }
  is_running_.store(false);
  return ret;
}

int LiteSession::Init(const Context *context) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
#if SUPPORT_NPU
  npu_manager_ = new (std::nothrow) NPUManager();
  if (npu_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu_manager_ failed";
    is_running_.store(false);
    return RET_ERROR;
  }
  npu_pass_manager_ = new (std::nothrow) NPUPassManager();
  if (npu_pass_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu_pass_manager_ failed";
    is_running_.store(false);
    return RET_ERROR;
  }
#endif
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    is_running_.store(false);
    return RET_NULL_PTR;
  }
#if SUPPORT_NPU
  this->context_ = new (std::nothrow) InnerContext(context, npu_manager_);
#else
  this->context_ = new (std::nothrow) InnerContext(context);
#endif
  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "New Context failed";
    is_running_.store(false);
    return RET_MEMORY_FAILED;
  }
  auto ret = this->context_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    is_running_.store(false);
    return ret;
  }
  ret = KernelRegistry::GetInstance()->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "KernelRegistry Init Failed.";
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
  if (this->context_->device_list_.empty()) {
    MS_LOG(ERROR) << "Device list is empty.";
    return;
  }
  if (!this->context_->IsCpuEnabled()) {
    return;
  }
  auto cpu_device_info = this->context_->GetCpuInfo();
  if (cpu_device_info.cpu_bind_mode_ != NO_BIND) {
    MS_ASSERT(this->context_->thread_pool_ != NULL);
    BindThreads(this->context_->thread_pool_, if_bind, cpu_device_info.cpu_bind_mode_);
  }
}

LiteSession::~LiteSession() {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return;
  }
  for (auto *kernel : kernels_) {
    delete kernel;
  }
  for (size_t i = 0; i < tensors_.size(); i++) {
    auto *tensor = tensors_.at(i);
    MS_ASSERT(tensor != nullptr);
    // data of weight tensor of node in packed_op can not be to free, we will free weight data when freeing meta_graph
    if (tensor->IsConst() && !IsContain(this->inputs_, tensor) && !IsContain(copyed_tensor_idxes_, i)) {
      tensor->set_data(nullptr);
    }
    delete tensor;
  }
  // Tensor * in input_map output_map are freed in tensors
  input_map_.clear();
  output_node_map_.clear();
  output_tensor_map_.clear();
  input_vec_.clear();
  delete this->context_;
  delete this->executor_;
  this->executor_ = nullptr;
#if SUPPORT_NPU
  MS_ASSERT(npu_manager_ != nullptr);
  MS_ASSERT(npu_pass_manager_ != nullptr);
  npu_pass_manager_->Clear();
  delete npu_pass_manager_;
  npu_manager_->Reset();
  delete npu_manager_;
#endif
#if GPU_OPENCL && !SUPPORT_TRAIN
  delete opencl_runtime_wrapper_;
#endif
  delete (model_);
  is_running_.store(false);
}

mindspore::tensor::MSTensor *LiteSession::GetInputsByTensorName(const std::string &name) const {
  auto ret = input_map_.find(name);
  if (ret == input_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << name << " is not exist";
    return nullptr;
  }
  return ret->second;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetOutputsByNodeName(const std::string &node_name) const {
  auto ret = output_node_map_.find(node_name);
  if (ret == output_node_map_.end()) {
    MS_LOG(WARNING) << "Node  " << node_name << " is not an output node";
    std::vector<mindspore::tensor::MSTensor *> empty_ret;
    return empty_ret;
  }
  return ret->second;
}

std::vector<std::string> LiteSession::GetOutputTensorNames() const { return this->output_tensor_names_; }

mindspore::tensor::MSTensor *LiteSession::GetOutputByTensorName(const std::string &tensor_name) const {
  auto ret = output_tensor_map_.find(tensor_name);
  if (ret == output_tensor_map_.end()) {
    MS_LOG(WARNING) << "Tensor  " << tensor_name << " is not an output node";
    return nullptr;
  }
  return ret->second;
}

std::unordered_map<std::string, mindspore::tensor::MSTensor *> LiteSession::GetOutputs() const {
  return this->output_tensor_map_;
}

int LiteSession::ResizeInputs(const std::vector<mindspore::tensor::MSTensor *> &inputs,
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
    inputs_[i]->set_shape(dims[i]);
  }
  return RET_OK;
}

void LiteSession::ResetInputsShape(const std::vector<std::vector<int>> &dims) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputs_[i]->FreeData();
    inputs_[i]->set_shape(dims[i]);
  }
}

int LiteSession::ReSizeKernels(const std::vector<kernel::LiteKernel *> &kernels) {
  bool infer_shape_interrupt = false;
  for (auto kernel : kernels) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      MS_LOG(ERROR) << "All node in graph should be sub_graph";
      return RET_ERROR;
    }
    auto ret = RET_OK;
    if (kernel->subgraph_type() == kernel::kGpuSubGraph) {
#if GPU_OPENCL
      auto sub_graph = reinterpret_cast<kernel::OpenCLSubGraph *>(kernel);
      ret = sub_graph->ReSize(false);
#endif
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      ret = sub_graph->ReSize(infer_shape_interrupt);
    }
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape is interrupted";
      infer_shape_interrupt = true;
      continue;
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReSize node " << kernel->name() << " failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int LiteSession::Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs,
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

  ret = ReSizeKernels(kernels_);
  if (ret != RET_OK) {
    ResetInputsShape(old_dims);
    auto resize_ret = ReSizeKernels(kernels_);
    if (resize_ret != RET_OK) {
      MS_LOG(ERROR) << "restore kernel size fail!ret: " << resize_ret;
    }
    is_running_.store(false);
    return ret;
  }
  is_running_.store(false);
  return RET_OK;
}

int LiteSession::InitGPURuntime() {
#if GPU_OPENCL && !SUPPORT_TRAIN
  if (this->context_->IsGpuEnabled()) {
    opencl_runtime_wrapper_ = new (std::nothrow) opencl::OpenCLRuntimeWrapper();
    if (opencl_runtime_wrapper_ == nullptr) {
      MS_LOG(ERROR) << "create OpenCLRuntimeWrapper failed";
      return RET_ERROR;
    }
    auto gpu_device_info = this->context_->GetGpuInfo();
    auto opencl_runtime = opencl_runtime_wrapper_->GetInstance();
    opencl_runtime->SetFp16Enable(gpu_device_info.enable_float16_);
    if (opencl_runtime->Init() != RET_OK) {
      this->context_->device_list_ = {{DT_CPU, {gpu_device_info.enable_float16_, MID_CPU}}};
      MS_LOG(WARNING) << "Init OpenCL runtime failed, change to CPU mode.";
    } else {
      MS_LOG(INFO) << "Init OpenCL runtime success.";
    }
  }
#elif GPU_VULKAN && !SUPPORT_TRAIN
  if (this->context_->IsGpuEnabled()) {
    auto gpu_device_info = this->context_->GetGpuInfo();
    vk_runtime_wrap_ = new (std::nothrow) gpu::GpuRuntimeWrapper<vulkan::VulkanRuntime>;
    if (vk_runtime_wrap_ == nullptr) {
      MS_LOG(ERROR) << "create vk_runtime failed";
      return RET_ERROR;
    }
    auto vk_runtime = vk_runtime_wrap_->GetInstance();
    vk_runtime->SetFp16Enable(gpu_device_info.enable_float16_);
    if (vk_runtime->Init() != RET_OK) {
      this->context_->device_list_ = {{DT_CPU, {gpu_device_info.enable_float16_, MID_CPU}}};
      MS_LOG(WARNING) << "Init Vulkan runtime failed, change to CPU mode.";
    } else {
      MS_LOG(INFO) << "Init Vulkan runtime success.";
    }
  }
#endif
  return RET_OK;
}
}  // namespace lite

session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {
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

session::LiteSession *session::LiteSession::CreateSession(const char *model_buf, size_t size,
                                                          const lite::Context *context) {
  auto *session = LiteSession::CreateSession(context);
  if (session == nullptr) {
    MS_LOG(ERROR) << "Create session failed";
    return nullptr;
  }
  auto *model = lite::ImportFromBuffer(model_buf, size, true);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return nullptr;
  }
  auto ret = session->CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    return nullptr;
  }
  model->buf = nullptr;
  (reinterpret_cast<lite::LiteSession *>(session))->set_model(model);
  return session;
}
}  // namespace mindspore

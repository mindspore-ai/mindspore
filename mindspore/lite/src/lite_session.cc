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

#include "src/lite_session.h"
#include <vector>
#include <utility>
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/scheduler.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/allocator.h"
#include "src/executor.h"
#include "src/common/utils.h"
#include "src/common/graph_util.h"
#include "src/kernel_registry.h"
#if SUPPORT_GPU
#include "src/runtime/opencl/opencl_runtime.h"
#endif

namespace mindspore {
namespace lite {
int LiteSession::ConvertTensors(const lite::Model *model) {
  MS_EXCEPTION_IF_NULL(model);
  auto meta_graph = model->GetMetaGraph();
  MS_EXCEPTION_IF_NULL(meta_graph);
  uint32_t tensorCount = meta_graph->allTensors()->size();
  for (uint32_t i = 0; i < tensorCount; i++) {
    auto *srcTensor = meta_graph->allTensors()->GetAs<schema::Tensor>(i);
    if (srcTensor == nullptr) {
      MS_LOG(ERROR) << i << "th tensor in meta_graph is nullptr";
      return RET_NULL_PTR;
    }
    std::vector<int> shape;
    if (srcTensor->dims() == nullptr) {
      MS_LOG(DEBUG) << "Dims of " << i << "th tensor is nullptr";
    } else {
      if (srcTensor->nodeType() == schema::NodeType_ValueNode) {
        for (size_t j = 0; j < srcTensor->dims()->size(); j++) {
          shape.push_back(srcTensor->dims()->data()[j]);
        }
      }
    }
    int dataType = srcTensor->dataType();
    auto *dstTensor = new tensor::Tensor(TypeId(dataType), shape, srcTensor->format(), srcTensor->nodeType());
    if (srcTensor->nodeType() == schema::NodeType_ValueNode && srcTensor->data() != nullptr &&
        srcTensor->data()->size() > 0) {
      if (shape.empty()) {
        shape.push_back(1);
      }
      MS_ASSERT(dstTensor != nullptr);
      MS_ASSERT(dstTensor->Size() == srcTensor->data()->size());
      // no copy data, do copy when call LiteKernel::Init
      dstTensor->SetData(const_cast<unsigned char *>(srcTensor->data()->data()));
    }
    auto quant_params = srcTensor->quantParams();
    if (quant_params != nullptr) {
      for (size_t j = 0; j < quant_params->size(); j++) {
        tensor::QuantArg quant_arg{};
        quant_arg.scale = quant_params->Get(j)->scale();
        quant_arg.zeroPoint = quant_params->Get(j)->zeroPoint();
        dstTensor->AddQuantParam(quant_arg);
      }
    }

    this->tensors_.emplace_back(dstTensor);
  }

  return RET_OK;
}

void LiteSession::InitGraphInputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->inputs_.empty());
  MS_ASSERT(meta_graph != nullptr);
  for (size_t i = 0; i < meta_graph->inputIndex()->size(); i++) {
    auto in_tensor_idx = size_t(meta_graph->inputIndex()->GetAs<uint32_t>(i));
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
    this->input_vec_.emplace_back(new lite::tensor::LiteTensor(input_tensor));
  }
}

void LiteSession::InitGraphOutputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->outputs_.empty());
  MS_ASSERT(meta_graph != nullptr);
  for (size_t i = 0; i < meta_graph->outputIndex()->size(); i++) {
    auto out_tensor_idx = size_t(meta_graph->outputIndex()->GetAs<uint32_t>(i));
    MS_ASSERT(out_tensor_idx < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(out_tensor_idx);
    MS_ASSERT(out_tensor != nullptr);
    this->outputs_.emplace_back(out_tensor);
  }
}

void LiteSession::InitGraphInputMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->input_map_.empty());
  MS_ASSERT(meta_graph != nullptr);
  auto graph_input_node_indexes = GetGraphInputNodes(meta_graph);
  for (auto in_node_index : graph_input_node_indexes) {
    auto *in_node = meta_graph->nodes()->GetAs<schema::CNode>(in_node_index);
    MS_ASSERT(nullptr != in_node);
    MS_ASSERT(this->input_map_.find(in_node->name()->str()) == this->input_map_.end());
    for (size_t i = 0; i < in_node->inputIndex()->size(); i++) {
      auto in_tensor_index = size_t(in_node->inputIndex()->GetAs<uint32_t>(i));
      bool is_graph_input = false;
      for (size_t j = 0; j < meta_graph->inputIndex()->size(); j++) {
        if (in_tensor_index == size_t(meta_graph->inputIndex()->GetAs<uint32_t>(j))) {
          is_graph_input = true;
          break;
        }
      }
      if (!is_graph_input) {
        continue;
      }
      MS_ASSERT(in_tensor_index < this->tensors_.size());
      auto *in_tensor = this->tensors_.at(in_tensor_index);
      MS_ASSERT(in_tensor != nullptr);
      auto *ms_tensor = new tensor::LiteTensor(in_tensor);
      MS_ASSERT(nullptr != ms_tensor);
      this->input_map_[in_node->name()->str()].emplace_back(ms_tensor);
    }
  }
}

void LiteSession::InitGraphOutputNodeMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->output_node_map_.empty());
  MS_ASSERT(meta_graph != nullptr);
  auto graph_output_node_indexes = GetGraphOutputNodes(meta_graph);
  for (auto out_node_index : graph_output_node_indexes) {
    auto *out_node = meta_graph->nodes()->GetAs<schema::CNode>(out_node_index);
    MS_ASSERT(nullptr != out_node);
    MS_ASSERT(this->output_map_.find(out_node->name()->str()) == this->output_map_.end());
    for (size_t i = 0; i < out_node->outputIndex()->size(); i++) {
      auto out_tensor_index = size_t(out_node->outputIndex()->GetAs<uint32_t>(i));
      bool is_graph_output = false;
      for (size_t j = 0; j < meta_graph->outputIndex()->size(); j++) {
        if (out_tensor_index == size_t(meta_graph->outputIndex()->GetAs<uint32_t>(j))) {
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        continue;
      }
      MS_ASSERT(out_tensor_index < this->tensors_.size());
      auto *out_tensor = this->tensors_.at(out_tensor_index);
      MS_ASSERT(out_tensor != nullptr);
      auto *ms_tensor = new tensor::LiteTensor(out_tensor);
      MS_ASSERT(nullptr != ms_tensor);
      this->output_node_map_[out_node->name()->str()].emplace_back(ms_tensor);
    }
  }
}

void LiteSession::InitGraphOutputTensorNames(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->output_tensor_names_.empty());
  MS_ASSERT(meta_graph != nullptr);
  for (auto output_index : *meta_graph->outputIndex()) {
    this->output_tensor_names_.emplace_back(std::to_string(output_index));
  }
}

void LiteSession::InitGraphOutputTensorMap(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto meta_graph = model->GetMetaGraph();
  MS_ASSERT(this->output_tensor_map_.empty());
  MS_ASSERT(meta_graph != nullptr);
  for (auto graph_out_index : *(meta_graph->outputIndex())) {
    MS_ASSERT(graph_out_index < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(graph_out_index);
    MS_ASSERT(out_tensor != nullptr);
    auto *ms_tensor = new tensor::LiteTensor(out_tensor);
    MS_ASSERT(nullptr != ms_tensor);
    this->output_tensor_map_.insert(std::make_pair(std::to_string(graph_out_index), ms_tensor));
  }
}

void LiteSession::InitGraphInOutTensors(const lite::Model *model) {
  InitGraphInputTensors(model);
  InitGraphInputMSTensors();
  InitGraphOutputTensors(model);
  InitGraphInputMap(model);
  InitGraphOutputNodeMap(model);
  InitGraphOutputTensorNames(model);
  InitGraphOutputTensorMap(model);
}

int LiteSession::CompileGraph(Model *model) {
  // model.MetaGraph ==> kernels
  if (model == nullptr) {
    MS_LOG(ERROR) << "The input model is nullptr.";
    return RET_PARAM_INVALID;
  }

  auto ret = ConvertTensors(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvertTensors failed: " << ret;
    return ret;
  }

  InitGraphInOutTensors(model);

  // scheduler kernels
  Scheduler scheduler(context_);
  ret = scheduler.Schedule(model, &tensors_, &kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule kernels failed: " << ret;
    return ret;
  }

  executor->Prepare(this->kernels_);
  return RET_OK;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetInputs() const { return this->input_vec_; }

int LiteSession::RunGraph(const session::KernelCallBack &before, const session::KernelCallBack &after) {
  MS_EXCEPTION_IF_NULL(this->context_);
  SetMaxWokerNum(context_->thread_num_);
  if (before == nullptr && after == nullptr) {
    return executor->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get());
  } else {
    return executor->Run(this->inputs_, this->outputs_, this->kernels_, this->context_->allocator.get(), before, after);
  }
}

int LiteSession::Init(Context *context) {
  MS_EXCEPTION_IF_NULL(context);
  this->context_ = new (std::nothrow) Context(context->thread_num_, context->allocator, context->device_ctx_);
  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "new context failed";
    return RET_MEMORY_FAILED;
  }
  this->context_->float16_priority = context->float16_priority;
  this->context_->cpu_bind_mode_ = context->cpu_bind_mode_;
  ConfigThreadPool(context->cpu_bind_mode_, context->thread_num_);
  auto ret = KernelRegistry::GetInstance()->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "KernelRegistry Init Failed.";
    return ret;
  }
#if SUPPORT_GPU
  if (context_->device_ctx_.type == DT_GPU) {
    auto opencl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    opencl_runtime->Init();
  }
#endif
  executor = new Executor();
  MS_EXCEPTION_IF_NULL(executor);
  return RET_OK;
}

void LiteSession::BindThread(bool if_bind) {
  if (this->context_->cpu_bind_mode_ != NO_BIND) {
    DoAllThreadBind(if_bind, static_cast<int>(this->context_->cpu_bind_mode_));
  }
}

LiteSession::~LiteSession() {
  for (auto *tensor : tensors_) {
    // weight data can not be to free, we will free weight data when freeing meta_graph
    if (tensor->TensorType() == schema::NodeType_ValueNode && !IsContain(this->inputs_, tensor)) {
      tensor->SetData(nullptr);
    }
    delete tensor;
  }
  // tensor::Tensor * in input_map output_map are freed in tensors
  for (auto iter : this->input_map_) {
    for (auto *ms_tensor : iter.second) {
      ((tensor::LiteTensor *)ms_tensor)->SetTensorImpl(nullptr);
      delete ms_tensor;
    }
    iter.second.clear();
  }
  input_map_.clear();
  for (auto iter : this->output_node_map_) {
    for (auto *ms_tensor : iter.second) {
      ((tensor::LiteTensor *)ms_tensor)->SetTensorImpl(nullptr);
      delete ms_tensor;
    }
    iter.second.clear();
  }
  output_node_map_.clear();
  for (auto iter : this->output_tensor_map_) {
    ((tensor::LiteTensor *)(iter.second))->SetTensorImpl(nullptr);
    delete (iter.second);
  }
  output_tensor_map_.clear();
  for (auto *kernel : kernels_) {
    delete kernel;
  }
  for (auto *ms_tensor : input_vec_) {
    if (ms_tensor != nullptr) {
      ((tensor::LiteTensor *)ms_tensor)->SetTensorImpl(nullptr);
      delete ms_tensor;
    }
  }
  input_vec_.clear();
  delete this->context_;
  delete this->executor;
  this->executor = nullptr;
}

std::vector<mindspore::tensor::MSTensor *> LiteSession::GetInputsByName(const std::string &name) const {
  auto ret = input_map_.find(name);
  if (ret == input_map_.end()) {
    MS_LOG(WARNING) << "Node  " << name << " is not an input node";
    std::vector<mindspore::tensor::MSTensor *> empty_ret;
    return empty_ret;
  }
  return ret->second;
}

std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> LiteSession::GetOutputMapByNode() const {
  return this->output_node_map_;
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

std::unordered_map<std::string, mindspore::tensor::MSTensor *> LiteSession::GetOutputMapByTensor() const {
  return this->output_tensor_map_;
}

int LiteSession::ResizeInputs(const std::vector<mindspore::tensor::MSTensor *> &inputs) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(ERROR) << "Inputs size " << inputs.size() << " is not equal to " << inputs_.size();
    return RET_PARAM_INVALID;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr) {
      MS_LOG(ERROR) << "Input tensor is nullptr!";
      return RET_PARAM_INVALID;
    }
    inputs_[i]->set_shape(inputs[i]->shape());
  }
  return RET_OK;
}

int LiteSession::Resize(const std::vector<mindspore::tensor::MSTensor *> &inputs) {
  std::vector<tensor::Tensor *> inputs_old(inputs_);
  auto ret = ResizeInputs(inputs);
  if (ret != RET_OK) {
    inputs_ = inputs_old;
    return ret;
  }

  Scheduler scheduler(context_);
  ret = scheduler.ReSizeKernels(kernels_);
  if (ret != RET_OK) {
    inputs_ = inputs_old;
    auto resize_ret = scheduler.ReSizeKernels(kernels_);
    if (resize_ret != RET_OK) {
      MS_LOG(ERROR) << "restore kernel size fail!ret: " << resize_ret;
    }
    return ret;
  }
  return RET_OK;
}
}  // namespace lite

session::LiteSession *session::LiteSession::CreateSession(lite::Context *context) {
  auto session = new lite::LiteSession();
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init sesssion failed";
    delete session;
    return nullptr;
  }
  return session;
}
}  // namespace mindspore

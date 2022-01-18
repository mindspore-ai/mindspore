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
#ifdef ENABLE_V0
#include <set>
#endif
#include <utility>
#ifndef _WIN32
#include <malloc.h>
#endif
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/scheduler.h"
#include "src/runtime/inner_allocator.h"
#include "src/executor.h"
#include "src/common/context_util.h"
#include "src/common/utils.h"
#include "src/common/prim_util.h"
#include "src/common/graph_util.h"
#include "src/common/tensor_util.h"
#include "src/common/file_utils.h"
#include "src/kernel_registry.h"
#include "src/lite_model.h"
#include "src/weight_decoder.h"
#include "src/lite_kernel_util.h"
#ifdef ENABLE_MINDRT
#include "src/mindrt_executor.h"
#endif
#if SUPPORT_NPU
#include "src/delegate/npu/npu_delegate.h"
#endif
#if GPU_OPENCL
#include "src/runtime/kernel/opencl/opencl_subgraph.h"
#endif
#if GPU_TENSORRT
#include "src/delegate/tensorrt/tensorrt_delegate.h"
#endif
#ifndef WEIGHT_DECODE_CLIP
#include "tools/converter/quantizer/fse_decoder.h"
#endif
namespace mindspore {
namespace lite {
namespace {
bool NeedBitUppackCheck(const schema::Tensor &src_tensor) {
  if (src_tensor.enableHuffmanCode()) {
    return true;
  }
  bool need_bit_unpack = src_tensor.quantParams() != nullptr && src_tensor.quantParams()->size() > 0 &&
                         src_tensor.quantParams()->Get(0) != nullptr;
  if (need_bit_unpack) {
    auto num_bits = src_tensor.quantParams()->Get(0)->numBits();
    need_bit_unpack = ((num_bits >= kBitNum1 && num_bits < kBitNum8) || (num_bits > kBitNum8 && num_bits < kBitNum16));
  }

  return need_bit_unpack;
}

int DecompressTensor(const schema::Tensor &src_tensor, Tensor *dst_tensor) {
  MS_ASSERT(dst_tensor != nullptr);
#ifndef WEIGHT_DECODE_CLIP
  if (src_tensor.weightQunatCompressType() == schema::WeightQunatCompressType_FSE) {
    return quant::FSEDecoder::DeCompress(src_tensor, dst_tensor);
  } else if (src_tensor.weightQunatCompressType() == schema::WeightQunatCompressType_INDEXING) {
    return IndexingDecompress(src_tensor, dst_tensor);
  } else if (src_tensor.weightQunatCompressType() == schema::WeightQunatCompressType_SPARSE) {
    return SparseDecompress(src_tensor, dst_tensor);
  }
#else
  if (src_tensor.weightQunatCompressType() != schema::WeightQunatCompressType_NONE) {
    MS_LOG(ERROR) << unsupport_weight_decode_log;
    return RET_ERROR;
  }
#endif
  if (!NeedBitUppackCheck(src_tensor)) {
    return RET_NO_CHANGE;
  } else {
#ifndef WEIGHT_DECODE_CLIP
    return WeightDecoder::UnPack(src_tensor, dst_tensor);
#else
    MS_LOG(ERROR) << unsupport_weight_decode_log;
    return RET_ERROR;
#endif
  }
}
}  // namespace

LiteSession::LiteSession() { this->is_running_.store(false); }

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

int LiteSession::ConvertTensorsData(size_t tensor_index, const schema::Tensor *src_tensor, lite::Tensor *dst_tensor) {
  MS_ASSERT(src_tensor != nullptr);
  MS_ASSERT(dst_tensor != nullptr);
  if (src_tensor->data() == nullptr || src_tensor->data()->size() <= 0) {
    MS_LOG(DEBUG) << "No valid data converted.";
    return RET_OK;
  }

  /* tensor list convert */
  if (dst_tensor->data_type() == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    auto tensor_list = reinterpret_cast<TensorList *>(dst_tensor);
    if (tensor_list->Decode(reinterpret_cast<const int *>(src_tensor->data()->data())) != RET_OK) {
      MS_LOG(ERROR) << "Decode tensorlist data failed";
      return RET_ERROR;
    }
    return RET_OK;
#else
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
    return RET_NOT_SUPPORT;
#endif
  }

  /* normal tensor check */
  auto shape_info = dst_tensor->shape();
  if (shape_info.end() !=
      std::find_if(shape_info.begin(), shape_info.end(), [](const int shape) { return shape <= 0; })) {
    MS_LOG(ERROR) << "Invalid shape size." << src_tensor->name()->c_str();
    return RET_ERROR;
  }

  auto ret = DecompressTensor(*src_tensor, dst_tensor);
  if (ret == RET_NO_CHANGE) {
    if (dst_tensor->Size() == 0 || src_tensor->data()->size() < dst_tensor->Size()) {
      MS_LOG(ERROR) << "Tensor data shape invalid";
      return RET_ERROR;
    }
    dst_tensor->set_data(const_cast<unsigned char *>(src_tensor->data()->data()));
    dst_tensor->set_own_data(false);
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
  auto src_category = TensorCategory(&src_tensor);
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
  }
  lite::Tensor *dst_tensor = nullptr;
  if (TypeId(data_type) == kObjectTypeTensorType) {
#ifndef CONTROLFLOW_TENSORLIST_CLIP
    dst_tensor = new (std::nothrow) TensorList(shape, std::vector<int>(), src_category);
    // set tensor list datatype
    auto tensor_list = reinterpret_cast<TensorList *>(dst_tensor);
    if (src_tensor.data() != nullptr) {
      auto tensor_data_type = TypeId(reinterpret_cast<const int *>(src_tensor.data()->data())[0]);
      tensor_list->set_tensors_data_type(tensor_data_type);
    }
#else
    MS_LOG(ERROR) << unsupport_controlflow_tensorlist_log;
#endif
  } else {
    dst_tensor = new (std::nothrow)
      Tensor(TypeId(data_type), shape, static_cast<mindspore::Format>(src_tensor.format()), src_category);
  }
  return dst_tensor;
}

#ifdef ENABLE_V0
void LiteSession::TensorNameCompatibleWithV0(const lite::Model *model) {
  if (reinterpret_cast<const LiteModel *>(model)->GetSchemaVersion() != SCHEMA_VERSION::SCHEMA_V0) {
    return;
  }
  auto model_output_indices = model->output_indices_;
  std::set<uint32_t> to_name_tensor_indices;
  for (uint32_t &model_output_index : model_output_indices) {
    auto *dst_tensor = this->tensors_[model_output_index];
    MS_ASSERT(dst_tensor != nullptr);
    if (!dst_tensor->tensor_name().empty()) {
      continue;
    }
    to_name_tensor_indices.insert(model_output_index);
  }
  if (model_output_indices.empty()) {
    return;
  }
  for (int i = static_cast<int>(model->all_nodes_.size()) - 1; i >= 0; i--) {
    const auto &node = model->all_nodes_.at(i);
    MS_ASSERT(node != nullptr);
    for (size_t j = 0; j < node->output_indices_.size(); j++) {
      const auto &output_index = node->output_indices_.at(j);
      if (to_name_tensor_indices.count(output_index) > 0) {
        auto *dst_tensor = this->tensors_[output_index];
        MS_ASSERT(dst_tensor != nullptr);
        if (node->output_indices_.size() == 1) {
          dst_tensor->set_tensor_name(node->name_);
        } else {
          dst_tensor->set_tensor_name(node->name_ + "_o:" + std::to_string(j));
        }
        to_name_tensor_indices.erase(output_index);
      }
    }
    if (to_name_tensor_indices.empty()) {
      break;
    }
  }
}
#endif

int LiteSession::ConvertTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  uint32_t tensor_count = model->all_tensors_.size();
  auto model_input_indices = model->input_indices_;
  auto model_output_indices = model->output_indices_;
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
    auto ret = ConvertTensorsData(i, src_tensor, dst_tensor);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Convert data of " << i << "th tensor failed";
      delete dst_tensor;
      return ret;
    }
    ConvertTensorsQuantParam(src_tensor, dst_tensor);
    if (IsContain(model_input_indices, i)) {
      if (dst_tensor->data() != nullptr) {
        MS_LOG(ERROR) << "Graph input shouldn't have data";
        delete dst_tensor;
        return RET_ERROR;
      }
      dst_tensor->set_category(Tensor::GRAPH_INPUT);
    }
    if (IsContain(model_output_indices, i)) {
      if (dst_tensor->data() != nullptr) {
        MS_LOG(ERROR) << "Graph output shouldn't have data";
        delete dst_tensor;
        return RET_ERROR;
      }
      // a tensor is as both input and output, would be treated as an input.
      if (!dst_tensor->IsGraphInput()) {
        dst_tensor->set_category(Tensor::GRAPH_OUTPUT);
      }
    }
    if (src_tensor->name() != nullptr) {
      dst_tensor->set_tensor_name(src_tensor->name()->str());
    }
    this->tensors_.emplace_back(dst_tensor);
  }
#ifdef ENABLE_V0
  TensorNameCompatibleWithV0(model);
#endif
  return RET_OK;
}

void LiteSession::InitGraphInputTensors(const lite::Model *model) {
  MS_ASSERT(model != nullptr);
  auto graph_in_size = model->input_indices_.size();
  for (size_t i = 0; i < graph_in_size; ++i) {
    auto in_tensor_idx = model->input_indices_[i];
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
  auto graph_out_size = model->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    auto out_tensor_idx = model->output_indices_[i];
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
  auto graph_in_size = model->input_indices_.size();
  for (auto in_node_index : graph_input_node_indexes) {
    auto in_node = model->all_nodes_[in_node_index];
    MS_ASSERT(in_node != nullptr);
    auto in_size = in_node->input_indices_.size();
    for (size_t i = 0; i < in_size; ++i) {
      MS_ASSERT(this->input_map_.find(in_node->name_ + std::to_string(i)) == this->input_map_.end());
      auto in_tensor_index = size_t(in_node->input_indices_[i]);
      bool is_graph_input = false;
      for (size_t j = 0; j < graph_in_size; ++j) {
        if (in_tensor_index == model->input_indices_[j]) {
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
  auto graph_output_node_indexes = GetGraphOutputNodes(model);
  auto graph_out_size = model->output_indices_.size();
  for (auto out_node_index : graph_output_node_indexes) {
    auto out_node = model->all_nodes_[out_node_index];
    MS_ASSERT(out_node != nullptr);
    auto out_size = out_node->output_indices_.size();
    for (size_t i = 0; i < out_size; ++i) {
      auto out_tensor_index = out_node->output_indices_[i];
      bool is_graph_output = false;
      for (size_t j = 0; j < graph_out_size; ++j) {
        if (out_tensor_index == model->output_indices_[j]) {
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
  auto graph_out_size = model->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->output_indices_[i];
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
  auto graph_out_size = model->output_indices_.size();
  for (size_t i = 0; i < graph_out_size; ++i) {
    size_t graph_out_index = model->output_indices_[i];
    MS_ASSERT(graph_out_index < this->tensors_.size());
    auto *out_tensor = this->tensors_.at(graph_out_index);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "out_tensor is null!";
      return;
    }
    out_tensor->set_init_ref_count(out_tensor->init_ref_count() + 1);
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
    Tensor *new_tensor =
      new Tensor(src_tensor->data_type(), src_tensor->shape(), src_tensor->format(), Tensor::GRAPH_OUTPUT);
    if (new_tensor == nullptr) {
      MS_LOG(ERROR) << "duplicate new outptu failed.";
      return RET_NULL_PTR;
    }
    new_tensor->set_allocator(src_tensor->allocator()); /* GPU use opencl allocator */
    new_tensor->set_tensor_name(src_tensor->tensor_name() + "_duplicate");
    for (LiteQuantParam quant : src_tensor->quant_params()) {
      new_tensor->AddQuantParam(quant);
    }
    new_tensor->set_init_ref_count(src_tensor->init_ref_count());

    /* src tensor set for graph calculate */
    if (src_tensor->data_type() == kNumberTypeFloat16) {
      src_tensor->set_data_type(kNumberTypeFloat32);
    }
    src_tensor->set_ref_count(1);

    graph_output_map_.insert(std::make_pair(new_tensor, src_tensor));

    /* set new tensor for calculate */
    for (auto subgraph : kernels_) {
      /* subgraph input and output */
      for (size_t i = 0; i < subgraph->in_tensors().size(); i++) {
        if (subgraph->in_tensors()[i] == src_tensor) {
          subgraph->set_in_tensor(new_tensor, i);
        }
      }
      for (size_t i = 0; i < subgraph->out_tensors().size(); i++) {
        if (subgraph->out_tensors()[i] == src_tensor) {
          subgraph->set_out_tensor(new_tensor, i);
        }
      }
#ifndef DELEGATE_CLIP
      if (subgraph->desc().arch == kernel::kDelegate) {
        continue;
      }
#endif
      /* node input and output */
      auto nodes = reinterpret_cast<kernel::SubGraphKernel *>(subgraph)->nodes();
      for (size_t i = 0; i < nodes.size(); i++) {
        auto node = nodes[i];
        for (size_t j = 0; j < node->out_tensors().size(); j++) {
          if (node->out_tensors()[j] == src_tensor) {
            node->set_out_tensor(new_tensor, j);
          }
        }
        for (size_t j = 0; j < node->in_tensors().size(); j++) {
          if (node->in_tensors()[j] == src_tensor) {
            node->set_in_tensor(new_tensor, j);
          }
        }
      }
    }
  }
  return RET_OK;
}

void LiteSession::FreePackOpWeight(const std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel != nullptr);
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (!IsPackedOp(kernel->type())) {
        continue;
      }
    } else {
      auto subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      FreePackOpWeight(subgraph->nodes());
    }
    auto inputs = kernel->in_tensors();
    for (auto *tensor : inputs) {
      MS_ASSERT(tensor != nullptr);
      if (!tensor->IsConst()) {
        continue;
      }
      tensor->FreeData();
    }
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
  InitGraphInputTensors(model);
  InitGraphOutputTensors(model);
#ifndef ENABLE_FP16
  if (context_->GetCpuInfo().enable_float16_) {
    MS_LOG(WARNING) << unsupport_fp16_log;
  }
#endif
  // scheduler kernels
  Scheduler scheduler(context_, ms_context_, model, &tensors_, inputs_, outputs_, is_train_session_, execution_plan_,
                      delegate_, delegate_device_type_);
  scheduler.SetupSchedulerCb(std::move(sched_cb_));
  ret = scheduler.Schedule(&kernels_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Schedule kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }
  InitGraphInOutTensorsMap(model);

  ret = PrepareKernels(model);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare kernels failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  if (is_train_session_) {
    is_running_.store(false);
    return RET_OK;
  }

#ifdef ENABLE_MINDRT
  ret = IsolateOutputTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Isolate output tensor failed.";
    is_running_.store(false);
    return ret;
  }
  executor_ = new (std::nothrow) MindrtExecutor(&graph_output_map_);
#else
  executor_ = new (std::nothrow) Executor();
#endif
  if (executor_ == nullptr) {
    MS_LOG(ERROR) << "New Executor failed";
    is_running_.store(false);
    return RET_ERROR;
  }

  ret = executor_->Prepare(this->kernels_, this->inputs_, this->outputs_, context_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare executor failed: " << ret;
    is_running_.store(false);
    return ret;
  }

  // For reducing runtime RAM, free packop weight because packop will pack weight and will not access to origin weight
  FreePackOpWeight(kernels_);
  is_running_.store(false);
#if !defined(_WIN32) && !defined(ENABLE_ARM)
  (void)malloc_trim(0);  // release memory manually.
#endif
  return RET_OK;
}

bool LiteSession::IsIsolatedSubGraph(kernel::LiteKernel *kernel) {
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

int LiteSession::SetAllocatorForDelegateKernels(const kernel::LiteKernel *kernel) {
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

int LiteSession::PrepareKernels(Model *model) {
  std::vector<kernel::LiteKernel *> all_kernels;
  for (auto kernel : this->kernels_) {
#ifndef DELEGATE_CLIP
    if (kernel->desc().arch == kernel::kDelegate) {
      all_kernels.push_back(kernel);
      continue;
    }
#endif
    auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
    MS_ASSERT(sub_graph != nullptr);
    auto kernel_in_subgraph = sub_graph->nodes();
    all_kernels.insert(all_kernels.end(), kernel_in_subgraph.begin(), kernel_in_subgraph.end());
  }

  // find in_kernels and out_kernels for kernels
  kernel::LiteKernelUtil::FindAllInoutKernels(all_kernels);

  // find in_sub and out_sub for subgraph
  kernel::LiteKernelUtil::FindAllInoutKernels(this->kernels_);

  // init init_ref_count for subgraphs and kernels
  for (auto *kernel : this->kernels_) {
    kernel->InitOutTensorInitRefCount();
#ifndef DELEGATE_CLIP
    if (kernel->desc().arch == kernel::kDelegate) {
      continue;
    }
#endif
    if (IsIsolatedSubGraph(kernel)) {
      static_cast<kernel::SubGraphKernel *>(kernel)->InitInputTensorInitRefCount();
    }
  }
  AdjustModelOutputTensorInitRefCount(model);
  for (auto kernel : this->kernels_) {
    if (kernel->desc().arch == kernel::kDelegate) {
      auto ret = SetAllocatorForDelegateKernels(kernel);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Prepare kernel " << kernel->name() << " failed: " << ret;
        return ret;
      }
    }
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
  STATUS ret = CheckTensorsInvalid(inputs_);
  if (ret != RET_OK) {
    is_running_.store(false);
    MS_LOG(ERROR) << "CheckInputs failed.";
    return ret;
  }
  MS_ASSERT(this->context_ != nullptr);
  if (before == nullptr && after == nullptr) {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_);
  } else {
    ret = executor_->Run(this->inputs_, this->outputs_, this->kernels_, before, after);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunGraph failed : " << ret;
  }
  is_running_.store(false);
  return ret;
}

int LiteSession::Init(InnerContext *context) {
  bool expected = false;
  if (!is_running_.compare_exchange_strong(expected, true)) {
    MS_LOG(ERROR) << "Not support multi-threading";
    return RET_ERROR;
  }
  if (context == nullptr) {
    MS_LOG(ERROR) << "context is nullptr";
    is_running_.store(false);
    return RET_NULL_PTR;
  }
  this->context_ = context;

  auto ret = this->context_->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init Context failed";
    is_running_.store(false);
    return ret;
  }

#ifdef MS_COMPILE_IOS
  context_->thread_pool()->SetMaxSpinCount(kDefaulLiteIosSpinCount);
  context_->thread_pool()->SetMinSpinCount(kDefaulLiteIosSpinCount);
#endif

  if (context->delegate != nullptr) {
#ifndef DELEGATE_CLIP
    delegate_ = context->delegate;
    delegate_device_type_ = -1;
#else
    MS_LOG(ERROR) << unsupport_delegate_log;
    is_running_.store(false);
    return RET_NOT_SUPPORT;
#endif
  }
  ms_context_ = MSContextFromContext(context);
  if (ms_context_ == nullptr) {
    MS_LOG(ERROR) << "transfer context to ms context failed.";
    is_running_.store(false);
    return RET_NULL_PTR;
  }
#ifndef DELEGATE_CLIP
#if SUPPORT_NPU
  if (delegate_ == nullptr && context_->IsNpuEnabled()) {
    delegate_ = std::make_shared<NPUDelegate>(context_->GetNpuInfo());
    if (delegate_ == nullptr) {
      MS_LOG(ERROR) << "New delegate_ failed";
      return RET_ERROR;
    }
    delegate_device_type_ = DT_NPU;
    this->context_->delegate = delegate_;
  }
#endif
#if GPU_TENSORRT
  if (delegate_ == nullptr && context_->IsGpuEnabled()) {
    delegate_ = std::make_shared<TensorRTDelegate>(ms_context_);
    if (delegate_ == nullptr) {
      MS_LOG(ERROR) << "New tensorrt delegate_ failed";
      return RET_ERROR;
    }
    delegate_device_type_ = DT_GPU;
    this->context_->delegate = delegate_;
  }
#endif

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

  for (auto item : graph_output_map_) {
    auto isolate_output_tensor = item.first;
    isolate_output_tensor->set_data(nullptr);
    delete isolate_output_tensor;
    isolate_output_tensor = nullptr;
  }

  // Tensor * in input_map output_map are freed in tensors
  input_map_.clear();
  output_node_map_.clear();
  output_tensor_map_.clear();
  input_vec_.clear();
  graph_output_map_.clear();

  delete this->executor_;
  this->executor_ = nullptr;
#if GPU_OPENCL
  delete opencl_runtime_wrapper_;
  opencl_runtime_wrapper_ = nullptr;
#endif
  delete ms_context_;
  ms_context_ = nullptr;
  delete this->context_;
  this->context_ = nullptr;
  delete (model_);
  model_ = nullptr;
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
  if (!is_train_session_) {
    executor_->Resize(inputs, dims);
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
  for (auto kernel : kernels) {
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "input kernel is nullptr!";
      return RET_ERROR;
    }
    auto ret = RET_OK;
#ifndef DELEGATE_CLIP
    if (kernel->desc().arch == kernel::kDelegate) {
      ret = kernel->ReSize();
    } else {
#endif
      if (kernel->subgraph_type() == kernel::kGpuFp16SubGraph || kernel->subgraph_type() == kernel::kGpuFp32SubGraph) {
#if GPU_OPENCL
        auto sub_graph = reinterpret_cast<kernel::OpenCLSubGraph *>(kernel);
        ret = sub_graph->ReSize(false);
#endif
      } else {
        auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
        ret = sub_graph->ReSize();
      }
#ifndef DELEGATE_CLIP
    }
#endif
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
#if !defined(_WIN32) && !defined(ENABLE_ARM)
  (void)malloc_trim(0);  // release memory manually.
#endif
  return RET_OK;
}

int LiteSession::InitGPURuntime() {
  if (context_->IsCpuEnabled()) {
    CpuBindMode cpu_bind_mode = context_->GetCpuDeviceInfo()->cpu_bind_mode_;
    ThreadPool *thread_pool = this->context_->thread_pool();
    if (thread_pool == nullptr) {
      MS_LOG(ERROR) << "thread pool is nullptr";
      is_running_.store(false);
      return RET_NULL_PTR;
    }
    thread_pool->SetProcessAffinity(static_cast<BindMode>(cpu_bind_mode));
  }
#if GPU_OPENCL
  if (this->context_->IsGpuEnabled()) {
    opencl_runtime_wrapper_ = new (std::nothrow) opencl::OpenCLRuntimeInnerWrapper();
    if (opencl_runtime_wrapper_ == nullptr) {
      MS_LOG(ERROR) << "create OpenCLRuntimeInnerWrapper failed";
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

    /* check chip support shared memory */
    auto enable_arm_import_memory = opencl_runtime->isExtensionEnable(EXT_ARM_IMPORT_MEMORY_HOST);
    if (!enable_arm_import_memory) {
      MS_LOG(WARNING) << "GPU do not support shared memory!";
    }
  }
#endif
  // Setting the binding core will affect the opencl drive scheduling.
  if (context_->IsCpuEnabled()) {
    ThreadPool *thread_pool = this->context_->thread_pool();
    thread_pool->SetProcessAffinity(static_cast<BindMode>(NO_BIND));
  }
  return RET_OK;
}
}  // namespace lite

session::LiteSession *session::LiteSession::CreateSession(const lite::Context *context) {
  if (context == nullptr) {
    return nullptr;
  }

  auto session = new (std::nothrow) lite::LiteSession();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }

  mindspore::lite::InnerContext *inner_context = new (std::nothrow) mindspore::lite::InnerContext(context);
  if (inner_context == nullptr) {
    MS_LOG(ERROR) << "new inner context failed";
    delete session;
    return nullptr;
  }

  auto ret = session->Init(inner_context);
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
  auto ret = reinterpret_cast<lite::LiteSession *>(session)->LoadModelAndCompileByBuf(model_buf, size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init session failed";
    delete session;
    return nullptr;
  }
  return session;
}

session::LiteSession *lite::LiteSession::CreateSession(const std::string &model_path, const lite::Context *context) {
  auto *session = session::LiteSession::CreateSession(context);
  if (session == nullptr) {
    MS_LOG(ERROR) << "Create session failed";
    return nullptr;
  }
  auto ret = reinterpret_cast<lite::LiteSession *>(session)->LoadModelAndCompileByPath(model_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init session failed";
    delete session;
    return nullptr;
  }
  return session;
}

int lite::LiteSession::LoadModelAndCompileByBuf(const char *model_buf, size_t buf_size) {
  auto *model = lite::ImportFromBuffer(model_buf, buf_size, true);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return RET_ERROR;
  }
  auto ret = CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    model->buf = nullptr;
    delete model;
    return RET_ERROR;
  }
  model->buf = nullptr;
  set_model(model);
  return RET_OK;
}

int lite::LiteSession::LoadModelAndCompileByPath(const std::string &model_path) {
  size_t model_size;
  auto model_buf = lite::ReadFile(model_path.c_str(), &model_size);
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed";
    return RET_ERROR;
  }
  auto *model = lite::ImportFromBuffer(model_buf, model_size, true);
  if (model == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return RET_ERROR;
  }
  (reinterpret_cast<lite::LiteModel *>(model))->set_keep_model_buf(true);
  auto ret = CompileGraph(model);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Compile model failed";
    return RET_ERROR;
  }
  set_model(model);
  return RET_OK;
}

}  // namespace mindspore

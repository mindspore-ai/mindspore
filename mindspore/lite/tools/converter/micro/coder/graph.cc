/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/graph.h"
#include <queue>
#include <deque>
#include <string>
#include <algorithm>
#include <set>
#include "coder/log.h"
#include "coder/opcoders/op_coder_register.h"
#include "coder/utils/type_cast.h"
#include "coder/utils/train_utils.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/common/prim_util.h"
#include "src/litert/lite_model.h"
#include "base/float16.h"

namespace mindspore::lite::micro {
CoderGraph::~CoderGraph() {
  if (model_ != nullptr) {
    model_->Free();
    delete model_;
    model_ = nullptr;
  }
  for (auto &tensor : all_tensors_) {
    delete tensor;
  }
}

int CoderGraph::ConvertTensors(bool enable_fp16) {
  MS_CHECK_PTR(model_);
  std::vector<Tensor *> all_tensors;
  auto clear_tensors = [&all_tensors]() {
    std::for_each(all_tensors.begin(), all_tensors.end(), [](Tensor *&t) {
      delete t;
      t = nullptr;
    });
    all_tensors.clear();
  };
  auto check_dim = [](int dim) -> int {
    MS_CHECK_TRUE(dim > 0, "invalid dim value!");
    return RET_OK;
  };

  // deal with allTensors
  uint32_t tensorCount = model_->graph_.all_tensors_.size();
  for (uint32_t i = 0; i < tensorCount; ++i) {
    schema::Tensor *origin_tensor = model_->graph_.all_tensors_.at(i);
    MS_CHECK_PTR_WITH_EXE(origin_tensor, clear_tensors());
    // tensor dims
    std::vector<int> shape;
    if (origin_tensor->dims() != nullptr) {
      for (uint32_t j = 0; j < origin_tensor->dims()->size(); j++) {
        MS_CHECK_PTR(origin_tensor->dims()->data());
        int dim = static_cast<int>(origin_tensor->dims()->data()[j]);
        MS_CHECK_RET_CODE_WITH_EXE(check_dim(dim), "parse shape failed!", clear_tensors());
        shape.push_back(dim);
      }
    }

    int origin_data_type = static_cast<int>(origin_tensor->dataType());
    Tensor *dstTensor = new (std::nothrow)
      lite::Tensor(TypeId(origin_data_type), shape, static_cast<mindspore::Format>(origin_tensor->format()),
                   TensorCategory(*origin_tensor));
    MS_CHECK_PTR(dstTensor);
    if (origin_tensor->nodeType() == NodeType_ValueNode && origin_tensor->data() != nullptr &&
        origin_tensor->data()->size() > 0) {
      // copy data, this is weight && bias
      if (enable_fp16 && origin_data_type == kNumberTypeFloat32) {
        dstTensor->set_data_type(kNumberTypeFloat16);
        auto data = dstTensor->MutableData();
        MS_CHECK_PTR_WITH_EXE(data, delete dstTensor);
        auto fp32_data = reinterpret_cast<const float *>(origin_tensor->data()->data());
        auto fp16_data = reinterpret_cast<float16 *>(data);
        CHECK_NULL_RETURN(fp32_data);
        CHECK_NULL_RETURN(fp16_data);
        for (int64_t j = 0; j < dstTensor->ElementsNum(); ++j) {
          fp16_data[j] = float16(fp32_data[j]);
        }
      } else {
        if (memcpy_s(dstTensor->MutableData(), dstTensor->Size(), origin_tensor->data()->data(),
                     origin_tensor->data()->size()) != EOK) {
          MS_LOG(ERROR) << "memcpy_s copy data failed!";
          delete dstTensor;
          return RET_ERROR;
        }
      }
    } else if (enable_fp16 && origin_data_type == kNumberTypeFloat32) {
      dstTensor->set_data_type(kNumberTypeFloat16);
    }
    if (origin_tensor->name() != nullptr) {
      dstTensor->set_tensor_name(origin_tensor->name()->str());
    }
    auto quant_params = origin_tensor->quantParams();
    if (quant_params != nullptr) {
      for (int j = 0; j < static_cast<int>(quant_params->size()); j++) {
        auto quant_param = quant_params->Get(j);
        LiteQuantParam quant_arg{};
        if (quant_param == nullptr) {
          quant_arg.inited = false;
        } else {
          quant_arg.inited = quant_param->inited();
          quant_arg.bitNum = quant_param->numBits();
          quant_arg.scale = quant_param->scale();
          quant_arg.zeroPoint = quant_param->zeroPoint();
          quant_arg.var_corr = quant_param->varCorr();
          quant_arg.mean_corr = quant_param->meanCorr();
          quant_arg.roundType = quant_param->roundType();
          quant_arg.multiplier = quant_param->multiplier();
          quant_arg.dstDtype = quant_param->dstDtype();
        }
        dstTensor->AddQuantParam(quant_arg);
      }
    }
    all_tensors.emplace_back(dstTensor);
  }
  SetAllTensors(all_tensors);
  return RET_OK;
}

int CoderGraph::InitGraphInOutTensors() {
  if (model_ == nullptr) {
    return RET_ERROR;
  }
  std::vector<size_t> graph_input_node_indexes = lite::GetGraphInputNodes(model_);
  std::vector<uint32_t> input_indices;
  for (auto in_node_index : graph_input_node_indexes) {
    in_node_index = static_cast<uint32_t>(in_node_index);
    auto in_node = model_->graph_.all_nodes_.at(in_node_index);
    MS_CHECK_PTR(in_node);
    for (uint32_t i = 0; i < in_node->input_indices_.size(); i++) {
      auto in_tensor_index = size_t(in_node->input_indices_.at(i));
      bool is_graph_input = false;
      for (uint32_t j = 0; j < model_->graph_.input_indices_.size(); j++) {
        if (in_tensor_index == size_t(model_->graph_.input_indices_.at(j))) {
          input_indices.push_back(static_cast<uint32_t>(in_tensor_index));
          is_graph_input = true;
          break;
        }
      }
      if (!is_graph_input) {
        continue;
      }
      if (in_tensor_index < all_tensors_.size()) {
        lite::Tensor *in_tensor = all_tensors_.at(in_tensor_index);
        AddInputMap(in_node->name_, in_tensor);
      }
    }
  }
  SetInputIndices(input_indices);
  std::vector<uint32_t> output_indices;
  auto graph_output_node_indexes = lite::GetGraphOutputNodes(model_);
  for (auto out_node_index : graph_output_node_indexes) {
    out_node_index = static_cast<uint32_t>(out_node_index);
    auto *out_node = model_->graph_.all_nodes_.at(out_node_index);
    for (uint32_t i = 0; i < out_node->output_indices_.size(); i++) {
      auto out_tensor_index = size_t(out_node->output_indices_.at(i));
      bool is_graph_output = false;
      for (uint32_t j = 0; j < model_->graph_.output_indices_.size(); j++) {
        if (out_tensor_index == size_t(model_->graph_.output_indices_.at(j))) {
          output_indices.push_back(static_cast<uint32_t>(out_tensor_index));
          is_graph_output = true;
          break;
        }
      }
      if (!is_graph_output) {
        continue;
      }
      if (out_tensor_index < all_tensors_.size()) {
        lite::Tensor *out_tensor = all_tensors_.at(out_tensor_index);
        if (out_tensor == nullptr) {
          MS_LOG(ERROR) << "can not find any output tensor in all_tensors";
          return RET_ERROR;
        }
        AddOutputMap(out_node->name_, out_tensor);
      }
    }
  }
  SetOutputIndices(output_indices);
  int ret = InitInputs();
  MS_CHECK_RET_CODE(ret, "init graph input tensors failed.");
  InitOutputs();
  return RET_OK;
}

std::vector<lite::Tensor *> CoderGraph::input_tensors() const { return input_tensors_; }

std::vector<lite::Tensor *> CoderGraph::output_tensors() const { return output_tensors_; }

std::vector<lite::Tensor *> CoderGraph::eval_output_tensors() const { return eval_output_tensors_; }

std::vector<lite::Tensor *> CoderGraph::train_output_tensors() const { return train_output_tensors_; }

int CoderGraph::InitInputs() {
  input_tensors_.clear();
  auto graph_in_size = model_->graph_.input_indices_.size();
  for (size_t i = 0; i < graph_in_size; i++) {
    auto in_tensor_idx = model_->graph_.input_indices_[i];
    MS_CHECK_TRUE_MSG(in_tensor_idx < all_tensors_.size(), RET_ERROR, "in tensor idx is out of range.");
    auto in_tensor = all_tensors_.at(in_tensor_idx);
    MS_CHECK_TRUE_MSG(in_tensor != nullptr, RET_ERROR, "in_tensor is nullptr.");
    input_tensors_.emplace_back(in_tensor);
  }
  return RET_OK;
}

void CoderGraph::InitOutputs() {
  output_tensors_.clear();
  (void)std::transform(output_indices_.begin(), output_indices_.end(), std::back_inserter(output_tensors_),
                       [&](uint32_t a) { return this->all_tensors_.at(a); });
}

int CoderGraph::CompileTrainOutputs(const std::vector<OperatorCoder *> &train_coders) {
  train_outputs_map_.clear();
  train_output_tensors_.clear();
  for (auto train_coder : train_coders) {
    MS_CHECK_TRUE_MSG(train_coder != nullptr, RET_ERROR, "train coder is nullptr.");
    if (outputs_map_.find(train_coder->name()) == outputs_map_.end() || IsMaskOutput(train_coder) ||
        train_outputs_map_.find(train_coder->name()) != train_outputs_map_.end()) {  // filter optimizer out tensors out
      continue;
    }
    MS_CHECK_TRUE_MSG(!train_coder->output_tensors().empty(), RET_ERROR, "output tensors is empty.");
    auto ms_tensor = train_coder->output_tensors().at(0);
    if (ms_tensor != nullptr) {
      train_outputs_map_[train_coder->name()].emplace_back(ms_tensor);
      train_output_tensors_.emplace_back(ms_tensor);
    }
  }
  if (train_outputs_map_.empty()) {
    train_outputs_map_ = outputs_map_;
  }
  if (train_output_tensors_.empty()) {
    train_output_tensors_ = output_tensors_;
  }
  return RET_OK;
}

int CoderGraph::CompileEvalOutputs(const std::vector<OperatorCoder *> &train_coders) {
  eval_outputs_map_.clear();
  eval_output_tensors_.clear();
  for (auto coder : train_coders) {
    MS_CHECK_TRUE_MSG(coder != nullptr, RET_ERROR, "coder is nullptr.");
    if (!IsLossCoder(coder) || IsGradCoder(coder)) {
      continue;
    }
    for (auto in_coder : coder->input_ops()) {
      if (IsLossCoder(in_coder) || IsGradCoder(in_coder)) {
        continue;
      }
      auto in_in_coders = in_coder->input_ops();
      bool is_loss = std::any_of(in_in_coders.begin(), in_in_coders.end(),
                                 [](const OperatorCoder *coder) { return IsLossCoder(coder); });
      if (is_loss || eval_outputs_map_.find(in_coder->name()) != eval_outputs_map_.end()) {
        continue;
      }
      MS_CHECK_TRUE_MSG(!in_coder->output_tensors().empty(), RET_ERROR, "output tensors is empty.");
      auto ms_tensor = in_coder->output_tensors().at(0);
      if (ms_tensor != nullptr) {
        ms_tensor->set_init_ref_count(ms_tensor->init_ref_count() + 1);
        eval_outputs_map_[in_coder->name()].emplace_back(ms_tensor);
        eval_output_tensors_.emplace_back(ms_tensor);
      }
    }
  }
  if (eval_outputs_map_.empty()) {
    eval_outputs_map_ = outputs_map_;
  }
  if (eval_output_tensors_.empty()) {
    eval_output_tensors_ = output_tensors_;
  }
  return RET_OK;
}

void CoderGraph::SetAllTensors(const std::vector<Tensor *> &all_tensors) {
  all_tensors_.insert(all_tensors_.end(), all_tensors.begin(), all_tensors.end());
}

void CoderGraph::SetInputIndices(const std::vector<uint32_t> &input_indices) {
  input_indices_.insert(input_indices_.end(), input_indices.begin(), input_indices.end());
}

void CoderGraph::SetOutputIndices(const std::vector<uint32_t> &output_indices) {
  output_indices_.insert(output_indices_.end(), output_indices.begin(), output_indices.end());
}

void CoderGraph::AddInputMap(const std::string &node_id, Tensor *input_tensor) {
  if (!input_tensor) {
    MS_LOG(ERROR) << "input tensor is nullptr, can not added to coder_graph";
    return;
  }
  this->inputs_map_[node_id].emplace_back(input_tensor);
}

void CoderGraph::AddOutputMap(const std::string &node_id, Tensor *output_tensor) {
  if (!output_tensor) {
    MS_LOG(ERROR) << "output tensor is nullptr, can not added to coder_graph";
    return;
  }
  this->outputs_map_[node_id].emplace_back(output_tensor);
}

std::vector<lite::Tensor *> CoderGraph::all_tensors() const { return this->all_tensors_; }

const std::map<std::string, std::vector<lite::Tensor *>> &CoderGraph::GetOutputsMap() const { return outputs_map_; }

const std::map<std::string, std::vector<Tensor *>> &CoderGraph::GetEvalOutputsMap() const { return eval_outputs_map_; }

std::vector<uint32_t> CoderGraph::input_indices() const { return this->input_indices_; }

std::vector<uint32_t> CoderGraph::output_indices() const { return this->output_indices_; }

void CoderGraph::DumpUnSupportLayer(Target target) {
  std::cerr << "==========dump all unsupported layer for codegen=====" << std::endl;
  std::for_each(
    model_->graph_.all_nodes_.begin(), model_->graph_.all_nodes_.end(), [this, target](const LiteGraph::Node *node) {
      if (node->primitive_ == nullptr) {
        return;
      }
      // fake create opcoders
      uint32_t input_idx = node->input_indices_.at(0);
      Tensor *t = all_tensors_.at(input_idx);
      TypeId dtype = t->data_type();
      int pt = GetPrimitiveType(node->primitive_, reinterpret_cast<lite::LiteModel *>(model_)->GetSchemaVersion());
      CoderKey key(target, dtype, pt);
      // search from the opcoder registry
      if (OpCoderFactory::GetInstance()->FindOpCoder(key) == nullptr) {
        std::cerr << node->name_ << ", primitive type: "
                  << mindspore::schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(pt))
                  << ", data_type: " << EnumNameDataType(dtype) << std::endl;
      }
    });
}
int CoderGraph::RemoveCast() {
  auto graph = &model_->graph_;
  std::unordered_map<uint32_t, std::vector<size_t>> tensor_post_ops;  // a tensor in many operator's inputs
  MS_ASSERT(graph.sub_graphs_.size() == 1);

  for (size_t i = 0; i < graph->all_nodes_.size(); i++) {
    for (auto tensor_idx : graph->all_nodes_[i]->input_indices_) {
      tensor_post_ops[tensor_idx].emplace_back(i);
    }
  }

  std::set<uint32_t> removed_cast_set;

  for (size_t i = 0; i < graph->all_nodes_.size(); i++) {
    auto node = graph->all_nodes_[i];
    if (node->node_type_ == mindspore::schema::PrimitiveType_Cast) {
      auto cast_output_tensor = node->output_indices_.front();
      auto cast_input_tensor = node->input_indices_.front();
      for (const auto &post_op_idx : tensor_post_ops[cast_output_tensor]) {
        auto post_op = graph->all_nodes_[post_op_idx];
        auto iter = std::find(post_op->input_indices_.begin(), post_op->input_indices_.end(), cast_output_tensor);
        if (iter == post_op->input_indices_.end()) {
          MS_LOG(ERROR) << "this op is not cast output op";
          return RET_ERROR;
        }
        *iter = cast_input_tensor;
        removed_cast_set.insert(i);
      }
    }
  }

  std::vector<LiteGraph::Node *> new_all_nodes;
  for (size_t i = 0; i < graph->all_nodes_.size(); i++) {
    if (removed_cast_set.find(i) == removed_cast_set.end()) {
      new_all_nodes.emplace_back(graph->all_nodes_[i]);
    }
  }

  graph->all_nodes_.swap(new_all_nodes);

  return RET_OK;
}
}  // namespace mindspore::lite::micro

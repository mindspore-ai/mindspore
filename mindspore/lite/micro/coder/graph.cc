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
#include <memory>
#include <algorithm>
#include <set>
#include "coder/log.h"
#include "coder/opcoders/op_coder_register.h"
#include "coder/utils/type_cast.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/common/prim_util.h"
#include "src/lite_model.h"

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

int CoderGraph::ConvertTensors() {
  if (model_ == nullptr) {
    MS_LOG(ERROR) << "Graph model is nullptr";
    return RET_ERROR;
  }
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
  uint32_t tensorCount = model_->all_tensors_.size();
  for (uint32_t i = 0; i < tensorCount; ++i) {
    schema::Tensor *origin_tensor = model_->all_tensors_.at(i);
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
    // tensor Datatype
    if (shape.empty()) {
      shape.push_back(1);
    }
    int origin_data_type = static_cast<int>(origin_tensor->dataType());
    Tensor *dstTensor = new (std::nothrow)
      lite::Tensor(TypeId(origin_data_type), shape, static_cast<mindspore::Format>(origin_tensor->format()),
                   TensorCategory(origin_tensor));
    MS_CHECK_PTR(dstTensor);
    if (origin_tensor->nodeType() == NodeType_ValueNode && origin_tensor->data() != nullptr &&
        origin_tensor->data()->size() > 0) {
      // copy data, this is weight && bias
      MS_CHECK_TRUE_WITH_EXE(origin_tensor->data()->size() > 0, "invalid meta_tensor data size.", delete dstTensor);
      auto data_size = static_cast<size_t>(origin_tensor->data()->size());
      MS_CHECK_RET_CODE_WITH_EXE(dstTensor->MallocData(), "dst tensor malloc data failed!", delete dstTensor);
      void *dst_data = dstTensor->data();
      MS_CHECK_RET_CODE_WITH_EXE(memcpy_s(dst_data, dstTensor->Size(), origin_tensor->data()->data(), data_size),
                                 "memcpy_s copy data failed!", delete dstTensor);
      dstTensor->set_data(dst_data);
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
    auto in_node = model_->all_nodes_.at(in_node_index);
    MS_CHECK_PTR(in_node);
    for (uint32_t i = 0; i < in_node->input_indices_.size(); i++) {
      auto in_tensor_index = size_t(in_node->input_indices_.at(i));
      bool is_graph_input = false;
      for (uint32_t j = 0; j < model_->input_indices_.size(); j++) {
        if (in_tensor_index == size_t(model_->input_indices_.at(j))) {
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
    auto *out_node = model_->all_nodes_.at(out_node_index);
    for (uint32_t i = 0; i < out_node->output_indices_.size(); i++) {
      auto out_tensor_index = size_t(out_node->output_indices_.at(i));
      bool is_graph_output = false;
      for (uint32_t j = 0; j < model_->output_indices_.size(); j++) {
        if (out_tensor_index == size_t(model_->output_indices_.at(j))) {
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
  InitInputs();
  InitOutputs();
  return RET_OK;
}

std::vector<lite::Tensor *> CoderGraph::input_tensors() const { return input_tensors_; }

std::vector<lite::Tensor *> CoderGraph::output_tensors() const { return output_tensors_; }

void CoderGraph::InitInputs() {
  for (const auto &pair : inputs_map_) {
    std::vector<Tensor *> tensors = pair.second;
    input_tensors_.insert(input_tensors_.end(), tensors.begin(), tensors.end());
  }
  // remove duplicate tensors
  std::set<lite::Tensor *> unique;
  unique.insert(input_tensors_.begin(), input_tensors_.end());
  input_tensors_.clear();
  input_tensors_.insert(input_tensors_.end(), unique.begin(), unique.end());
}

void CoderGraph::InitOutputs() {
  std::transform(output_indices_.begin(), output_indices_.end(), std::back_inserter(output_tensors_),
                 [&](uint32_t a) { return this->all_tensors_.at(a); });
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

std::vector<uint32_t> CoderGraph::input_indices() const { return this->input_indices_; }

std::vector<uint32_t> CoderGraph::output_indices() const { return this->output_indices_; }

void CoderGraph::DumpUnSupportLayer(Target target) {
  std::cerr << "==========dump all unsupported layer for codegen=====" << std::endl;
  std::for_each(model_->all_nodes_.begin(), model_->all_nodes_.end(), [this, target](const Model::Node *node) {
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
}  // namespace mindspore::lite::micro

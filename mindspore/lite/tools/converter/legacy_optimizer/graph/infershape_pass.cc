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

#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include <vector>
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "src/common/prim_util.h"
#include "src/ops/populate/populate_register.h"
#include "src/runtime/infer_manager.h"
#include "tools/common/node_util.h"
#include "tools/converter/converter_flags.h"
#include "src/common/string_util.h"

using mindspore::lite::converter::FmkType_TF;
namespace mindspore {
namespace lite {
namespace {
constexpr int DEFAULT_DIM_VALUE = -1;
constexpr size_t kInitialSize = 1024;

void FreeTensors(std::vector<Tensor *> *input_tensors, std::vector<Tensor *> *output_tensors) {
  if (input_tensors == nullptr) {
    return;
  }
  for (auto &tensor : *input_tensors) {
    delete tensor;
    tensor = nullptr;
  }
  if (output_tensors == nullptr) {
    return;
  }
  for (auto &tensor : *output_tensors) {
    delete tensor;
    tensor = nullptr;
  }
  input_tensors->resize(0);
  output_tensors->resize(0);
}

void ConvertTensorList(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  auto tensor_shape = tensorT->dims;
  TypeId type = kTypeUnknown;
  std::vector<int> element_shape;
  if (!tensorT->data.empty()) {
    int *data = reinterpret_cast<int *>(tensorT->data.data());
    type = TypeId(data[0]);
    if (tensorT->data.size() < 8 || (data[1] != 0 && (data[1] + 2) * 4 != static_cast<int>(tensorT->data.size()))) {
      MS_LOG(ERROR) << "tensorlist data length illegal";
      *convert_succ = false;
      return;
    }
    for (int j = 0; j < data[1]; ++j) {
      element_shape.push_back(data[j + 2]);
    }
  }
  lite_tensor = std::make_unique<TensorList>(tensor_shape, element_shape);
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensorlist is nullptr";
    *convert_succ = false;
    return;
  }
  reinterpret_cast<TensorList *>(lite_tensor.get())->set_tensors_data_type(type);
  lite_tensors->emplace_back(lite_tensor.release());
}

void ConvertString(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  auto tensor_shape = tensorT->dims;
  lite_tensor = std::make_unique<Tensor>(
    TypeId(tensorT->dataType), tensor_shape, tensorT->format,
    TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensor is nullptr";
    *convert_succ = false;
    return;
  }
  auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
  // when tensorT as param input
  if (lite_tensor_size == 0) {
    lite_tensors->emplace_back(lite_tensor.release());
    return;
  }
  auto string_buffer = ParseStringBuffer(tensorT->data.data());
  auto ret = WriteStringsToTensor(lite_tensor.get(), string_buffer);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WriteStringsToTensor failed";
    *convert_succ = false;
    return;
  }
  lite_tensors->emplace_back(lite_tensor.release());
}

void ConvertOtherTensor(MetaGraphT *graph, uint32_t index, bool *convert_succ, std::vector<Tensor *> *lite_tensors) {
  std::unique_ptr<Tensor> lite_tensor = nullptr;
  auto &tensorT = graph->allTensors.at(index);
  auto tensor_shape = tensorT->dims;
  lite_tensor = std::make_unique<Tensor>(
    TypeId(tensorT->dataType), tensor_shape, tensorT->format,
    TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
  if (lite_tensor == nullptr) {
    MS_LOG(ERROR) << "lite tensor is nullptr";
    *convert_succ = false;
    return;
  }
  auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
  // when tensorT as param input
  if (lite_tensor_size == 0) {
    lite_tensors->emplace_back(lite_tensor.release());
    return;
  }
  auto ret = lite_tensor->MallocData();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Malloc tensor data failed";
    *convert_succ = false;
    return;
  }
  if (memcpy_s(lite_tensor->data_c(), lite_tensor->Size(), tensorT->data.data(), tensorT->data.size()) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed";
    *convert_succ = false;
    return;
  }
  lite_tensors->emplace_back(lite_tensor.release());
}

std::vector<Tensor *> ConvertTensorToLiteTensor(MetaGraphT *graph, const std::vector<uint32_t> &tensor_indexs) {
  MS_ASSERT(graph != nullptr);
  std::vector<Tensor *> lite_tensors;
  bool convert_succ = true;
  for (size_t i = 0; i < tensor_indexs.size(); i++) {
    auto &tensorT = graph->allTensors.at(tensor_indexs[i]);
    switch (tensorT->dataType) {
      case kObjectTypeTensorType:
        ConvertTensorList(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
      case kObjectTypeString:
        ConvertString(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
      default:
        ConvertOtherTensor(graph, tensor_indexs[i], &convert_succ, &lite_tensors);
        break;
    }
  }
  if (!convert_succ) {
    FreeTensors(&lite_tensors, {});
    return {};
  }
  return lite_tensors;
}

STATUS NodeInferShape(const std::unique_ptr<schema::CNodeT> &node, const std::vector<Tensor *> &inputs,
                      std::vector<Tensor *> *outputs, bool infer_interrupt) {
  flatbuffers::FlatBufferBuilder fbb(kInitialSize);
  auto prim = ConvertToPrimitive(node->primitive.get(), &fbb);
  if (prim == nullptr) {
    MS_LOG(ERROR) << "get primitive failed.";
    fbb.Clear();
    return RET_ERROR;
  }
  auto parameter_gen = lite::PopulateRegistry::GetInstance()->GetParameterCreator(prim->value_type(), SCHEMA_CUR);
  if (parameter_gen == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: " << schema::EnumNamePrimitiveType(prim->value_type());
    return RET_ERROR;
  }
  auto parameter = parameter_gen(prim);
  if (parameter == nullptr) {
    fbb.Clear();
    MS_LOG(ERROR) << "parameter is nullptr.";
    return RET_ERROR;
  }
  parameter->quant_type_ = node->quantType;
  if (infer_interrupt) {
    parameter->infer_flag_ = false;
  } else {
    parameter->infer_flag_ = true;
  }
  auto ret = KernelInferShape(inputs, outputs, parameter);
  fbb.Clear();
  free(parameter);
  return ret;
}

#ifdef Debug
void PrintTensorShape(const std::vector<Tensor *> &input_tensors, const std::vector<Tensor *> &output_tensors) {
  int i = 0;
  for (auto input_tensor : input_tensors) {
    std::ostringstream oss;
    for (auto &dim : input_tensor->shape()) {
      oss << " " << dim;
    }
    MS_LOG(DEBUG) << "input shape " << i++ << ":" << oss.str();
  }
  i = 0;
  for (auto output_tensor : output_tensors) {
    std::ostringstream oss;
    for (auto &dim : output_tensor->shape()) {
      oss << " " << dim;
    }
    MS_LOG(DEBUG) << "output shape" << i++ << ":" << oss.str();
  }
}
#endif

void SetDataType(MetaGraphT *graph, const std::vector<Tensor *> &output_tensors, std::vector<InferTensor> *tensors_,
                 uint32_t i, uint32_t infer_node_index) {
  auto &node = graph->nodes.at(infer_node_index);
  auto &output_tensor = graph->allTensors.at(node->outputIndex[i]);
  output_tensor->format = output_tensors[i]->format();
  output_tensor->dataType = output_tensors[i]->data_type();
  if (output_tensors[i]->data_type() == kObjectTypeTensorType) {
    auto tensor_list = reinterpret_cast<TensorList *>(output_tensors[i]);
    if (output_tensor->data.empty()) {
      output_tensor->data.resize(8, 0);
    }
    if (tensor_list->tensors_data_type() == kTypeUnknown) {
      tensors_->at(node->outputIndex[i]).is_inferred_ = false;
      return;
    }
    output_tensor->data.at(0) = tensor_list->tensors_data_type();
  } else if (output_tensors[i]->data_type() == kTypeUnknown) {
    tensors_->at(node->outputIndex[i]).is_inferred_ = false;
    return;
  }
  tensors_->at(node->outputIndex[i]).is_inferred_ = true;
  return;
}
}  // namespace

STATUS InferShapePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  InitSearchTensor(graph);
  for (auto input_idx : graph->inputIndex) {
    auto input_tensor = graph->allTensors[input_idx].get();
    for (auto &dim : input_tensor->dims) {
      if (dim == 0) {
        MS_LOG(WARNING) << "One dimension of the input shape is 0, which would be set to -1 as a default value.";
        dim = DEFAULT_DIM_VALUE;
      }
    }
    auto input_shape = graph->allTensors.at(input_idx)->dims;
    if (std::find(input_shape.begin(), input_shape.end(), -1) != input_shape.end() || fmk_type_ == FmkType_TF) {
      infer_interrupt_ = true;
    }
  }
  while (!infer_node_indexes_.empty()) {
    auto infer_node_index = infer_node_indexes_.front();
    auto &node = graph->nodes.at(infer_node_index);
    auto node_type = node->primitive->value.type;
    if (node_type == PrimitiveType_Switch && node->outputIndex.size() != 2 * (node->inputIndex.size() - 1)) {
      MS_LOG(WARNING) << "do infershape after switch pass.";
      return RET_OK;
    }
    infer_node_indexes_.erase(infer_node_indexes_.begin());
    if (node_type == PrimitiveType_PartialFusion) {
      continue;
    }
    auto input_tensors = ConvertTensorToLiteTensor(graph, node->inputIndex);
    auto output_tensors = ConvertTensorToLiteTensor(graph, node->outputIndex);
    if (output_tensors.empty() || output_tensors.size() != node->outputIndex.size() || input_tensors.empty() ||
        input_tensors.size() != node->inputIndex.size()) {
      MS_LOG(ERROR) << "convert lite tensor error";
      FreeTensors(&input_tensors, &output_tensors);
      return RET_INFER_ERR;
    }
    auto status = NodeInferShape(node, input_tensors, &output_tensors, infer_interrupt_);
    MS_LOG(DEBUG) << "cur node:" << node->name;
    if (status == RET_OK) {
#ifdef Debug
      PrintTensorShape(input_tensors, output_tensors);
#endif
      // copy output shape to tensorT
      for (size_t i = 0; i < output_tensors.size(); i++) {
        auto output_dims = output_tensors[i]->shape();
        auto &output_tensor = graph->allTensors.at(node->outputIndex[i]);
        output_tensor->dims.swap(output_dims);
        SetDataType(graph, output_tensors, &tensors_, i, infer_node_index);
      }
    } else if (status == RET_INFER_INVALID) {
      for (size_t i = 0; i < output_tensors.size(); i++) {
        SetDataType(graph, output_tensors, &tensors_, i, infer_node_index);
      }
      infer_interrupt_ = true;
    } else {
      MS_LOG(WARNING) << "InferShape failed, name: " << node->name
                      << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type);
      FreeTensors(&input_tensors, &output_tensors);
      return RET_INFER_ERR;
    }
    FreeTensors(&input_tensors, &output_tensors);
    AddOutputNodes(graph, infer_node_index);
  }
  return RET_OK;
}

void InferShapePass::InitSearchTensor(MetaGraphT *graph) {
  std::vector<uint32_t> all_node_output_tensor_indexes = {};
  tensors_.resize(graph->allTensors.size());
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    auto node_input_indexes = node->inputIndex;
    //  init in_nodes index
    for (size_t j = 0; j < node_input_indexes.size(); j++) {
      tensors_[node_input_indexes[j]].next_nodes_.push_back(i);
    }
    auto node_output_indexes = node->outputIndex;
    for (size_t j = 0; j < node_output_indexes.size(); j++) {
      tensors_[node_output_indexes[j]].prev_nodes_.push_back(i);
    }
    all_node_output_tensor_indexes.insert(all_node_output_tensor_indexes.end(), node_output_indexes.begin(),
                                          node_output_indexes.end());
  }
  for (uint32_t i = 0; i < tensors_.size(); i++) {
    if (tensors_[i].prev_nodes_.empty() || IsContain(graph->inputIndex, i) || !graph->allTensors.at(i)->data.empty()) {
      tensors_[i].is_inferred_ = true;
    }
  }
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    if (std::all_of(node->inputIndex.begin(), node->inputIndex.end(),
                    [&](uint32_t idx) { return tensors_[idx].is_inferred_; })) {
      infer_node_indexes_.push_back(i);
    }
  }
}

void InferShapePass::AddOutputNodes(MetaGraphT *graph, uint32_t infer_node_index) {
  auto &node = graph->nodes.at(infer_node_index);
  for (size_t i = 0; i < node->outputIndex.size(); i++) {
    auto next_nodes_indexes = tensors_[node->outputIndex[i]].next_nodes_;
    for (size_t j = 0; j < next_nodes_indexes.size(); j++) {
      auto &next_node = graph->nodes.at(next_nodes_indexes[j]);
      if (std::any_of(next_node->outputIndex.begin(), next_node->outputIndex.end(),
                      [&](uint32_t idx) { return !tensors_[idx].is_inferred_; })) {
        AddNextInferShapeNode(graph, next_nodes_indexes, j);
      }
    }
  }
}

void InferShapePass::AddNextInferShapeNode(MetaGraphT *graph, std::vector<uint32_t> next_nodes_indexes, size_t index) {
  auto &next_node = graph->nodes.at(next_nodes_indexes[index]);
  if (find(infer_node_indexes_.begin(), infer_node_indexes_.end(), next_nodes_indexes[index]) ==
      infer_node_indexes_.end()) {
    auto next_node_type = next_node->primitive->value.type;
    if (next_node_type == schema::PrimitiveType_Merge) {
      if (std::all_of(next_node->inputIndex.begin(), next_node->inputIndex.begin() + next_node->inputIndex.size() / 2,
                      [&](uint32_t i) { return tensors_[i].is_inferred_; }) ||
          std::all_of(next_node->inputIndex.begin() + next_node->inputIndex.size() / 2, next_node->inputIndex.end(),
                      [&](uint32_t i) { return tensors_[i].is_inferred_; })) {
        infer_node_indexes_.push_back(next_nodes_indexes[index]);
      }
    } else if (std::all_of(next_node->inputIndex.begin(), next_node->inputIndex.end(),
                           [&](uint32_t i) { return tensors_[i].is_inferred_; }) ||
               std::any_of(next_node->inputIndex.begin(), next_node->inputIndex.end(),
                           [&](uint32_t i) { return graph->allTensors.at(i)->dataType == kObjectTypeTensorType; })) {
      infer_node_indexes_.push_back(next_nodes_indexes[index]);
    }
  }
}

}  // namespace lite
}  // namespace mindspore

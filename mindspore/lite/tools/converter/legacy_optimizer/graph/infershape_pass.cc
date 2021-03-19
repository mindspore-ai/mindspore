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

using mindspore::lite::Tensor;
namespace mindspore {
namespace lite {
namespace {
constexpr int DEFAULT_DIM_VALUE = -1;
constexpr size_t INITIAL_SIZE = 1024;

void FreeTensors(std::vector<Tensor *> input_tensors, std::vector<Tensor *> output_tensors) {
  for (auto &tensor : input_tensors) {
    delete tensor;
    tensor = nullptr;
  }
  for (auto &tensor : output_tensors) {
    delete tensor;
    tensor = nullptr;
  }
  input_tensors.clear();
  input_tensors.shrink_to_fit();
  output_tensors.clear();
  output_tensors.shrink_to_fit();
}

std::vector<Tensor *> ConvertTensorToLiteTensor(MetaGraphT *graph, const std::vector<uint32_t> &tensor_indexs,
                                                const schema::PrimitiveType node_type) {
  MS_ASSERT(graph != nullptr);
  std::vector<Tensor *> lite_tensors;
  bool convert_succ = true;
  for (size_t i = 0; i < tensor_indexs.size(); i++) {
    std::unique_ptr<Tensor> lite_tensor = nullptr;
    auto &tensorT = graph->allTensors.at(tensor_indexs[i]);
    if (tensorT->dataType != kObjectTypeTensorType) {  // convert to lite::Tensor
      auto tensor_shape = tensorT->dims;
      lite_tensor = std::make_unique<Tensor>(
        TypeId(tensorT->dataType), tensor_shape, tensorT->format,
        TensorCategory(tensorT->nodeType, tensorT->dims.size(), TypeId(tensorT->dataType), tensorT->data.size()));
      if (lite_tensor == nullptr) {
        MS_LOG(ERROR) << "lite tensor is nullptr";
        convert_succ = false;
        break;
      }
      auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
      // when tensorT as param input
      if (lite_tensor_size == 0) {
        lite_tensors.emplace_back(lite_tensor.release());
        continue;
      }
      auto ret = lite_tensor->MallocData();
      if (ret != 0) {
        MS_LOG(ERROR) << "Malloc tensor data failed";
        convert_succ = false;
        break;
      }
      if (memcpy_s(lite_tensor->MutableData(), lite_tensor->Size(), tensorT->data.data(), lite_tensor_size) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        convert_succ = false;
        break;
      }
    } else {  // convert to lite::TensorList
      auto tensor_shape = tensorT->dims;
      TypeId type = kTypeUnknown;
      std::vector<int> element_shape;
      if (!tensorT->data.empty()) {
        int *data = reinterpret_cast<int *>(tensorT->data.data());
        type = TypeId(data[0]);
        if (tensorT->data.size() < 8 || (data[1] + 2) * 4 != static_cast<int>(tensorT->data.size())) {
          MS_LOG(ERROR) << "tensorlist data length illegal";
          convert_succ = false;
          break;
        }
        for (int j = 0; j < data[1]; ++j) {
          element_shape.push_back(data[j + 2]);
        }
      }
      lite_tensor = std::make_unique<TensorList>(tensor_shape, element_shape);
      if (lite_tensor == nullptr) {
        MS_LOG(ERROR) << "lite tensorlist is nullptr";
        convert_succ = false;
        break;
      }
      reinterpret_cast<TensorList *>(lite_tensor.get())->set_tensors_data_type(type);
    }
    lite_tensors.emplace_back(lite_tensor.release());
  }
  if (!convert_succ) {
    FreeTensors(lite_tensors, {});
    return {};
  }
  return lite_tensors;
}

STATUS NodeInferShape(const std::unique_ptr<schema::CNodeT> &node, const std::vector<Tensor *> &inputs,
                      std::vector<Tensor *> *outputs) {
  flatbuffers::FlatBufferBuilder fbb(INITIAL_SIZE);
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
  parameter->infer_flag_ = true;
  auto ret = KernelInferShape(inputs, outputs, parameter);
  fbb.Clear();
  free(parameter);
  return ret;
}

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
}  // namespace

STATUS InferShapePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto idx : graph->inputIndex) {
    auto input_tensor = graph->allTensors[idx].get();
    for (auto &dim : input_tensor->dims) {
      if (dim == 0) {
        MS_LOG(WARNING) << "One dimension of the input shape is 0, which would be set to -1 as a default value.";
        dim = DEFAULT_DIM_VALUE;
      }
    }
  }
  for (auto g_input_idx : graph->inputIndex) {
    auto g_input_shape = graph->allTensors.at(g_input_idx)->dims;
    if (std::find(g_input_shape.begin(), g_input_shape.end(), -1) != g_input_shape.end()) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime";
      return RET_OK;
    }
  }
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    auto input_tensors = ConvertTensorToLiteTensor(graph, node->inputIndex, node->primitive->value.type);
    std::vector<Tensor *> output_tensors;
    if (input_tensors.empty() || input_tensors.size() != node->inputIndex.size()) {
      MS_LOG(ERROR) << "convert input lite tensor error";
      FreeTensors(input_tensors, output_tensors);
      return RET_INFER_ERR;
    }
    output_tensors = ConvertTensorToLiteTensor(graph, node->outputIndex, node->primitive->value.type);
    if (output_tensors.empty() || output_tensors.size() != node->outputIndex.size()) {
      MS_LOG(ERROR) << "convert output lite tensor error";
      FreeTensors(input_tensors, output_tensors);
      return RET_INFER_ERR;
    }
    auto status = NodeInferShape(node, input_tensors, &output_tensors);
    MS_LOG(DEBUG) << "cur node:" << node->name;
    if (status == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << node->name
                   << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type) << "flag set to false.";
      FreeTensors(input_tensors, output_tensors);
      return RET_INFER_INVALID;
    } else if (status != RET_OK) {
      MS_LOG(WARNING) << "InferShape failed, name: " << node->name
                      << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type);
      FreeTensors(input_tensors, output_tensors);
      return RET_INFER_ERR;
    }
    PrintTensorShape(input_tensors, output_tensors);
    // copy output shape to tensorT
    for (size_t i = 0; i < output_tensors.size(); i++) {
      auto output_dims = output_tensors[i]->shape();
      auto &output_tensor = graph->allTensors.at(node->outputIndex[i]);
      output_tensor->dims.swap(output_dims);
      output_tensor->format = output_tensors[i]->format();
      output_tensor->dataType = output_tensors[i]->data_type();
    }
    FreeTensors(input_tensors, output_tensors);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

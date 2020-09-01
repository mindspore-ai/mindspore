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

#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include <vector>
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "src/ir/tensor.h"
#include "src/ops/primitive_c.h"

using mindspore::lite::tensor::Tensor;
using mindspore::lite::PrimitiveC;
namespace mindspore {
namespace lite {
namespace {
std::vector<tensor::Tensor *> ConvertTensorToLiteTensor(MetaGraphT *graph, const std::vector<uint32_t> &tensor_indexs,
                                                        const schema::PrimitiveType node_type) {
  std::vector<tensor::Tensor *> lite_tensors;
  for (size_t i = 0; i < tensor_indexs.size(); i++) {
    auto &tensorT = graph->allTensors.at(tensor_indexs[i]);
    auto tensor_shape = tensorT->dims;
    auto lite_tensor =
        new(std::nothrow) tensor::Tensor(TypeId(tensorT->dataType), tensor_shape, tensorT->format, tensorT->nodeType);
    if (lite_tensor == nullptr) {
      MS_LOG(ERROR) << "lite tensor is nullptr";
      return std::vector<tensor::Tensor *>();
    }
    // reshape op must get tensor data to infershape
    if (node_type == schema::PrimitiveType_Reshape && i == 1 && tensorT->nodeType == NodeType_ValueNode) {
      auto lite_tensor_size = tensorT->data.size() * sizeof(uint8_t);
      // when tensorT as param input
      if (lite_tensor_size == 0) {
        delete lite_tensor;
        return std::vector<tensor::Tensor *>();
      }
      auto tensor_data = new(std::nothrow) char[lite_tensor_size / sizeof(char)];
      if (tensor_data == nullptr) {
        MS_LOG(ERROR) << "tensor_data is nullptr";
        delete lite_tensor;
        return std::vector<tensor::Tensor *>();
      }
      auto ret = memcpy_s(tensor_data, lite_tensor_size, tensorT->data.data(), lite_tensor_size);
      if (ret != EOK) {
        delete lite_tensor;
        delete[] tensor_data;
        MS_LOG(ERROR) << "memcpy error: " << ret;
        return std::vector<tensor::Tensor *>();
      }
      lite_tensor->SetData(tensor_data);
      lite_tensors.emplace_back(lite_tensor);
      continue;
    }
    lite_tensors.emplace_back(lite_tensor);
  }
  return lite_tensors;
}
}  // namespace
STATUS InferShapePass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    auto input_tensors = ConvertTensorToLiteTensor(graph, node->inputIndex, node->primitive->value.type);
    if (input_tensors.empty() || input_tensors.size() != node->inputIndex.size()) {
      MS_LOG(ERROR) << "convert input lite tensor error";
      return RET_INFER_ERR;
    }
    auto output_tensors = ConvertTensorToLiteTensor(graph, node->outputIndex, node->primitive->value.type);
    if (output_tensors.empty() || output_tensors.size() != node->outputIndex.size()) {
      MS_LOG(ERROR) << "convert output lite tensor error";
      return RET_INFER_ERR;
    }
    std::unique_ptr<PrimitiveT> primitiveT(new (std::nothrow) PrimitiveT(*node->primitive));
    if (primitiveT == nullptr) {
      MS_LOG(ERROR) << "copy primitiveT error";
      return RET_ERROR;
    }
    auto primitiveC = std::shared_ptr<PrimitiveC>(PrimitiveC::UnPackFromSchemaPrimitiveT(primitiveT.release()));
    if (primitiveC == nullptr) {
      MS_LOG(ERROR) << "unpack primitiveT error";
      return RET_ERROR;
    }
    auto ret = primitiveC->InferShape(input_tensors, output_tensors);
    if (ret == RET_INFER_INVALID) {
      MS_LOG(INFO) << "InferShape shouldn't be done before runtime, name: " << node->name
                   << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type)
                   << "flag set to false.";
    } else if (ret != RET_OK) {
      MS_LOG(WARNING) << "InferShape failed, name: " << node->name
                    << ", type: " << schema::EnumNamePrimitiveType(node->primitive->value.type);
      return RET_INFER_ERR;
    }
    // copy output shape to tensorT
    for (size_t i = 0; i < output_tensors.size(); i++) {
      auto output_dims = output_tensors[i]->shape();
      auto &output_tensor =  graph->allTensors.at(node->outputIndex[i]);
      output_tensor->dims.swap(output_dims);
      output_tensor->format = output_tensors[i]->GetFormat();
      output_tensor->dataType = output_tensors[i]->data_type();
    }
    for (auto input_tensor : input_tensors) {
      delete input_tensor;
    }
    for (auto output_tensor : output_tensors) {
      delete output_tensor;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

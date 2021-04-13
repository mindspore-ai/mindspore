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

#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
#include <string>
#include <set>
#include <vector>
#include <unordered_map>
#include "tools/common/node_util.h"
#include "tools/converter/converter_context.h"
#include "src/common/common.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
#define kMinInputNum 1
#define kOutputNum 1

STATUS DTypeTransPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);

  auto status = DoModelInputDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelInputDTypeTrans error: " << status;
    return status;
  }

  status = DoNodeInoutDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoNodeInoutDTypeTrans error: " << status;
    return status;
  }

  status = DoModelOutputDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelOutputDTypeTrans error: " << status;
    return status;
  }

  return RET_OK;
}

STATUS DTypeTransPass::DoModelInputDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  auto &graph_in_idxes = graph->inputIndex;
  if (this->input_data_dtype != TypeId::kNumberTypeFloat32 && this->input_data_dtype != TypeId::kNumberTypeUInt8 &&
      this->input_data_dtype != TypeId::kNumberTypeInt8 && this->input_data_dtype != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid inputDataType: " << this->input_data_dtype;
    return RET_ERROR;
  }
  for (auto graph_in_idx : graph_in_idxes) {
    MS_ASSERT(graph_in_idx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graph_in_idx);
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      continue;
    }
    int32_t tensor_data_type = this->input_data_dtype != TypeId::kTypeUnknown
                                 ? this->input_data_dtype
                                 : TensorDataType::GetInstance()->GetTensorType(graph_in_idx);
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      auto node_name = (*iter)->name;
      for (size_t input_indexidx = 0; input_indexidx < (*iter)->inputIndex.size(); input_indexidx++) {
        if ((*iter)->inputIndex.at(input_indexidx) == graph_in_idx) {
          STATUS status = RET_OK;

          // insert dtype cast node between input tensor and input node
          if (tensor_data_type != tensor->dataType && tensor_data_type != kTypeUnknown) {
            iter =
              InsertDTypeTransNode(graph, iter, kBefore, input_indexidx, tensor_data_type, tensor->dataType, &status);
          }

          if (status != RET_OK) {
            MS_LOG(ERROR) << "InsertDTypeTransNode before " << node_name.c_str() << " failed";
            return status;
          }
        }
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::DoModelOutputDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  if (this->output_data_dtype != TypeId::kNumberTypeFloat32 && this->output_data_dtype != TypeId::kNumberTypeUInt8 &&
      this->output_data_dtype != TypeId::kNumberTypeInt8 && this->output_data_dtype != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid outputDataType: " << this->output_data_dtype;
    return RET_ERROR;
  }
  auto &graph_out_idxes = graph->outputIndex;
  for (auto graph_out_idx : graph_out_idxes) {
    MS_ASSERT(graph_out_idx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graph_out_idx);
    if (tensor->quantParams.empty() || !tensor->quantParams.front()->inited) {
      continue;
    }
    int32_t tensor_data_type = this->output_data_dtype != TypeId::kTypeUnknown
                                 ? this->output_data_dtype
                                 : TensorDataType::GetInstance()->GetTensorType(graph_out_idx);
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      auto node_name = (*iter)->name;
      MS_ASSERT(node != nullptr);
      for (size_t outputIndexIdx = 0; outputIndexIdx < (*iter)->outputIndex.size(); outputIndexIdx++) {
        if ((*iter)->outputIndex.at(outputIndexIdx) == graph_out_idx) {
          // insert transNode
          STATUS status = RET_OK;
          if (tensor_data_type != tensor->dataType && tensor_data_type != kTypeUnknown) {
            iter =
              InsertDTypeTransNode(graph, iter, kAfter, outputIndexIdx, tensor->dataType, tensor_data_type, &status);
          }
          if (status != RET_OK) {
            MS_LOG(ERROR) << "InsertDTypeTransNode after " << node_name.c_str() << " failed";
            return status;
          }
          break;
        }
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::InsetDTypeTransNodeForWrongDtypeQuantOp(schema::MetaGraphT *graph, NodeIter *iter) {
  auto node_name = (**iter)->name;
  auto status = RET_OK;
  // insert fp32 to int8 before
  for (size_t i = 0; i < (**iter)->inputIndex.size(); i++) {
    auto &pre_tensor = graph->allTensors.at((**iter)->inputIndex.at(i));
    if (pre_tensor->dataType == kNumberTypeFloat32 && !pre_tensor->quantParams.empty() &&
        pre_tensor->quantParams.front()->inited) {
      *iter = InsertDTypeTransNode(graph, *iter, kBefore, i, kNumberTypeFloat32, kNumberTypeInt8, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertFloat32ToInt8Node before " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }

  // insert int8 to fp32 after
  for (size_t i = 0; i < (**iter)->outputIndex.size(); i++) {
    auto &post_tensor = graph->allTensors.at((**iter)->outputIndex.at(i));
    if (post_tensor->dataType == kNumberTypeFloat32 && !post_tensor->quantParams.empty() &&
        post_tensor->quantParams.front()->inited) {
      *iter = InsertDTypeTransNode(graph, *iter, kAfter, i, kNumberTypeInt8, kNumberTypeFloat32, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertInt8ToFloat32Node before " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::InsetDTypeTransNodeForUnsupportedInt8Op(schema::MetaGraphT *graph, NodeIter *iter) {
  auto node_name = (**iter)->name;
  auto status = RET_OK;
  // insert int8 to fp32 before
  for (size_t i = 0; i < (**iter)->inputIndex.size(); i++) {
    auto &pre_tensor = graph->allTensors.at((**iter)->inputIndex.at(i));
    if (pre_tensor->dataType == kNumberTypeInt8 && !pre_tensor->quantParams.empty() &&
        pre_tensor->quantParams.front()->inited) {
      *iter = InsertDTypeTransNode(graph, *iter, kBefore, i, kNumberTypeInt8, kNumberTypeFloat32, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertInt8ToFloat32Node before " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }

  // insert fp32 to int8 after
  for (size_t i = 0; i < (**iter)->outputIndex.size(); i++) {
    auto &post_tensor = graph->allTensors.at((**iter)->outputIndex.at(i));
    if (post_tensor->dataType == kNumberTypeInt8 && !post_tensor->quantParams.empty() &&
        post_tensor->quantParams.front()->inited) {
      *iter = InsertDTypeTransNode(graph, *iter, kAfter, i, kNumberTypeInt8, kNumberTypeFloat32, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertFloat32ToInt8Node before " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::DoNodeInoutDTypeTrans(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto node_name = (*iter)->name;
    if ((*iter)->inputIndex.empty()) {
      MS_LOG(ERROR) << "Op " << node_name.c_str() << " should have " << kMinInputNum << " input tensor at least";
      return RET_ERROR;
    }

    if ((*iter)->primitive->value.type == schema::PrimitiveType_QuantDTypeCast ||
        (*iter)->primitive->value.type == schema::PrimitiveType_Cast) {
      continue;
    }

    STATUS status = RET_OK;
    // quant_type is quant_all, but inputs/outputs are float32
    if ((*iter)->quantType == QuantType_QUANT_ALL) {
      status = InsetDTypeTransNodeForWrongDtypeQuantOp(graph, &iter);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertFloat32ToInt8Node before " << node_name.c_str() << " failed";
        return status;
      }
      continue;
    }

    // quant_type is quant_none, but inputs/outputs have quant params and dtype is int8, which means this int8 op is not
    // supported yet
    if ((*iter)->quantType == QuantType_QUANT_NONE) {
      status = InsetDTypeTransNodeForUnsupportedInt8Op(graph, &iter);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "InsertFloat32ToInt8Node before " << node_name.c_str() << " failed";
        return status;
      }
    }
  }
  return RET_OK;
}

NodeIter DTypeTransPass::InsertDTypeTransNode(schema::MetaGraphT *graph, NodeIter exist_node_iter, InsertPlace place,
                                              size_t inout_idx, int32_t input_data_type, int32_t output_data_type,
                                              STATUS *error_code) {
  MS_ASSERT((*exist_node_iter) != nullptr);
  auto exist_node_name = (*exist_node_iter)->name;
  std::string tile_name;
  if (place == kBefore) {
    tile_name = exist_node_name + "_pre";
  } else {
    tile_name = exist_node_name + "_post";
  }
  auto trans_node = std::unique_ptr<CNodeT>(new (std::nothrow) CNodeT);
  if (trans_node == nullptr) {
    MS_LOG(ERROR) << "new TransNode failed";
    *error_code = RET_ERROR;
    return graph->nodes.end();
  }
  auto quant_dtype_cast_param = new (std::nothrow) QuantDTypeCastT;
  if (quant_dtype_cast_param == nullptr) {
    MS_LOG(ERROR) << "new quantDTypeCastParam failed";
    *error_code = RET_ERROR;
    return graph->nodes.end();
  }
  trans_node->primitive = std::make_unique<schema::PrimitiveT>();
  trans_node->primitive->value.value = quant_dtype_cast_param;
  trans_node->primitive->value.type = PrimitiveType_QuantDTypeCast;
  trans_node->quantType = QuantType_AwareTraining;
  quant_dtype_cast_param->src_t = input_data_type;
  quant_dtype_cast_param->dst_t = output_data_type;
  if (input_data_type == TypeId::kNumberTypeInt8 && output_data_type == TypeId::kNumberTypeFloat32) {
    trans_node->name = "int8toft32_" + tile_name + std::to_string(id_++);
  } else if (input_data_type == TypeId::kNumberTypeFloat32 && output_data_type == TypeId::kNumberTypeInt8) {
    trans_node->name = "ft32toint8_" + tile_name + std::to_string(id_++);
  } else if (input_data_type == TypeId::kNumberTypeUInt8 && output_data_type == TypeId::kNumberTypeInt8) {
    trans_node->name = "uint8toint8_" + tile_name + std::to_string(id_++);
  } else if (input_data_type == TypeId::kNumberTypeInt8 && output_data_type == TypeId::kNumberTypeUInt8) {
    trans_node->name = "int8touint8_" + tile_name + std::to_string(id_++);
  }
  trans_node->primitive->value.value = quant_dtype_cast_param;
  int insert_num = 0;
  return InsertNode(graph, exist_node_iter, place, inout_idx, std::move(trans_node), error_code, &insert_num,
                    castOpCopyer);
}

void DTypeTransPass::set_input_data_dtype(TypeId data_type) { this->input_data_dtype = data_type; }

void DTypeTransPass::set_output_data_dtype(TypeId data_type) { this->output_data_dtype = data_type; }

}  // namespace lite
}  // namespace mindspore

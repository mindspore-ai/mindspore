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

#define USE_DEPRECATED_API
#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
#include <string>
#include <vector>
#include "tools/common/node_util.h"
#include "tools/converter/converter_context.h"
#include "src/common/utils.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
constexpr int kMinInputNum = 1;

STATUS DTypeTransPass::Run(schema::MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);

  auto status = DoModelInputDTypeTrans(graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoModelInputDTypeTrans error: " << status;
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
  if ((this->input_data_dtype == TypeId::kNumberTypeInt32) || (this->input_data_dtype == TypeId::kNumberTypeInt64)) {
    MS_LOG(INFO) << "Unsupported inputDataType: " << this->input_data_dtype;
    return RET_OK;
  }
  if (this->input_data_dtype != TypeId::kNumberTypeFloat32 && this->input_data_dtype != TypeId::kNumberTypeUInt8 &&
      this->input_data_dtype != TypeId::kNumberTypeInt8 && this->input_data_dtype != TypeId::kTypeUnknown) {
    MS_LOG(ERROR) << "Invalid inputDataType: " << this->input_data_dtype;
    return RET_ERROR;
  }
  for (size_t i = 0; i < graph_in_idxes.size(); i++) {
    size_t graph_in_idx = graph_in_idxes.at(i);
    MS_ASSERT(graph_in_idx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graph_in_idx);
    CHECK_NULL_RETURN(tensor);
    if (!TensorQuantParamsInited(*tensor)) {
      continue;
    }

    if (this->input_data_dtype == TypeId::kTypeUnknown) {
      auto origin_input_dtype = ConverterInnerContext::GetInstance()->GetGraphInputDType(static_cast<int32_t>(i));
      if (origin_input_dtype != kTypeUnknown && tensor->dataType != origin_input_dtype) {
        MS_LOG(ERROR) << "Change graph input dtype is not allowed.";
        return RET_ERROR;
      }
      continue;
    }

    int32_t tensor_data_type = this->input_data_dtype;
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      CHECK_NULL_RETURN(*iter);
      auto node_name = (*iter)->name;
      for (size_t input_indexidx = 0; input_indexidx < (*iter)->inputIndex.size(); input_indexidx++) {
        if ((*iter)->inputIndex.at(input_indexidx) == graph_in_idx) {
          STATUS status = RET_OK;

          // insert dtype cast node between input tensor and input node
          if (tensor_data_type != tensor->dataType) {
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
  for (size_t i = 0; i < graph_out_idxes.size(); i++) {
    size_t graph_out_idx = graph_out_idxes.at(i);
    MS_ASSERT(graph_out_idx < graph->allTensors.size());
    auto &tensor = graph->allTensors.at(graph_out_idx);
    CHECK_NULL_RETURN(tensor);
    if (!TensorQuantParamsInited(*tensor)) {
      continue;
    }

    if (this->output_data_dtype == TypeId::kTypeUnknown) {
      auto origin_output_dtype = ConverterInnerContext::GetInstance()->GetGraphOutputDType(static_cast<int32_t>(i));
      if (origin_output_dtype != kTypeUnknown && tensor->dataType != origin_output_dtype) {
        MS_LOG(ERROR) << "Change graph output dtype is not allowed.";
        return RET_ERROR;
      }
      continue;
    }

    int32_t tensor_data_type = this->output_data_dtype;
    for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
      CHECK_NULL_RETURN(*iter);
      auto node_name = (*iter)->name;
      MS_ASSERT(node != nullptr);
      for (size_t outputIndexIdx = 0; outputIndexIdx < (*iter)->outputIndex.size(); outputIndexIdx++) {
        if ((*iter)->outputIndex.at(outputIndexIdx) == graph_out_idx) {
          // insert transNode
          STATUS status = RET_OK;
          if (tensor_data_type != tensor->dataType) {
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
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(iter);
  auto node_name = (**iter)->name;
  auto status = RET_OK;
  // insert fp32/uint8 to int8 before
  for (size_t i = 0; i < (**iter)->inputIndex.size(); i++) {
    auto &pre_tensor = graph->allTensors.at((**iter)->inputIndex.at(i));
    CHECK_NULL_RETURN(pre_tensor);
    // insert quant cast op for tensor which should be int8
    if ((pre_tensor->dataType == kNumberTypeFloat32 || pre_tensor->dataType == kNumberTypeUInt8) &&
        TensorQuantParamsInited(*pre_tensor)) {
      if (!pre_tensor->data.empty()) {
        MS_LOG(ERROR) << "tensor with float data should be quantized at tensor_quant_pass.";
        return RET_ERROR;
      }
      *iter = InsertDTypeTransNode(graph, *iter, kBefore, i, pre_tensor->dataType, kNumberTypeInt8, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Insert float32 or uint8 to int8 node after before " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }

  // insert int8 to fp32/uint8 after
  for (size_t i = 0; i < (**iter)->outputIndex.size(); i++) {
    auto &post_tensor = graph->allTensors.at((**iter)->outputIndex.at(i));
    // insert quant cast op for tensor which should be int8
    // e.g: reshape's shape tensor don't need insert quant op so its quant param isn't inited
    if ((post_tensor->dataType == kNumberTypeFloat32 || post_tensor->dataType == kNumberTypeUInt8) &&
        TensorQuantParamsInited(*post_tensor)) {
      *iter = InsertDTypeTransNode(graph, *iter, kAfter, i, kNumberTypeInt8, post_tensor->dataType, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Insert int8 to float32 or uint8 node after " << node_name.c_str() << " failed";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS DTypeTransPass::InsetDTypeTransNodeForUnsupportedInt8Op(schema::MetaGraphT *graph, NodeIter *iter) {
  CHECK_NULL_RETURN(graph);
  CHECK_NULL_RETURN(iter);
  auto node_name = (**iter)->name;
  auto status = RET_OK;
  // insert int8 to fp32 before
  for (size_t i = 0; i < (**iter)->inputIndex.size(); i++) {
    auto &pre_tensor = graph->allTensors.at((**iter)->inputIndex.at(i));
    if (pre_tensor->dataType == kNumberTypeInt8 && TensorQuantParamsInited(*pre_tensor)) {
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
    CHECK_NULL_RETURN(post_tensor);
    if (post_tensor->dataType == kNumberTypeInt8 && TensorQuantParamsInited(*post_tensor)) {
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
    CHECK_NULL_RETURN(*iter);
    auto node_name = (*iter)->name;
    if ((*iter)->inputIndex.empty()) {
      MS_LOG(WARNING) << "Op " << node_name.c_str() << " should have " << kMinInputNum << " input tensor at least";
      continue;
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
  auto trans_node = std::make_unique<CNodeT>();
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
  } else if (input_data_type == TypeId::kNumberTypeUInt8 && output_data_type == TypeId::kNumberTypeFloat32) {
    trans_node->name = "uint8toft32_" + tile_name + std::to_string(id_++);
  } else if (input_data_type == TypeId::kNumberTypeFloat32 && output_data_type == TypeId::kNumberTypeUInt8) {
    trans_node->name = "ft32touint8_" + tile_name + std::to_string(id_++);
  }
  trans_node->primitive->value.value = quant_dtype_cast_param;
  int insert_num = 0;
  return InsertNode(graph, exist_node_iter, place, inout_idx, std::move(trans_node), error_code, &insert_num,
                    castOpCopyer);
}
}  // namespace lite
}  // namespace mindspore

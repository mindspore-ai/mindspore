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

#include <string>
#include <memory>
#include <utility>
#include "tools/converter/legacy_optimizer/graph/eltwise_format_trans_pass.h"
#include "tools/common/converter_op_utils.h"
#include "tools/common/node_util.h"
#include "utils/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {

STATUS EltwiseFormatTransPass::Run(schema::MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto iter = graph->nodes.begin(); iter != graph->nodes.end(); iter++) {
    auto &node = *iter;
    if (node->primitive->value.type != PrimitiveType_Eltwise) {
      continue;
    }
    auto node_name = node->name;
    auto input_node_indexes = GetInputNodeIdx(*graph, *node);
    auto pre_type = schema::PrimitiveType_NONE;
    size_t has_trans_count = 0;
    auto can_fusion = true;
    for (auto input_node_index : input_node_indexes) {
      MS_ASSERT(graph->nodes.size() > input_node_index);
      auto &pre_node = graph->nodes.at(input_node_index);
      MS_ASSERT(pre_node != nullptr);
      if (pre_type == schema::PrimitiveType_NONE) {
        if (pre_node->primitive->value.type == schema::PrimitiveType_Nchw2Nhwc ||
            pre_node->primitive->value.type == schema::PrimitiveType_Nhwc2Nchw) {
          pre_type = pre_node->primitive->value.type;
          has_trans_count++;
        }
      } else {
        if (pre_node->primitive->value.type == schema::PrimitiveType_Nchw2Nhwc ||
            pre_node->primitive->value.type == schema::PrimitiveType_Nhwc2Nchw) {
          if (pre_type != pre_node->primitive->value.type) {
            can_fusion = false;
            break;
          } else {
            has_trans_count++;
          }
        }
      }
    }
    if (!can_fusion) {
      continue;
    }
    auto output_node_indexes = GetOutputNodeIdx(*graph, *node);
    auto post_type = schema::PrimitiveType_NONE;
    for (auto output_node_index : output_node_indexes) {
      MS_ASSERT(graph->nodes.size() > output_node_index);
      auto &post_node = graph->nodes.at(output_node_index);
      MS_ASSERT(post_node != nullptr);
      if (post_type == schema::PrimitiveType_NONE) {
        if (post_node->primitive->value.type == schema::PrimitiveType_Nchw2Nhwc ||
            post_node->primitive->value.type == schema::PrimitiveType_Nhwc2Nchw) {
          post_type = post_node->primitive->value.type;
          has_trans_count++;
        }
      } else {
        if (post_node->primitive->value.type == schema::PrimitiveType_Nchw2Nhwc ||
            post_node->primitive->value.type == schema::PrimitiveType_Nhwc2Nchw) {
          if (post_type != post_node->primitive->value.type) {
            can_fusion = false;
            break;
          } else {
            has_trans_count++;
          }
        }
      }
    }
    if (!can_fusion) {
      continue;
    }
    auto total_node_count = input_node_indexes.size() + output_node_indexes.size();
    size_t half_count = total_node_count / 2;
    if (total_node_count % 2 == 0) {
      can_fusion = has_trans_count > half_count;
    } else {
      can_fusion = has_trans_count >= half_count;
    }
    if (!can_fusion) {
      continue;
    }
    FormatTransNodeType pre_insert_trans_type = kNHWC2NCHW;
    FormatTransNodeType post_insert_trans_type = kNHWC2NCHW;
    if (pre_type == PrimitiveType_NONE && post_type != PrimitiveType_NONE) {
      pre_insert_trans_type = post_type == schema::PrimitiveType_Nhwc2Nchw ? kNHWC2NCHW : kNCHW2NHWC;
      post_insert_trans_type = post_type == schema::PrimitiveType_Nhwc2Nchw ? kNCHW2NHWC : kNHWC2NCHW;
    } else if (pre_type != PrimitiveType_NONE && post_type == PrimitiveType_NONE) {
      pre_insert_trans_type = pre_type == schema::PrimitiveType_Nhwc2Nchw ? kNCHW2NHWC : kNHWC2NCHW;
      post_insert_trans_type = pre_type == schema::PrimitiveType_Nhwc2Nchw ? kNHWC2NCHW : kNCHW2NHWC;
    } else if (pre_type == PrimitiveType_NONE && post_type == PrimitiveType_NONE) {
      continue;
    } else {
      if (pre_type == post_type) {
        MS_LOG(ERROR) << "Unknow error";
        return RET_ERROR;
      }
      pre_insert_trans_type = pre_type == schema::PrimitiveType_Nhwc2Nchw ? kNCHW2NHWC : kNHWC2NCHW;
      post_insert_trans_type = post_type == schema::PrimitiveType_Nhwc2Nchw ? kNCHW2NHWC : kNHWC2NCHW;
    }

    STATUS status = RET_OK;
    auto input_tensor_size = (*iter)->inputIndex.size();
    for (auto i = 0; i < input_tensor_size; i++) {
      iter = InsertFormatTransNode(graph, iter, kBefore, i, pre_insert_trans_type, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Insert" << pre_insert_trans_type << "before " << (*iter)->name << " failed";
        return status;
      }
    }
    auto output_tensor_size = (*iter)->outputIndex.size();
    for (auto i = 0; i < output_tensor_size; i++) {
      iter = InsertFormatTransNode(graph, iter, kAfter, i, post_insert_trans_type, &status);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Insert" << post_insert_trans_type << "Node before " << (*iter)->name << " failed";
        return status;
      }
    }
  }
  return RET_OK;
}

NodeIter EltwiseFormatTransPass::InsertFormatTransNode(schema::MetaGraphT *graph, NodeIter existNodeIter,
                                                       InsertPlace place, size_t inoutIdx, FormatTransNodeType nodeType,
                                                       STATUS *errorCode) {
  MS_ASSERT((*existNodeIter) != nullptr);
  auto existNodeName = (*existNodeIter)->name;
  std::string tileName;
  if (place == kBefore) {
    tileName = existNodeName + "_pre";
  } else {
    tileName = existNodeName + "_post";
  }
  auto transNode = std::make_unique<schema::CNodeT>();
  transNode->primitive = std::make_unique<schema::PrimitiveT>();

  if (nodeType == kNCHW2NHWC) {
    transNode->name = "nchw2nhwc_" + tileName + std::to_string(id++);
    transNode->primitive->value.type = schema::PrimitiveType_Nchw2Nhwc;
  } else {
    transNode->name = "nhwc2nchw_" + tileName + std::to_string(id++);
    transNode->primitive->value.type = schema::PrimitiveType_Nhwc2Nchw;
  }
  return InsertNode(graph, existNodeIter, place, inoutIdx, std::move(transNode), errorCode);
}

void EltwiseFormatTransPass::SetQuantType(QuantType quantType) { this->quantType = quantType; }

void EltwiseFormatTransPass::SetFmk(converter::FmkType fmkType) { this->fmkType = fmkType; }
}  // namespace lite
}  // namespace mindspore

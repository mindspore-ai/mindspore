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

#include "tools/common/node_util.h"
#include <memory>
#include <vector>
#include "src/common/common.h"
#include "utils/log_adapter.h"
#include "tools/common/graph_util.h"
#include "tools/common/tensor_util.h"

namespace mindspore {
namespace lite {
STATUS BroadCastQuantParam(schema::MetaGraphT *graphT, const std::unique_ptr<CNodeT> &node) {
  MS_ASSERT(graphT != nullptr);
  MS_ASSERT(node != nullptr);
  // set quantParam to preNode
  for (size_t i = 0; i < node->inputIndex.size(); i++) {
    auto preNodeIdexes = GetInputNodeIdx(*graphT, *(node.get()), i);
    for (auto preNodeIdx : preNodeIdexes) {
      MS_ASSERT(graphT->nodes.size() > preNodeIdx);
      auto &preNode = graphT->nodes.at(preNodeIdx);
      MS_ASSERT(preNode != nullptr);
      // if preNode is not init, it maybe not a quantNode, so skip
      //      if (preNode->inputIndex.size() + preNode->outputIndex.size() != preNode->quantParam.size()) {
      //        continue;
      //      }
      auto preNodeOutputIndexes = preNode->outputIndex;
      int32_t currentNodeIndexInPre = -1;
      for (auto index : preNodeOutputIndexes) {
        currentNodeIndexInPre++;
        if (index == node->inputIndex.at(i)) {
          break;
        }
      }
      MS_ASSERT(currentNodeIndexInPre != -1);
      MS_ASSERT(node->quantParam.size() > i);
      MS_ASSERT(node->quantParam.at(i) != nullptr);
      //      auto quantParamArrayCopy = CopyQuantParamArrayT(node->quantParam.at(i));
      //      if (quantParamArrayCopy == nullptr) {
      //        //MS_LOG(ERROR)("CopyQuantParamArray return nullptr, node: %s", node->name.c_str());
      //        return RET_ERROR;
      //      }
      //      preNode->quantParam.at(preNode->inputIndex.size() + currentNodeIndexInPre) =
      //        std::move(CopyQuantParamArrayT(quantParamArrayCopy));
    }
  }

  // set quantParam to postNode
  for (size_t i = 0; i < node->outputIndex.size(); i++) {
    auto postNodeIdexes = GetOutputNodeIdx(*graphT, *(node.get()), i);
    for (auto postNodeIdx : postNodeIdexes) {
      MS_ASSERT(graphT->nodes.size() > postNodeIdx);
      auto &postNode = graphT->nodes.at(postNodeIdx);
      MS_ASSERT(postNode != nullptr);
      // if postNode is not init, it maybe not a quantNode, so skip
      //      if (postNode->inputIndex.size() + postNode->outputIndex.size() != postNode->quantParam.size()) {
      //        continue;
      //      }
      auto postNodeInputIndexes = postNode->inputIndex;
      int32_t currentNodeIndexInPost = -1;
      for (auto index : postNodeInputIndexes) {
        currentNodeIndexInPost++;
        if (index == node->outputIndex.at(i)) {
          break;
        }
      }
      MS_ASSERT(currentNodeIndexInPost != -1);
      MS_ASSERT(node->quantParam.size() > node->inputIndex.size() + i);
      MS_ASSERT(node->quantParam.at(node->inputIndex.size() + i) != nullptr);
      //      auto quantParamArrayCopy = CopyQuantParamArrayT(node->quantParam.at(node->inputIndex.size() + i));
      //      if (quantParamArrayCopy == nullptr) {
      //        //MS_LOG(ERROR)("CopyQuantParamArray return nullptr, node: %s", node->name.c_str());
      //        return RET_ERROR;
      //      }
      //      postNode->quantParam.at(currentNodeIndexInPost) = std::move(CopyQuantParamArrayT(quantParamArrayCopy));
    }
  }
  return RET_OK;
}

static const std::vector<schema::PrimitiveType> nhwcOpList = {
  schema::PrimitiveType_Conv2D,          schema::PrimitiveType_DeConv2D,
  schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_DeDepthwiseConv2D,
  schema::PrimitiveType_Pooling,         schema::PrimitiveType_Resize,
  schema::PrimitiveType_BatchNorm,       schema::PrimitiveType_FusedBatchNorm};

static const std::vector<schema::PrimitiveType> fp32FullOpList = {
  schema::PrimitiveType_Concat, schema::PrimitiveType_Add,
  schema::PrimitiveType_Floor};  // fp32 ops support C4 and nhwc in fp32

static const std::vector<schema::PrimitiveType> uint8NeedNhwcOpList = {};

static const std::vector<schema::PrimitiveType> uint8OpList = {
  schema::PrimitiveType_Nchw2Nhwc,       schema::PrimitiveType_Nhwc2Nchw, schema::PrimitiveType_Conv2D,
  schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_Add,       schema::PrimitiveType_Pooling,
  schema::PrimitiveType_Concat,          schema::PrimitiveType_SoftMax,   schema::PrimitiveType_Reshape,
  schema::PrimitiveType_Activation};

std::vector<schema::PrimitiveType> Getfp32FullOpList() { return fp32FullOpList; }

std::vector<schema::PrimitiveType> GetNhwcOpList() { return nhwcOpList; }

std::vector<schema::PrimitiveType> GetUint8NhwcOpList() { return uint8NeedNhwcOpList; }

std::vector<schema::PrimitiveType> GetUint8OpList() { return uint8OpList; }

STATUS NodeUtils::ConvertDims(mindspore::lite::Format src_format, const std::vector<int32_t> &src_dims,
                              mindspore::lite::Format dst_format, std::vector<int32_t> *dst_dims) {
  if ((src_dims.size() != DIM_DEFAULT_SIZE && src_dims.size() != 3) || src_format == dst_format) {
    // MS_LOG(ERROR)("Convert format , src size %lu <3 or src format is equal to dst format,not need convert",
    // src_dims.size());
    *dst_dims = src_dims;
    return RET_PARAM_INVALID;
  }

  std::vector<int32_t> nchw_dim;
  switch (src_format) {
    case Format_NCHW:
      nchw_dim = src_dims;
      break;
    case Format_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        nchw_dim.push_back(src_dims[NHWC_N]);
        nchw_dim.push_back(src_dims[NHWC_C]);
        nchw_dim.push_back(src_dims[NHWC_H]);
        nchw_dim.push_back(src_dims[NHWC_W]);
      } else {
        nchw_dim.push_back(src_dims[HWC_C]);
        nchw_dim.push_back(src_dims[HWC_H]);
        nchw_dim.push_back(src_dims[HWC_W]);
      }
      break;
    default:
      // MS_LOG(ERROR)("Not support src format: %d", src_format);
      return RET_ERROR;
  }

  if (nchw_dim.size() == 0) {
    // MS_LOG(ERROR)("Param nchw_dim is empty!");
    return RET_ERROR;
  }

  switch (dst_format) {
    case Format_NCHW:
      *dst_dims = nchw_dim;
      break;
    case Format_NHWC:
      if (src_dims.size() == DIM_DEFAULT_SIZE) {
        dst_dims->push_back(nchw_dim[NCHW_N]);
        dst_dims->push_back(nchw_dim[NCHW_H]);
        dst_dims->push_back(nchw_dim[NCHW_W]);
        dst_dims->push_back(nchw_dim[NCHW_C]);
      }
      break;
    default:
      // MS_LOG(ERROR)("Not support dst format: %d", dst_format);
      return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore



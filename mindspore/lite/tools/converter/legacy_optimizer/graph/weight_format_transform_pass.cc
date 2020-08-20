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

#include "tools/converter/legacy_optimizer/graph/weight_format_transform_pass.h"
#include <queue>
#include "tools/common/node_util.h"
#include "tools/common/converter_op_utils.h"
#include "utils/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace lite {
void WeightFormatTransformPass::SetQuantType(QuantType quantType) { this->quantType = quantType; }

void WeightFormatTransformPass::SetFmkType(converter::FmkType fmkType) { this->fmkType = fmkType; }

void WeightFormatTransformPass::SetDstFormat(Format format) { this->dstFormat = format; }

STATUS WeightFormatTransformPass::Run(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  if (this->quantType == QuantType_AwareTraining) {
    auto status = QuantDataFormatTrans(graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "QuantDataFormatTrans failed: " << status;
      return status;
    }
  } else {
    auto status = NonQuantDataFormatTrans(graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "NonQuantDataFormatTrans failed: " << status;
      return status;
    }
  }
  return RET_OK;
}

STATUS WeightFormatTransformPass::QuantDataFormatTrans(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    MS_ASSERT(node != nullptr);
    MS_ASSERT(node->primitive != nullptr);
    auto opType = node->primitive->value.type;
    if (opType != PrimitiveType_Conv2D && opType != PrimitiveType_DepthwiseConv2D) {
      continue;
    }
    MS_ASSERT(node->inputIndex.size() >= 2);
    auto weightIndex = node->inputIndex.at(1);
    MS_ASSERT(subGraph->allTensors.size() > weightIndex);
    auto &weightTensor = graph->allTensors[weightIndex];
    MS_ASSERT(weightTensor->dataType == DataType_DT_UINT8 || weightTensor->dataType == DataType_DT_FLOAT);
    STATUS status;
    if (opType == PrimitiveType_Conv2D || opType == PrimitiveType_DepthwiseConv2D) {  // weight should be HWCK
      Format curDstFormat;
      if (this->dstFormat == Format_NUM_OF_FORMAT) {
        curDstFormat = Format_KHWC;
      } else {
        curDstFormat = this->dstFormat;
      }
      status = TransFilterFormat(weightTensor.get(), curDstFormat);
      if (status == RET_OK) {
        //        node->primitive->value.AsConv2D()->format = schema::Format_NHWC;
        weightTensor->format = curDstFormat;
      } else {
        MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(weightTensor->format) << "To"
                        << EnumNameFormat(curDstFormat) << " failed, node : " << node->name;
        return ERROR;
      }
    }
  }
  return RET_OK;
}

STATUS WeightFormatTransformPass::NonQuantDataFormatTrans(MetaGraphT *graph) {
  MS_ASSERT(graph != nullptr);
  for (auto &node : graph->nodes) {
    MS_ASSERT(node != nullptr);
    MS_ASSERT(node->primitive != nullptr);
    auto opType = node->primitive->value.type;
    if (opType != PrimitiveType_Conv2D && opType != PrimitiveType_DepthwiseConv2D && opType != PrimitiveType_DeConv2D &&
        opType != PrimitiveType_DeDepthwiseConv2D) {
      continue;
    }
    MS_ASSERT(node->inputIndex.size() >= 2);
    auto weightIndex = node->inputIndex.at(1);
    MS_ASSERT(subGraph->allTensors.size() > weightIndex);
    auto &weightTensor = graph->allTensors[weightIndex];
    MS_ASSERT(weightTensor->dataType == DataType_DT_UINT8 || weightTensor->dataType == DataType_DT_FLOAT);
    STATUS status;
    if (opType == PrimitiveType_Conv2D || opType == PrimitiveType_DepthwiseConv2D ||
        opType == schema::PrimitiveType_DeConv2D) {
      Format curDstFormat;
      if (this->dstFormat == Format_NUM_OF_FORMAT) {
        curDstFormat = Format_KHWC;
      } else {
        curDstFormat = this->dstFormat;
      }
      status = TransFilterFormat(weightTensor.get(), curDstFormat);
      if (status == RET_OK) {
        //          node->attr.AsConv2D()->format = Format_NCHW;
        weightTensor->format = curDstFormat;
      } else {
        MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(weightTensor->format) << "To"
                        << EnumNameFormat(curDstFormat) << " failed, node : " << node->name;
        return ERROR;
      }
    } else {  // weight should be CKHW
      Format curDstFormat;
      if (this->dstFormat == Format_NUM_OF_FORMAT) {
        curDstFormat = Format_KHWC;
      } else {
        curDstFormat = this->dstFormat;
      }
      status = TransFilterFormat(weightTensor.get(), curDstFormat);
      if (status == RET_OK) {
        //          node->attr.AsDepthwiseConv2D()->format = Format_NCHW;
        weightTensor->format = curDstFormat;
      } else {
        MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(weightTensor->format) << "To"
                        << EnumNameFormat(curDstFormat) << " failed, node : " << node->name;
        return ERROR;
      }
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

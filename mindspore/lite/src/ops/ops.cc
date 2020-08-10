/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/ops.h"
#include <vector>
#include "utils/log_adapter.h"
#include "include/errorcode.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
Primitive *Primitive::CreatePrimitive(schema::Primitive *primitive) {
  MS_ASSERT(primitive != nullptr);
  auto op_type = primitive->value_type();
  switch (op_type) {
    case schema::PrimitiveType_SoftMax:
      return new lite::SoftMax(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Activation:
      return new lite::Activation(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Conv2D:
      return new lite::Conv2D(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reduce:
      return new lite::Reduce(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Pooling:
      return new lite::Pooling(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ROIPooling:
      return new lite::ROIPooling(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_DepthwiseConv2D:
      return new lite::DepthwiseConv2D(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FusedBatchNorm:
      return new lite::FusedBatchNorm(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_BatchNorm:
      return new lite::BatchNorm(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FullConnection:
      return new lite::FullConnection(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Power:
      return new lite::Power(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Pad:
      return new lite::Pad(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Range:
      return new lite::Range(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Mul:
      return new lite::Mul(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Add:
      return new lite::Add(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Sub:
      return new lite::Sub(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Div:
      return new lite::Div(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_BiasAdd:
      return new lite::BiasAdd(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ExpandDims:
      return new lite::ExpandDims(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ArgMax:
      return new lite::ArgMax(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ArgMin:
      return new lite::ArgMin(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Cast:
      return new lite::Cast(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reshape:
      return new lite::Reshape(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Eltwise:
      return new lite::Eltwise(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Ceil:
      return new lite::Ceil(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Concat:
      return new lite::Concat(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Fill:
      return new lite::Fill(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Nhwc2Nchw:
      return new lite::Nhwc2Nchw(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Nchw2Nhwc:
      return new lite::Nchw2Nhwc(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Transpose:
      return new lite::Transpose(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Squeeze:
      return new lite::Squeeze(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SquaredDifference:
      return new lite::SquaredDifference(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Split:
      return new lite::Split(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FloorDiv:
      return new lite::FloorDiv(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_FloorMod:
      return new lite::FloorMod(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Reverse:
      return new lite::Reverse(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Scale:
      return new lite::Scale(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_GatherNd:
      return new lite::GatherNd(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Tile:
      return new lite::Tile(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_TopK:
      return new lite::TopK(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unique:
      return new lite::Unique(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unstack:
      return new lite::Unstack(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ReverseSequence:
      return new lite::ReverseSequence(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Round:
      return new lite::Round(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ZerosLike:
      return new lite::ZerosLike(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Where:
      return new lite::Where(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Floor:
      return new lite::Floor(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Shape:
      return new lite::Shape(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_ScatterND:
      return new lite::ScatterND(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Unsqueeze:
      return new lite::Unsqueeze(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Flatten:
      return new lite::Flatten(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_StridedSlice:
      return new lite::StridedSlice(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_Resize:
      return new lite::Resize(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_OneHot:
      return new lite::OneHot(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_PriorBox:
      return new lite::PriorBox(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SpaceToDepth:
      return new lite::SpaceToDepth(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_SpaceToBatch:
      return new lite::SpaceToBatch(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_QuantDTypeCast:
      return new lite::QuantDTypeCast(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_MatMul:
      return new lite::MatMul(const_cast<schema::Primitive *>(primitive));
    case schema::PrimitiveType_EmbeddingLookup:
      return new lite::EmbeddingLookup(const_cast<schema::Primitive *>(primitive));
    default:
      break;
  }
  return nullptr;
}

int Primitive::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_shape(input->shape());
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite

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

#include "src/ops/space_to_batch_nd.h"
#include <limits>
#include "src/common/common.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
namespace {
constexpr int kSpaceToBatchNDOutputNum = 1;
constexpr int kSpaceToBatchNDInputNum = 1;
}  // namespace

#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SpaceToBatchND::GetBlockShape() const {
  return this->primitive_->value.AsSpaceToBatchND()->blockShape;
}
std::vector<int> SpaceToBatchND::GetPaddings() const { return this->primitive_->value.AsSpaceToBatchND()->paddings; }

void SpaceToBatchND::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive_->value.AsSpaceToBatchND()->blockShape = block_shape;
}
void SpaceToBatchND::SetPaddings(const std::vector<int> &paddings) {
  this->primitive_->value.AsSpaceToBatchND()->paddings = paddings;
}

#else

std::vector<int> SpaceToBatchND::GetBlockShape() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatchND()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SpaceToBatchND::GetPaddings() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatchND()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

int SpaceToBatchND::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SpaceToBatchND();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SpaceToBatch return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> blockShape;
  if (attr->blockShape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->blockShape()->size()); i++) {
      blockShape.push_back(attr->blockShape()->data()[i]);
    }
  }
  std::vector<int32_t> paddings;
  if (attr->paddings() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->paddings()->size()); i++) {
      paddings.push_back(attr->paddings()->data()[i]);
    }
  }
  auto val_offset = schema::CreateSpaceToBatchDirect(*fbb, &blockShape, &paddings);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SpaceToBatchND, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SpaceToBatchNDCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SpaceToBatchND>(primitive);
}
Registry SpaceToBatchNDRegistry(schema::PrimitiveType_SpaceToBatchND, SpaceToBatchNDCreator);

#endif  // PRIMITIVE_WRITEABLE

int SpaceToBatchND::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  if (outputs.size() != kSpaceToBatchNDOutputNum || inputs.size() != kSpaceToBatchNDInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->format() != schema::Format::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch_nd only support NHWC now!";
    return RET_ERROR;
  }
  outputs.at(0)->set_data_type(input->data_type());
  outputs.at(0)->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kQuadrupleNum) {
    MS_LOG(ERROR) << "input shape dimension size only support " << kQuadrupleNum << " now!";
    return RET_ERROR;
  }
  auto block_shape = GetBlockShape();
  auto padding = GetPaddings();
  int padding_left = 0;
  int padding_right = 0;
  int block_w = 1;
  if (block_shape.size() == 2) {
    padding_left = padding.at(2);
    padding_right = padding.at(3);
    block_w = block_shape.at(1);
  }
  std::vector<int32_t> output_shape(input_shape.size());
  if (block_shape.at(0) * block_w > std::numeric_limits<int>::max() / input_shape.at(NHWC_N)) {
    MS_LOG(ERROR) << "The value of block_shape.at(0) * block_w is too big";
    return RET_ERROR;
  }
  output_shape.at(NHWC_N) = input_shape.at(NHWC_N) * block_shape.at(0) * block_w;
  if (padding.at(0) + padding.at(1) > std::numeric_limits<int>::max() - input_shape.at(NHWC_H)) {
    MS_LOG(ERROR) << "The value of padding.at(0) + padding.at(1) is too big";
    return RET_ERROR;
  }
  output_shape.at(NHWC_H) = (input_shape.at(NHWC_H) + padding.at(0) + padding.at(1)) / block_shape.at(0);
  if (padding_left + padding_right > std::numeric_limits<int>::max() - input_shape.at(NHWC_W)) {
    MS_LOG(ERROR) << "The value of padding_left + padding_right is too big";
    return RET_ERROR;
  }
  output_shape.at(NHWC_W) = (input_shape.at(NHWC_W) + padding_left + padding_right) / block_w;
  output_shape.at(NHWC_C) = input_shape.at(NHWC_C);
  outputs.at(0)->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

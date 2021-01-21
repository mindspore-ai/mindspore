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

#include "src/ops/space_to_batch.h"
#include "src/common/common.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SpaceToBatch::GetBlockShape() const { return this->primitive_->value.AsSpaceToBatch()->blockShape; }
std::vector<int> SpaceToBatch::GetPaddings() const { return this->primitive_->value.AsSpaceToBatch()->paddings; }

void SpaceToBatch::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive_->value.AsSpaceToBatch()->blockShape = block_shape;
}
void SpaceToBatch::SetPaddings(const std::vector<int> &paddings) {
  this->primitive_->value.AsSpaceToBatch()->paddings = paddings;
}

#else

std::vector<int> SpaceToBatch::GetBlockShape() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatch()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> SpaceToBatch::GetPaddings() const {
  auto fb_vector = this->primitive_->value_as_SpaceToBatch()->paddings();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int SpaceToBatch::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SpaceToBatch();
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
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SpaceToBatch, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *SpaceToBatchCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<SpaceToBatch>(primitive);
}
Registry SpaceToBatchRegistry(schema::PrimitiveType_SpaceToBatch, SpaceToBatchCreator);

#endif

namespace {
constexpr int kSpaceToBatchNDOutputNum = 1;
constexpr int kSpaceToBatchNDInputNum = 1;
}  // namespace

int SpaceToBatch::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kSpaceToBatchNDOutputNum || inputs.size() != kSpaceToBatchNDInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return 1;
  }

  auto input = inputs.at(0);
  if (input->format() != schema::Format::Format_NHWC) {
    MS_LOG(ERROR) << "space_to_batch only support NHWC now!";
    return 1;
  }
  outputs[0]->set_data_type(input->data_type());
  outputs[0]->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kQuadrupleNum) {
    MS_LOG(ERROR) << "Space_to_batch op only support 4D input currently. But got %d dimensionality input."
                  << kQuadrupleNum;
    return RET_ERROR;
  }

  auto block_shape_vector = GetBlockShape();
  for (int &iter : block_shape_vector) {
    block_sizes_.emplace_back(iter);
  }

  in_shape_.clear();
  padded_in_shape_.clear();
  paddings_.clear();
  in_shape_.emplace_back(input_shape.at(NHWC_N));
  padded_in_shape_.emplace_back(input_shape.at(NHWC_N));
  auto block_shape_size = block_shape_vector.size();
  for (size_t i = 0; i < block_shape_size; i++) {
    in_shape_.emplace_back(input_shape.at(i + 1));
    padded_in_shape_.emplace_back(input_shape.at(i + 1) + (paddings_.at(2 * i) + paddings_.at(2 * i + 1)));
    paddings_.emplace_back(paddings_.at(2 * i));
    paddings_.emplace_back(paddings_.at(2 * i + 1));
    if (paddings_.back() % block_sizes_.at(i)) {
      MS_LOG(ERROR) << "Padded shape does not divide block size " << block_sizes_.at(i);
      return 1;
    }
  }
  in_shape_.emplace_back(input_shape.at(NHWC_C));
  padded_in_shape_.emplace_back(input_shape.at(NHWC_C));
  int padding_left = 0;
  int padding_right = 0;
  int block_w = 1;
  if (block_shape_size == 2) {
    padding_left = paddings_[2];
    padding_right = paddings_[3];
    block_w = block_sizes_[1];
  }

  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N] * (block_sizes_[0] * block_w);
  output_shape[NHWC_H] = (input_shape[NHWC_H] + paddings_[0] + paddings_[1]) / block_sizes_[0];
  output_shape[NHWC_W] = (input_shape[NHWC_W] + padding_left + padding_right) / block_w;
  output_shape[NHWC_C] = input_shape[NHWC_C];
  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

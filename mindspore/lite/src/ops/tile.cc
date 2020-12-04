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

#include "src/ops/tile.h"
#include <algorithm>

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Tile::GetMultiples() const { return this->primitive_->value.AsTile()->multiples; }

void Tile::SetMultiples(const std::vector<int> &multiples) { this->primitive_->value.AsTile()->multiples = multiples; }

std::vector<int> Tile::GetDims() const { return this->primitive_->value.AsTile()->dims; }

void Tile::SetDims(const std::vector<int> &dims) { this->primitive_->value.AsTile()->dims = dims; }

int Tile::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Tile;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Tile) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TileT();

    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    if (prim.GetAttr("dims") == nullptr) {
      MS_LOG(INFO) << "Tile's attr dims is set to default";
      attr->dims = {1};
    } else {
      attr->dims = CastToInt(prim.GetAttr("dims"));
    }
    if (inputs.size() == kAnfPopulaterInputNumTwo) {
      auto inputNode = inputs[kAnfPopulaterInputNumOne];
      MS_ASSERT(inputNode != nullptr);
      if (inputNode->isa<ValueNode>()) {
        auto valueNode = inputNode->cast<ValueNodePtr>();
        MS_ASSERT(valueNode != nullptr);
        auto value = valueNode->value();
        MS_ASSERT(value != nullptr);
        if (value->isa<ValueTuple>()) {
          auto valTuplPtr = dyn_cast<ValueTuple>(value);
          MS_ASSERT(valTuplPtr != nullptr);
          for (size_t i = 0; i < valTuplPtr->size(); i++) {
            auto elem = (*valTuplPtr)[i];
            MS_ASSERT(elem != nullptr);
            attr->multiples.emplace_back(CastToInt(elem).front());
          }
        } else {
          int multiple = CastToInt(value).front();
          attr->multiples = {multiple};
        }
      }
    }
    this->primitive_->value.value = attr;
  }
  return RET_OK;
}

#else

std::vector<int> Tile::GetMultiples() const {
  auto fb_vector = this->primitive_->value_as_Tile()->multiples();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

std::vector<int> Tile::GetDims() const {
  auto fb_vector = this->primitive_->value_as_Tile()->dims();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int Tile::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Tile();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Tile return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> multiples;
  if (attr->multiples() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->multiples()->size()); i++) {
      multiples.push_back(attr->multiples()->data()[i]);
    }
  }
  std::vector<int32_t> dims;
  if (attr->dims() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->dims()->size()); i++) {
      dims.push_back(attr->dims()->data()[i]);
    }
  }
  auto val_offset = schema::CreateTileDirect(*fbb, &multiples, &dims);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Tile, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TileCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Tile>(primitive); }
Registry TileRegistry(schema::PrimitiveType_Tile, TileCreator);
#endif

int Tile::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  std::vector<int> out_shape;
  std::vector<int> multiples = GetMultiples();
  std::vector<int> dims = GetDims();
  const size_t in_dims = input->shape().size();

  MS_ASSERT(multiples.size() == dims.size());
  for (size_t i = 0; i < in_dims; ++i) {
    out_shape.push_back(input->shape()[i]);
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    out_shape[dims[i]] = input->shape()[dims[i]] * (multiples[i]);
  }

  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

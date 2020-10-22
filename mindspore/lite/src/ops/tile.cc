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

#include "src/ops/ops_register.h"
#include "nnacl/fp32/tile.h"

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
    if (inputs.size() == kAnfPopulaterTwo) {
      auto inputNode = inputs[kAnfPopulaterOne];
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
            auto elem = dyn_cast<Int32Imm>((*valTuplPtr)[i]);
            MS_ASSERT(elem != nullptr);
            attr->multiples.emplace_back(elem->value());
          }
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

OpParameter *PopulateTileParameter(const mindspore::lite::PrimitiveC *primitive) {
  TileParameter *tile_param = reinterpret_cast<TileParameter *>(malloc(sizeof(TileParameter)));
  if (tile_param == nullptr) {
    MS_LOG(ERROR) << "malloc TileParameter failed.";
    return nullptr;
  }
  memset(tile_param, 0, sizeof(TileParameter));
  tile_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::Tile *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto multiples = param->GetMultiples();
  tile_param->in_dim_ = multiples.size();
  for (int i = 0; i < tile_param->in_dim_; ++i) {
    tile_param->multiples_[i] = multiples[i];
  }
  return reinterpret_cast<OpParameter *>(tile_param);
}

Registry TileParameterRegistry(schema::PrimitiveType_Tile, PopulateTileParameter);

int Tile::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  MS_ASSERT(tile_prim != nullptr);
  std::vector<int> out_shape;
  std::vector<int> multiples = GetMultiples();
  const size_t in_dims = input->shape().size();
  const size_t delta_dims = in_dims - multiples.size();

  size_t i = 0;
  for (; i < delta_dims; ++i) {
    int tmp = input->shape()[i];
    out_shape.push_back(tmp);
  }
  for (; i < in_dims; ++i) {
    int tmp = input->shape()[i] * (multiples[i - delta_dims]);
    out_shape.push_back(tmp);
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

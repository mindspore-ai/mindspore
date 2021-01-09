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
#include <limits>
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
    if (prim.GetAttr("dims") == nullptr) {
      MS_LOG(INFO) << "Tile's attr dims is set to default. The operator in mindspore has no attribute"
                      "named dims and all the dimensions needs to be multiplied by default.";
      for (size_t i = 0; i < attr->multiples.size(); i++) {
        attr->dims.push_back(i);
      }
    } else {
      attr->dims = CastToInt(prim.GetAttr("dims"));
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
  std::vector<int> multiples;
  if (inputs_.size() == 2) {
    if (inputs_[1]->data_c() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return RET_INFER_INVALID;
    }
    int data_num = inputs_[1]->ElementsNum();
    if (data_num > static_cast<int>(input->shape().size())) {
      MS_LOG(ERROR) << "multiples data num cannot be larger than input shape size.";
      return RET_INPUT_TENSOR_ERROR;
    }
    multiples.resize(data_num);
    memcpy(multiples.data(), inputs_[1]->data_c(), inputs_[1]->Size());
  } else {
    multiples = GetMultiples();
  }
#ifdef SUPPORT_TRAIN
  const size_t in_dims = input->shape().size();
  const size_t delta_dims = in_dims - multiples.size();

  size_t i = 0;
  for (; i < delta_dims; ++i) {
    int tmp = input->shape().at(i);
    out_shape.push_back(tmp);
  }
  for (; i < in_dims; ++i) {
    int tmp = input->shape().at(i) * (multiples[i - delta_dims]);
    out_shape.push_back(tmp);
  }
#else
  std::vector<int> dims = GetDims();
  if (inputs_.size() == 2 && dims.empty()) {
    for (int dim = 0; dim < inputs_[1]->ElementsNum(); ++dim) {
      dims.push_back(dim);
    }
  }
  const size_t in_dims = input->shape().size();

  MS_ASSERT(multiples.size() == dims.size());
  for (size_t i = 0; i < in_dims; ++i) {
    out_shape.push_back(input->shape().at(i));
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    if (multiples.at(i) > std::numeric_limits<int>::max() / input->shape().at(dims.at(i))) {
      MS_LOG(ERROR) << "The value of multiples[" << i << "] is too big";
      return RET_ERROR;
    }
    out_shape.at(dims.at(i)) = input->shape().at(dims.at(i)) * (multiples.at(i));
  }
#endif
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

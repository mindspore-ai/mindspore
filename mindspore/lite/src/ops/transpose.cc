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

#include "src/ops/transpose.h"
#include <memory>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Transpose::GetPerm() const { return this->primitive_->value.AsTranspose()->perm; }
void Transpose::SetPerm(const std::vector<int> &perm) { this->primitive_->value.AsTranspose()->perm = perm; }

int Transpose::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Transpose;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Transpose) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::TransposeT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new TransposeT failed";
      return RET_ERROR;
    }
    MS_ASSERT(inputs.size() == kAnfPopulaterInputNumTwo);
    auto inputNode = inputs[kAnfPopulaterInputNumOne];
    if (inputNode->isa<ValueNode>()) {
      auto valNode = inputNode->cast<ValueNodePtr>();
      MS_ASSERT(valNode != nullptr);
      auto val = valNode->value();
      MS_ASSERT(val != nullptr);
      if (val->isa<ValueTuple>()) {
        auto tuple = val->cast<ValueTuplePtr>();
        MS_ASSERT(tuple != nullptr);
        for (size_t i = 0; i < tuple->size(); i++) {
          auto elem = tuple->value().at(i);
          MS_ASSERT(elem != nullptr);
          attr->perm.emplace_back(CastToInt(elem).front());
        }
      }
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else

std::vector<int> Transpose::GetPerm() const {
  auto fb_vector = this->primitive_->value_as_Transpose()->perm();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

int Transpose::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Transpose();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Transpose return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> perm;
  if (attr->perm() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->perm()->size()); i++) {
      perm.push_back(attr->perm()->data()[i]);
    }
  }

  auto val_offset = schema::CreateTransposeDirect(*fbb, &perm, attr->conjugate());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Transpose, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *TransposeCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<Transpose>(primitive);
}
Registry TransposeRegistry(schema::PrimitiveType_Transpose, TransposeCreator);

#endif

int Transpose::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  auto input = inputs_.front();
  auto output = outputs_.front();
  MS_ASSERT(input != nullptr);
  MS_ASSERT(output != nullptr);

  std::vector<int> perm = GetPerm();
  if (inputs_.size() == kDoubleNum) {
    auto input_perm = inputs_.at(1);
    MS_ASSERT(input_perm != nullptr);
    if (input_perm->data_c() == nullptr) {
      return RET_INFER_INVALID;
    }
    int *perm_data = reinterpret_cast<int *>(input_perm->data_c());
    perm = std::vector<int>{perm_data, perm_data + input_perm->ElementsNum()};
  }
  std::vector<int> nchw2nhwc_perm = {0, 2, 3, 1};
  std::vector<int> nhwc2nchw_perm = {0, 3, 1, 2};
  std::vector<int> in_shape = input->shape();

  output->set_data_type(input->data_type());
  if (input->format() == schema::Format::Format_NCHW && perm == nchw2nhwc_perm) {
    output->set_format(schema::Format::Format_NHWC);
  } else if (input->format() == schema::Format::Format_NHWC && perm == nhwc2nchw_perm) {
    output->set_format(schema::Format::Format_NCHW);
  } else {
    output->set_format(input->format());
  }
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  if (in_shape.size() != 4 && perm.size() == 4) {
    output->set_shape(in_shape);
    return RET_OK;
  }
  std::vector<int> out_shape;
  out_shape.resize(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    out_shape.at(i) = in_shape.at(perm.at(i));
  }
  if (perm.empty()) {
    auto shape_size = in_shape.size();
    out_shape.resize(shape_size);
    for (size_t i = 0; i < shape_size; ++i) {
      out_shape[shape_size - i - 1] = in_shape[i];
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

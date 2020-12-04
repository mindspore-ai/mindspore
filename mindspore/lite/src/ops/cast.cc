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

#include "src/ops/cast.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Cast::GetSrcT() const { return this->primitive_->value.AsCast()->srcT; }
int Cast::GetDstT() const { return this->primitive_->value.AsCast()->dstT; }

void Cast::SetSrcT(int src_t) { this->primitive_->value.AsCast()->srcT = src_t; }
void Cast::SetDstT(int dst_t) { this->primitive_->value.AsCast()->dstT = dst_t; }

int Cast::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Cast;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Cast) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::CastT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    auto srcAnf = reinterpret_cast<mindspore::Number *>(prim.GetAttr("SrcT").get());
    auto dstAnf = reinterpret_cast<mindspore::Number *>(prim.GetAttr("DstT").get());
    attr->srcT = srcAnf->number_type();
    attr->dstT = dstAnf->number_type();
    this->primitive_->value.value = attr;
  }

  return RET_OK;
}

#else
int Cast::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Cast();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Cast return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateCast(*fbb, attr->srcT(), attr->dstT());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Cast, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Cast::GetSrcT() const { return this->primitive_->value_as_Cast()->srcT(); }
int Cast::GetDstT() const { return this->primitive_->value_as_Cast()->dstT(); }

PrimitiveC *CastCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Cast>(primitive); }
Registry CastRegistry(schema::PrimitiveType_Cast, CastCreator);
#endif

int Cast::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "tensor number is error.";
    return RET_INPUT_TENSOR_ERROR;
  }
  output->set_format(input->format());

  output->set_data_type(static_cast<TypeId>(GetDstT()));
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  if (GetSrcT() != 0 && input->data_type() != GetSrcT()) {
    MS_LOG(ERROR) << "input dataType is error";
    return RET_INPUT_TENSOR_ERROR;
  }
  if (kSupportDataType.find(input->data_type()) == kSupportDataType.end()) {
    MS_LOG(ERROR) << "Unsupported input data type " << input->data_type();
    return RET_INPUT_TENSOR_ERROR;
  }

  output->set_shape(input->shape());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

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
#include "src/ops/lsh_projection.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int LshProjection::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) { return RET_OK; }
int LshProjection::GetLshType() const { return this->primitive_->value.AsLshProjection()->type; }
#else
int LshProjection::GetLshType() const { return this->primitive_->value_as_LshProjection()->type(); }

int LshProjection::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_LshProjection();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "LshProjection attr is nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateLshProjection(*fbb, attr->type());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_LshProjection, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *LshProjectionCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<LshProjection>(primitive);
}
Registry LshProjectionRegistry(schema::PrimitiveType_LshProjection, LshProjectionCreator);

#endif

int LshProjection::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != kDoubleNum && inputs_.size() != kTripleNum) {
    MS_LOG(ERROR) << "inputs to LshProjection operator should be 2 or 3, but " << inputs_.size() << " is given.";
    return RET_ERROR;
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "outputs to Shape operator should be 1, but " << outputs_.size() << " is given.";
    return RET_ERROR;
  }

  auto in_hash = inputs_.at(0);
  MS_ASSERT(in_hash->shape().size() == 2);
  MS_ASSERT(in_hash->DimensionSize(1) <= 32);
  MS_ASSERT(inputs_.at(1)->shape().size() >= 1);

  if (inputs_.size() == kTripleNum) {
    MS_ASSERT(inputs_.at(2)->shape().size() == 1);
    MS_ASSERT(inputs_.at(2)->DimensionSize(0) == inputs_.at(1)->DimensionSize(0));
  }

  auto out_tensor = outputs_.front();
  out_tensor->set_data_type(kNumberTypeInt32);
  out_tensor->set_format(schema::Format::Format_NHWC);

  std::vector<int> out_shape;
  switch (GetLshType()) {
    case schema::LshProjectionType_SPARSE:
      out_shape.push_back(in_hash->DimensionSize(0));
      break;
    case schema::LshProjectionType_DENSE:
      out_shape.push_back(in_hash->DimensionSize(0) * in_hash->DimensionSize(1));
      break;
    default:
      return RET_ERROR;
  }
  out_tensor->set_shape(out_shape);
  return RET_OK;
}

}  // namespace lite
}  // namespace mindspore

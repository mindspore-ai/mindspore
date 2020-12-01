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

#include <memory>
#include "src/ops/unsorted_segment_sum.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE

int UnsortedSegmentSum::GetNumSegments() const { return this->primitive_->value.AsUnsortedSegmentSum()->numSegments; }

int UnsortedSegmentSum::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitive error";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_UnsortedSegmentSum;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_UnsortedSegmentSum) {
    MS_LOG(ERROR) << "UnSortedSegmentSum primitive value type :  "
                  << schema::EnumNamePrimitiveType(primitive_->value.type) << "is  not equal"
                  << schema::EnumNamePrimitiveType(schema::PrimitiveType_UnsortedSegmentSum);
    delete this->primitive_;
    this->primitive_ = nullptr;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    std::unique_ptr<schema::UnsortedSegmentSumT> attr = std::make_unique<schema::UnsortedSegmentSumT>();
    if (inputs.at(2)->isa<ValueNode>()) {
      ValuePtr value = inputs.at(2)->cast<ValueNodePtr>()->value();
      attr->numSegments = CastToInt(value).front();
      this->primitive_->value.value = attr.release();
    }
  }
  return RET_OK;
}
#else
int UnsortedSegmentSum::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_UnsortedSegmentSum();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_UnsortedSegmentSum return nullptr";
    return RET_ERROR;
  }
  int num_segments = attr->numSegments();
  auto val_offset = schema::CreateUnsortedSegmentSum(*fbb, num_segments);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_UnsortedSegmentSum, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

int UnsortedSegmentSum::GetNumSegments() const {
  int ret = this->primitive_->value_as_UnsortedSegmentSum()->numSegments();
  return ret;
}

PrimitiveC *UnsortedSegmentSumCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<UnsortedSegmentSum>(primitive);
}
Registry UnsortedSegmentSumRegistry(schema::PrimitiveType_UnsortedSegmentSum, UnsortedSegmentSumCreator);
#endif
int UnsortedSegmentSum::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  // check inputs and outputs
  if (inputs_.size() != 3) {
    MS_LOG(ERROR) << "invalid inputs numbers";
    return RET_ERROR;
  }
  if (outputs_.size() != 1) {
    MS_LOG(ERROR) << "invalid outputs numbers";
    return RET_ERROR;
  }
  Tensor *out = outputs_.front();
  Tensor *x = inputs_.front();
  Tensor *segment_id = inputs_.at(1);
  std::vector<int> x_shape = x->shape();
  std::vector<int> segment_id_shape = segment_id->shape();
  int num_segments = GetNumSegments();
  std::vector<int> output_shape;
  output_shape.push_back(num_segments);
  for (int index = segment_id_shape.size(); index < static_cast<int>(x_shape.size()); index++) {
    output_shape.push_back(x_shape.at(index));
  }
  out->set_shape(output_shape);
  out->set_format(x->format());
  out->set_data_type(x->data_type());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

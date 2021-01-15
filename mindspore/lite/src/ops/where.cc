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

#include "src/ops/where.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<bool> Where::GetCondition() const { return this->primitive_->value.AsWhere()->condition; }

void Where::SetCondition(const std::vector<bool> &condition) {
  this->primitive_->value.AsWhere()->condition = condition;
}

#else

std::vector<bool> Where::GetCondition() const {
  auto fb_vector = this->primitive_->value_as_Where()->condition();
  return std::vector<bool>(fb_vector->begin(), fb_vector->end());
}
int Where::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Where();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Where return nullptr";
    return RET_ERROR;
  }
  std::vector<uint8_t> condition;
  if (attr->condition() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->condition()->size()); i++) {
      condition.push_back(attr->condition()->data()[i]);
    }
  }
  auto val_offset = schema::CreateWhereDirect(*fbb, &condition);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Where, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *WhereCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Where>(primitive); }
Registry WhereRegistry(schema::PrimitiveType_Where, WhereCreator);

#endif

int Where::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  // Need to dynamically allocate at runtime.
  if (inputs_.size() == kSingleNum) {
    return RET_INFER_INVALID;
  }

  if (inputs_.size() < kTripleNum || outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "where input or output number invalid, Input size:" << inputs_.size()
                  << ", output size: " << outputs_.size();
    return RET_INPUT_TENSOR_ERROR;
  }

  auto input0 = inputs_.at(0);
  auto input1 = inputs_.at(1);
  auto input2 = inputs_.at(2);
  output->set_data_type(input->data_type());
  output->set_format(input->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  int num = input0->ElementsNum();
  int num1 = input1->ElementsNum();
  int num2 = input2->ElementsNum();
  int nummax = num > num1 ? num : (num1 > num2 ? num1 : num2);
  auto shape_tmp = inputs_.at(0)->shape();
  auto shape_tmp1 = inputs_.at(1)->shape();
  auto shape_tmp2 = inputs_.at(2)->shape();
  int axisout = 0;
  size_t temp = 0;
  for (size_t j = 0; j < shape_tmp.size(); j++) {
    if (shape_tmp.at(j) == shape_tmp1.at(j) && shape_tmp.at(j) != shape_tmp2.at(j)) {
      axisout = j;
      break;
    }
    if (shape_tmp.at(j) == shape_tmp2.at(j) && shape_tmp.at(j) != shape_tmp1.at(j)) {
      axisout = j;
      break;
    }
    if (shape_tmp1.at(j) == shape_tmp2.at(j) && shape_tmp.at(j) != shape_tmp1.at(j)) {
      axisout = j;
      break;
    }
    temp += 1;
    if (temp == shape_tmp.size()) {
      outputs_.at(0)->set_shape(shape_tmp);
      output->set_data_type(input->data_type());
      return RET_OK;
    }
  }
  auto output_shape = shape_tmp;
  output_shape.at(axisout) = nummax;
  outputs_.at(0)->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

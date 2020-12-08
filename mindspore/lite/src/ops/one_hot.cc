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

#include "src/ops/one_hot.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int OneHot::GetAxis() const { return this->primitive_->value.AsOneHot()->axis; }

void OneHot::SetAxis(int axis) { this->primitive_->value.AsOneHot()->axis = axis; }

int OneHot::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_OneHot;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_OneHot) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::OneHotT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    attr->axis = -1;
    if (prim.GetAttr("axis") != nullptr) {
      attr->axis = CastToInt(prim.GetAttr("axis")).front();
    }
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
#else

int OneHot::GetAxis() const { return this->primitive_->value_as_OneHot()->axis(); }

int OneHot::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_OneHot();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_OneHot return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateOneHot(*fbb, attr->axis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_OneHot, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}

PrimitiveC *OneHotCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<OneHot>(primitive); }
Registry OneHotRegistry(schema::PrimitiveType_OneHot, OneHotCreator);
#endif

namespace {
constexpr size_t kOneHotInputNum = 4;
constexpr size_t kOneHotInputNumOpt = 3;
}  // namespace
int OneHot::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (this->primitive_ == nullptr) {
    return RET_NULL_PTR;
  }

  int axis = GetAxis();
  // indices, depth, on_value, off_value
  if (inputs.size() != kOneHotInputNum && inputs.size() != kOneHotInputNumOpt) {
    MS_LOG(ERROR) << "OneHot got inputs num " << inputs.size() << ", should be " << kOneHotInputNum << " or "
                  << kOneHotInputNumOpt;
    return RET_ERROR;
  }
  auto depth_tensor = inputs.at(1);
  if (depth_tensor == nullptr) {
    return RET_NULL_PTR;
  }
  const int *depth = static_cast<int *>(depth_tensor->MutableData());
  auto input = inputs.front();
  if (input == nullptr) {
    return RET_NULL_PTR;
  }
  auto on_value = inputs.at(2);
  if (on_value == nullptr) {
    return RET_NULL_PTR;
  }
  auto output = outputs.front();
  if (output == nullptr) {
    return RET_NULL_PTR;
  }
  output->set_data_type(on_value->data_type());
  output->set_format(on_value->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  const auto input_shape = input->shape();
  int input_rank = static_cast<int>(input_shape.size());
  if (axis < 0) {
    axis += input_rank + 1;
  }
  std::vector<int> output_shape(input_shape);
  output_shape.insert(output_shape.cbegin() + axis, *depth);
  output->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

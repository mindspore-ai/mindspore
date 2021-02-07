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

#include "src/ops/concat.h"
#include <memory>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Concat::GetAxis() const { return this->primitive_->value.AsConcat()->axis; }

void Concat::SetAxis(int axis) { this->primitive_->value.AsConcat()->axis = axis; }

int Concat::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_Concat;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_Concat) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::ConcatT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
    }
    auto prim_axis = CastToInt(prim.GetAttr("axis")).front();
    attr->axis = prim_axis;
    this->primitive_->value.value = attr;
    if (this->primitive_->value.value == nullptr) {
      MS_LOG(ERROR) << "primitive value is nullptr";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

#else
int Concat::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_Concat();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_Concat return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateConcat(*fbb, attr->axis());
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_Concat, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
int Concat::GetAxis() const { return this->primitive_->value_as_Concat()->axis(); }

PrimitiveC *ConcatCreator(const schema::Primitive *primitive) { return PrimitiveC::NewPrimitiveC<Concat>(primitive); }
Registry ConcatRegistry(schema::PrimitiveType_Concat, ConcatCreator);

#endif

namespace {
constexpr int kConcatOutputNum = 1;
}
int Concat::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (this->primitive_ == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr!";
    return RET_PARAM_INVALID;
  }
  auto input0 = inputs_.front();
  auto output = outputs_.front();
  if (outputs_.size() != kConcatOutputNum) {
    MS_LOG(ERROR) << "output size is error";
    return RET_PARAM_INVALID;
  }
  output->set_data_type(input0->data_type());
  output->set_format(input0->format());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto input0_shape = inputs_.at(0)->shape();
  auto axis = GetAxis() < 0 ? GetAxis() + input0_shape.size() : GetAxis();
  if (axis < 0 || axis >= input0_shape.size()) {
    MS_LOG(ERROR) << "Invalid axis: " << axis;
    return RET_PARAM_INVALID;
  }
  auto input0_shape_without_axis = input0_shape;
  input0_shape_without_axis.erase(input0_shape_without_axis.begin() + axis);
  int output_axis_dim = input0_shape.at(axis);
  for (size_t i = 1; i < inputs_.size(); ++i) {
    auto shape_tmp = inputs_.at(i)->shape();
    if (shape_tmp.size() != input0_shape.size()) {
      MS_LOG(ERROR) << "All inputs should have the same dim num!";
      return RET_PARAM_INVALID;
    }
    if ((inputs_.at(i)->data_type() != output->data_type()) &&
        !((inputs_.at(i)->data_type() == kNumberTypeFloat16 && output->data_type() == kNumberTypeFloat32) ||
          (inputs_.at(i)->data_type() == kNumberTypeFloat32 && output->data_type() == kNumberTypeFloat16))) {
      MS_LOG(ERROR) << "All inputs should have the same type!";
      return RET_PARAM_INVALID;
    }
    auto axis_tmp = shape_tmp[axis];
    shape_tmp.erase(shape_tmp.begin() + axis);
    if (input0_shape_without_axis != shape_tmp) {
      MS_LOG(ERROR) << "Inputs should have the same dim except axis!";
      return RET_PARAM_INVALID;
    }
    output_axis_dim += axis_tmp;
  }
  auto output_shape = input0_shape;
  output_shape[axis] = output_axis_dim;
  outputs_[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

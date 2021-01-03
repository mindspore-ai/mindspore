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

#include "include/errorcode.h"
#include "src/ops/maximum_grad.h"
#include "src/common/log_adapter.h"
#ifdef PRIMITIVE_WRITEABLE
#include <float.h>
#include "src/param_value_lite.h"
#endif

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int MaximumGrad::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  if (this->primitive_ == nullptr) {
    this->primitive_ = new (std::nothrow) schema::PrimitiveT;
    if (this->primitive_ == nullptr) {
      MS_LOG(ERROR) << "new primitiveT failed";
      return RET_ERROR;
    }
    this->primitive_->value.type = schema::PrimitiveType_MaximumGrad;
  }
  if (this->primitive_->value.type != schema::PrimitiveType_MaximumGrad) {
    MS_LOG(ERROR) << "Primitive type is error :" << this->primitive_->value.type;
    return RET_ERROR;
  }
  if (this->primitive_->value.value == nullptr) {
    auto attr = new (std::nothrow) schema::MaximumGradT();
    if (attr == nullptr) {
      MS_LOG(ERROR) << "new primitiveT value failed";
      return RET_ERROR;
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
int MaximumGrad::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto val_offset = schema::CreateMaximumGrad(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_MaximumGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
PrimitiveC *MaximumGradCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<MaximumGrad>(primitive);
}
Registry MaximumGradRegistry(schema::PrimitiveType_MaximumGrad, MaximumGradCreator);

#endif
int MaximumGrad::InferShape(std::vector<Tensor *> inputs_, std::vector<Tensor *> outputs_) {
  if (inputs_.size() != 3) {
    MS_LOG(ERROR) << "The number of input must be 3";
    return RET_ERROR;
  }
  if (outputs_.size() != 2) {
    MS_LOG(ERROR) << "The number of output must be 2";
    return RET_ERROR;
  }

  auto x1 = inputs_[0];
  auto x2 = inputs_[1];
  auto dy = inputs_[2];
  auto dx1 = outputs_[0];
  auto dx2 = outputs_[1];

  MS_ASSERT(dy != nullptr);
  MS_ASSERT(x1 != nullptr);
  MS_ASSERT(x2 != nullptr);
  MS_ASSERT(dx1 != nullptr);
  MS_ASSERT(dx2 != nullptr);
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }

  auto inShape0 = x1->shape();
  auto inShape1 = x2->shape();
  auto outShape = dy->shape();

  ndim_ = outShape.size();
  x1_shape_.resize(ndim_);
  x2_shape_.resize(ndim_);
  dy_shape_.resize(ndim_);
  auto fillDimNum0 = outShape.size() - inShape0.size();
  auto fillDimNum1 = outShape.size() - inShape1.size();
  int j0 = 0;
  int j1 = 0;
  for (unsigned int i = 0; i < outShape.size(); i++) {
    x1_shape_[i] = (i < fillDimNum0) ? 1 : inShape0[j0++];
    x2_shape_[i] = (i < fillDimNum1) ? 1 : inShape1[j1++];
    dy_shape_[i] = outShape[i];
  }

  dx1->set_shape(x1->shape());
  dx2->set_shape(x2->shape());
  dx1->set_data_type(dy->data_type());
  dx2->set_data_type(dy->data_type());
  dx1->set_format(dy->format());
  dx2->set_format(dy->format());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

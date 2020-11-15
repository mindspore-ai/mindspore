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

#include "src/ops/arithmetic_grad.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

namespace mindspore {
namespace lite {
int ArithmeticGrad::InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) {
  if (inputs_.size() != 3) {
    MS_LOG(ERROR) << "The number of input must be 3";
    return RET_ERROR;
  }
  if (outputs_.size() != 2) {
    MS_LOG(ERROR) << "The number of output must be 2";
    return RET_ERROR;
  }
  auto dy = inputs_[0];
  auto x1 = inputs_[1];
  auto x2 = inputs_[2];
  auto dx1 = outputs_[0];
  auto dx2 = outputs_[1];

  MS_ASSERT(dy != nullptr);
  MS_ASSERT(x1 != nullptr);
  MS_ASSERT(x2 != nullptr);
  MS_ASSERT(dx1 != nullptr);
  MS_ASSERT(dx2 != nullptr);

  if ((Type() == schema::PrimitiveType_MaximumGrad) || (Type() == schema::PrimitiveType_MinimumGrad)) {
    x1 = inputs_[0];
    x2 = inputs_[1];
    dy = inputs_[2];
  }

  auto inShape0 = x1->shape();
  auto inShape1 = x2->shape();
  auto outShape = dy->shape();

  if ((Type() == schema::PrimitiveType_AddGrad) || (Type() == schema::PrimitiveType_SubGrad) ||
      (Type() == schema::PrimitiveType_MaximumGrad) || (Type() == schema::PrimitiveType_MinimumGrad)) {
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
  } else {
    if (dx1->ElementsNum() < dx2->ElementsNum()) {
      ndim_ = inShape1.size();
      x1_shape_.resize(ndim_);
      x2_shape_.resize(ndim_);
      dy_shape_.resize(ndim_);
      auto fillDimNum = inShape1.size() - inShape0.size();  // This will not work for batch!
      int j = 0;
      for (unsigned int i = 0; i < inShape1.size(); i++) {
        if (i < fillDimNum) {
          x2_shape_[i] = 1;
        } else {
          x2_shape_[i] = inShape0[j++];
        }
        x1_shape_[i] = inShape1[i];
        dy_shape_[i] = outShape[i];
      }
    } else if (dx2->ElementsNum() < dx1->ElementsNum()) {  // if (inShape0.size() > inShape1.size())
      ndim_ = inShape0.size();
      x1_shape_.resize(ndim_);
      x2_shape_.resize(ndim_);
      dy_shape_.resize(ndim_);
      broadcasting_ = true;
      int j = 0;
      auto fillDimNum = inShape0.size() - inShape1.size();
      for (unsigned int i = 0; i < inShape0.size(); i++) {
        if (i < fillDimNum) {
          x2_shape_[i] = 1;
        } else {
          x2_shape_[i] = inShape1[j++];
        }
        x1_shape_[i] = inShape0[i];
        dy_shape_[i] = outShape[i];
      }
    } else {
      broadcasting_ = false;
      for (unsigned int i = 0; i < inShape0.size(); i++) {
        x2_shape_[i] = inShape1[i];
        x1_shape_[i] = inShape0[i];
        dy_shape_[i] = outShape[i];
      }
    }
  }

  dx1->set_shape(x1->shape());
  dx2->set_shape(x2->shape());
  dx1->set_data_type(dy->data_type());
  dx2->set_data_type(dy->data_type());
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

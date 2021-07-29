/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/strided_slice_grad.h"
#include <string>
#include <memory>
#include <bitset>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void StridedSliceGrad::Init(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                            int64_t shrink_axis_mask) {
  this->set_begin_mask(begin_mask);
  this->set_end_mask(end_mask);
  this->set_ellipsis_mask(ellipsis_mask);
  this->set_new_axis_mask(new_axis_mask);
  this->set_shrink_axis_mask(shrink_axis_mask);
}

void StridedSliceGrad::set_begin_mask(int64_t begin_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kBeginMask, MakeValue(begin_mask));
}
int64_t StridedSliceGrad::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_end_mask(int64_t end_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kEndMask, MakeValue(end_mask));
}
int64_t StridedSliceGrad::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_ellipsis_mask(int64_t ellipsis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * 8> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask();
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  (void)this->AddAttr(kEllipsisMask, MakeValue(ellipsis_mask));
}
int64_t StridedSliceGrad::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_new_axis_mask(int64_t new_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kNewAxisMask, MakeValue(new_axis_mask));
}
int64_t StridedSliceGrad::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_shrink_axis_mask(int64_t shrink_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kShrinkAxisMask, MakeValue(shrink_axis_mask));
}
int64_t StridedSliceGrad::get_shrink_axis_mask() const {
  auto value_ptr = GetAttr(kShrinkAxisMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameStridedSliceGrad, StridedSliceGrad);
}  // namespace ops
}  // namespace mindspore

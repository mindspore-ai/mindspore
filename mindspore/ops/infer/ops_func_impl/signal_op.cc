/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/signal_op.h"

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
ShapeArray SignalOpFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> SignalOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  if (input_infos[kInputIndex1]->GetType() != kNumberTypeInt64) {
    MS_LOG(EXCEPTION) << "For 'SignalOp', the 'signal_offset' must be Tensor(int64), but got: "
                      << input_infos[kInputIndex2]->GetType();
  }
  if (input_infos[kInputIndex2]->GetType() != kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "For 'SignalOp', the 'signal_value' must be Tensor(int32), but got: "
                      << input_infos[kInputIndex2]->GetType();
  }
  if (input_infos[kInputIndex0]->GetType() != kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "For 'SignalOp', the 'signal' must be Tensor(int32), but got: "
                      << input_infos[kInputIndex0]->GetType();
  }
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace mindspore::ops

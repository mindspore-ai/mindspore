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
#include "infer/ops_func_impl/put_mem_signal.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
ShapeArray PutMemSignalFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape(), input_infos[kInputIndex5]->GetShape()};
}

std::vector<TypeId> PutMemSignalFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  std::vector<TypeId> target_support_dtype{TypeId::kNumberTypeInt32, TypeId::kNumberTypeInt64,
                                           TypeId::kNumberTypeFloat16, TypeId::kNumberTypeBFloat16,
                                           TypeId::kNumberTypeFloat32};
  if (std::find(target_support_dtype.begin(), target_support_dtype.end(), input_infos[kInputIndex0]->GetType()) ==
      target_support_dtype.end()) {
    MS_LOG(EXCEPTION) << "For 'PutMemSignal', the 'target' must be Tensor(int32, int64, float16, bfloat16, float32), "
                         "but got: "
                      << input_infos[kInputIndex0]->GetType();
  }
  if (input_infos[kInputIndex0]->GetType() != input_infos[kInputIndex2]->GetType()) {
    MS_LOG(EXCEPTION) << "For 'PutMemSignal', the 'target' and 'src' must be same dtype, but got: "
                      << input_infos[kInputIndex0]->GetType() << " and " << input_infos[kInputIndex2]->GetType();
  }
  std::vector<size_t> check_indices = {kInputIndex1, kInputIndex3, kInputIndex4, kInputIndex6};
  for (const auto &index : check_indices) {
    if (index < input_infos.size() && input_infos[index]->GetType() != kNumberTypeInt64) {
      MS_LOG(EXCEPTION) << "For 'PutMemSignal', the target_offset, src_offset, "
                        << "size and signal_offset must be int64, but got: " << input_infos[index]->GetType();
    }
  }
  if (input_infos[kInputIndex7]->GetType() != kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "For 'PutMemSignal', the 'signal_value' must be Tensor(int32), but got: "
                      << input_infos[kInputIndex7]->GetType();
  }
  if (input_infos[kInputIndex5]->GetType() != kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "For 'PutMemSignal', the 'signal' must be Tensor(int32), but got: "
                      << input_infos[kInputIndex5]->GetType();
  }
  return {input_infos[kInputIndex0]->GetType(), input_infos[kInputIndex5]->GetType()};
}
}  // namespace mindspore::ops

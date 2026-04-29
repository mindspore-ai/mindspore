/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/searchsorted.h"

#include <tuple>
#include <string>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SearchSortedAscendCustomize(const TensorPtr &sorted_sequence, const TensorPtr &values,
                                              const std::optional<TensorPtr> &sorter, const Int64ImmPtr &dtype,
                                              const BoolImmPtr &right) {
  auto output = InferOutput<1>(prim::kPrimSearchSorted, sorted_sequence, values, sorter, dtype, right);
  // Convert ValuePtr to c++ scalar
  auto in_dtype = GetValue<int64_t>(dtype);
  auto in_right = GetValue<bool>(right);

  TensorToDevice(sorted_sequence, values, sorter);
  TensorAllocate(output);
  auto in_dtype_id = static_cast<TypeId>(in_dtype);
  bool out_int32 = false;
  if (in_dtype_id == kNumberTypeInt32) {
    out_int32 = true;
  }
  DISPATCH_ACLNN(aclnnSearchSorted, sorted_sequence, values, out_int32, in_right, sorter, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

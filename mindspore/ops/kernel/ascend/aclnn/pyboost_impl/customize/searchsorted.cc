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

#include <string>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/common/searchsorted_validate_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr auto kSearchSortedOpName = "SearchSorted";

void CheckSearchSortedAclnnInputs(const TensorPtr &sorted_sequence, const TensorPtr &values,
                                  const std::optional<TensorPtr> &sorter) {
  searchsorted::CheckSearchsortedInputShapes(sorted_sequence->shape(), values->shape(), kSearchSortedOpName);
  if (!sorter.has_value()) {
    return;
  }

  const auto &sorter_tensor = sorter.value();
  MS_EXCEPTION_IF_NULL(sorter_tensor);
  searchsorted::CheckSearchsortedSorterType(static_cast<TypeId>(sorter_tensor->data_type_c()), kSearchSortedOpName);
  searchsorted::CheckSearchsortedSorterShape(sorted_sequence->shape(), sorter_tensor->shape(), kSearchSortedOpName);
}
}  // namespace

tensor::TensorPtr SearchSortedAscendCustomize(const TensorPtr &sorted_sequence, const TensorPtr &values,
                                              const std::optional<TensorPtr> &sorter, const Int64ImmPtr &dtype,
                                              const BoolImmPtr &right) {
  CheckSearchSortedAclnnInputs(sorted_sequence, values, sorter);
  auto output = InferOutput<1>(prim::kPrimSearchSorted, sorted_sequence, values, sorter, dtype, right);
  // Convert ValuePtr to c++ scalar
  auto in_dtype = GetValue<int64_t>(dtype);
  auto in_right = GetValue<bool>(right);

  TensorToDevice(sorted_sequence, values, sorter);
  TensorMalloc(output);
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

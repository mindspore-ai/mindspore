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
#include "view/unstack_ext_view_strides_calc.h"
#include <vector>
#include <memory>
#include <set>
#include <string>
#include "view/unstack_strides_calc.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList UnstackExtViewBasicTypeCalc(const tensor::TensorPtr &x_tensor, const int64_t &dim) {
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto storage_info_list = UnstackStridesCalc(x_tensor->shape(), x_tensor->stride(), x_tensor->storage_info(), dim);
  MS_CHECK_VALUE(storage_info_list.size() > 0, "For 'UnstackExtView', output_num should be greater than 0, but got " +
                                                 std::to_string(storage_info_list.size()));
  return storage_info_list;
}
}  // namespace mindspore::ops

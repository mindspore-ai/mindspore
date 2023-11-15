/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "ops/view/real_strides_calc.h"
#include <vector>
#include <memory>
#include <set>
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
constexpr size_t kRealInputsNum = 1;
TensorStorageInfoPtrList RealCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kRealInputsNum)) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }
  auto tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto input_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> all_types_with_complex = {kBool,    kInt,     kInt8,    kInt16,     kInt32,     kInt64,
                                                    kUInt,    kUInt8,   kUInt16,  kUInt32,    kUInt64,    kFloat,
                                                    kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, all_types_with_complex, prim->name());

  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  if (tensor->data_type() == kNumberTypeComplex64 || tensor->data_type() == kNumberTypeComplex128) {
    size_t num = 2;
    auto new_shape = old_shape;
    auto new_strides = old_strides;
    auto new_storage_offset = old_storage_offset * num;

    for (size_t i = 0; i < new_strides.size(); i++) {
      new_strides[i] *= num;
    }

    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                          old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
    return {new_storage_info};
  } else {
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_storage_offset, old_tensor_info->ori_shape,
                                          old_tensor_info->ori_strides, IsContiguous(old_shape, old_strides));
    return {new_storage_info};
  }
}
REG_VIEW_STRIDES_CALC_FUN(Real, RealCalc);

}  // namespace mindspore::ops

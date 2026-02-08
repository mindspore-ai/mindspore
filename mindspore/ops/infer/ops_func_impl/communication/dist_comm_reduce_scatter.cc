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

#include "infer/ops_func_impl/communication/dist_comm_reduce_scatter.h"
#include <memory>
#include <set>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/communication/op_comm_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommReduceScatterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  auto &value = input_infos[kIndex2];
  auto rank_size = GetRankValue(primitive->name(), value);
  auto output_shape = input_infos[kIndex0]->GetShape();
  output_shape[kIndex0] = output_shape[kIndex0] * rank_size;
  return {output_shape};
}

std::vector<TypeId> DistCommReduceScatterFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const InferInfoPtrList &input_infos) const {
  auto out_type = input_infos[kIndex0]->GetType();
  auto x_list = input_infos[kIndex1]->GetSequenceElements();
  for (size_t i = 1; i < x_list.size(); i++) {
    auto in_type = x_list[i]->GetType();
    CheckInferTypes(primitive->name(), in_type, out_type, true);
  }
  return {out_type};
}
}  // namespace ops
}  // namespace mindspore

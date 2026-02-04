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
#include <set>
#include <memory>
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/nsa_compress_grad.h"

namespace mindspore {
namespace ops {
ShapeArray NsaCompressGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  // grad inputs: grad, input, weight, compress_block_size, compress_stride, actual_seq_len
  // returns: input_grad, weight_grad
  auto &input_tensor = input_infos[kIndex1];   // input
  auto &weight_tensor = input_infos[kIndex2];  // weight

  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();

  return {input_shape, weight_shape};
}

std::vector<TypeId> NsaCompressGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex1]->GetType();   // input
  auto weight_type = input_infos[kIndex2]->GetType();  // weight

  // input_grad has same type as input, weight_grad has same type as weight
  return {input_type, weight_type};
}

}  // namespace ops
}  // namespace mindspore

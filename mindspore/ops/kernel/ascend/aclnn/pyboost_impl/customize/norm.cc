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

#include "kernel/ascend/aclnn/pyboost_impl/customize/norm.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_n.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NormAscendCustomize(const TensorPtr &input_x_tensor, const FP32ImmPtr &p,
                                      const std::optional<ValueTuplePtr> &dim, const BoolImmPtr &keepdim,
                                      const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "Call Norm start";
  // Convert ValuePtr to c++ scalar
  auto output = InferOutput<1>(prim::kPrimNorm, input_x_tensor, p, dim, keepdim, dtype);
  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  ScalarPtr p_scalar = nullptr;
  MAKE_SCALAR(GetValue<float>(p), kNumberTypeFloat32, p_scalar);

  const auto keepdim_imm = GetValue<bool>(keepdim);
  TensorToDevice(input_x_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnNorm, input_x_tensor, p_scalar, dim_vector, keepdim_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

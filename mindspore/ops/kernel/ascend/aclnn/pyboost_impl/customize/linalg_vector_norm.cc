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

#include "kernel/ascend/aclnn/pyboost_impl/customize/linalg_vector_norm.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr LinalgVectorNormAscendCustomize(const TensorPtr &x_tensor, const FP32ImmPtr &ord,
                                                  const std::optional<ValueTuplePtr> &dim, const BoolImmPtr &keepdim,
                                                  const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "Call LinalgVectorNorm start";
  // Convert ValuePtr to c++ scalar
  auto output = InferOutput<1>(prim::kPrimLinalgVectorNorm, x_tensor, ord, dim, keepdim, dtype);
  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  ScalarPtr ord_scalar = nullptr;
  MAKE_SCALAR(GetValue<float>(ord), kNumberTypeFloat32, ord_scalar);
  const auto keepdim_imm = GetValue<bool>(keepdim);

  TypeId out_dtype = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : x_tensor->data_type();
  TensorToDevice(x_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnLinalgVectorNorm, x_tensor, ord_scalar, dim_vector, keepdim_imm, out_dtype, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore

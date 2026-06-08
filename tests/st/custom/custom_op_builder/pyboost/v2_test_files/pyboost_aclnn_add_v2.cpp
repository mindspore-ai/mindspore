/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ms_extension/all.h"

namespace custom {

inline ShapeVector BroadcastShapes(const ShapeVector &a, const ShapeVector &b) {
  size_t na = a.size(), nb = b.size();
  size_t n = std::max(na, nb);
  ShapeVector result(n);
  for (size_t i = 0; i < n; ++i) {
    int64_t da = (i < n - na) ? 1 : a[i - (n - na)];
    int64_t db = (i < n - nb) ? 1 : b[i - (n - nb)];
    if (da != db && da != 1 && db != 1) {
      throw std::runtime_error("Cannot broadcast shapes");
    }
    result[i] = std::max(da, db);
  }
  return result;
}

// Infer: shape = broadcast(x, y), dtype = x.dtype
// Consistent with BinaryOpFuncImpl (binary_op.cc)
ms::Tensor InferAdd(const ms::Tensor &x, const ms::Tensor &y) {
  return ms::Tensor(x.data_type(), BroadcastShapes(x.shape(), y.shape()));
}

// aclnnAdd(input, other, alpha, output): output = input + alpha * other
// alpha: ScalarPtr, passed through via Arg() template fallback (aclnn_op_runner.h:60)
// Ref: mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/add.cc
ms::Tensor npu_add_v2(const ms::Tensor &x, const ms::Tensor &y, double alpha) {
  auto result = InferAdd(x, y);
  mindspore::ScalarPtr alpha_scalar = std::make_shared<mindspore::FP32Imm>(static_cast<float>(alpha));
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(result);
  MS_DISPATCH_ACLNN(aclnnAdd, x, y, alpha_scalar, result);
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_add_v2", PYBOOST_CALLER(1, custom::npu_add_v2)); }

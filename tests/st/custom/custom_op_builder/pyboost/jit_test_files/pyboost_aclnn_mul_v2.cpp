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

// Broadcast shape inference, consistent with CalBroadCastShape (binary_op.cc)
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
ms::Tensor InferMul(const ms::Tensor &x, const ms::Tensor &y) {
  return ms::Tensor(x.data_type(), BroadcastShapes(x.shape(), y.shape()));
}

ms::Tensor npu_mul_v2(const ms::Tensor &x, const ms::Tensor &y) {
  auto result = InferMul(x, y);
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(result);
  MS_DISPATCH_ACLNN(aclnnMul, x, y, result);
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_mul_v2", PYBOOST_CALLER(1, custom::npu_mul_v2)); }

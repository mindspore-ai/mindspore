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

// Infer: shape = input shape, dtype = input dtype
// Consistent with EltwiseOpFuncImpl
ms::Tensor InferRelu(const ms::Tensor &x) {
  return ms::Tensor(x.data_type(), x.shape());
}

ms::Tensor npu_relu_v2(const ms::Tensor &x) {
  auto result = InferRelu(x);
  ms::TensorToDevice(x);
  ms::TensorAllocate(result);
  MS_DISPATCH_ACLNN(aclnnRelu, x, result);
  return result;
}

}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_relu_v2", PYBOOST_CALLER(1, custom::npu_relu_v2)); }

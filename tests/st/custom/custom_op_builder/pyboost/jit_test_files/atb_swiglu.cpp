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

#include "ms_extension/api.h"

namespace atb {
template <>
struct HashOpParam<atb::infer::ActivationParam> {
  void operator()(const atb::infer::ActivationParam &param) const {
    add_param_to_buf("activationType", param.activationType);
    add_param_to_buf("scale", param.scale);
    add_param_to_buf("dim", param.dim);
    add_param_to_buf("geluMode", param.geluMode);
  }
};
}  // namespace atb

ms::Tensor InferSwigluForward(const ms::Tensor &x, int32_t dim) {
  ShapeVector out_tensor_shape(x.shape());
  int64_t split_dim = dim;
  if (split_dim < 0) {
    split_dim += out_tensor_shape.size();
  }
  const int64_t split_num = 2;
  out_tensor_shape[split_dim] /= split_num;
  return ms::Tensor(x.data_type(), out_tensor_shape);
}

ms::Tensor npu_swiglu(const ms::Tensor &x, int32_t dim) {
  auto y = InferSwigluForward(x, dim);

  atb::infer::ActivationParam param;
  param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
  param.dim = dim;

  ms::pynative::RunAtbOp("Swiglu", param, {x}, {y});
  return y;
}

auto pyboost_npu_swiglu(const ms::Tensor &x, int32_t dim) {
  return ms::pynative::PyboostRunner::Call<1>(npu_swiglu, x, dim);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_swiglu", &pyboost_npu_swiglu, "swiglu realization", pybind11::arg("x"), pybind11::arg("dim") = -1);
}

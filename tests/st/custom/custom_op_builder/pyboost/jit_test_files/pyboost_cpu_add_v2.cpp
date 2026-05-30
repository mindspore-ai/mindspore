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

#include <vector>
#include "ms_extension/api.h"

// Stateless custom_v2 pure function — no class inheritance needed.
ms::Tensor add_v2(const ms::Tensor &x, const ms::Tensor &y) {
  auto out = ms::Tensor(x.data_type(), x.shape());
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(out);
  ms::DispatchKernel([x, y, out]() {
    auto *xp = x.data_ptr<int32_t>();
    auto *yp = y.data_ptr<int32_t>();
    auto *op = out.data_ptr<int32_t>();
    for (size_t i = 0; i < x.numel(); i++) {
      op[i] = xp[i] + yp[i];
    }
  });
  return out;
}

ms::Tensor add_scaled_v2(const ms::Tensor &x, const ms::Tensor &y, int64_t scale) {
  auto out = ms::Tensor(x.data_type(), x.shape());
  ms::TensorToDevice(x, scale, y);
  ms::TensorAllocate(out);
  ms::DispatchKernel([x, y, out, scale]() {
    auto *xp = x.data_ptr<int32_t>();
    auto *yp = y.data_ptr<int32_t>();
    auto *op = out.data_ptr<int32_t>();
    auto int_scale = static_cast<int32_t>(scale);
    for (size_t i = 0; i < x.numel(); i++) {
      op[i] = xp[i] + int_scale * yp[i];
    }
  });
  return out;
}

ms::Tensor sum_vector_v2(const std::vector<ms::Tensor> &inputs) {
  auto out = ms::Tensor(inputs[0].data_type(), inputs[0].shape());
  ms::TensorToDevice(inputs);
  ms::TensorAllocate(out);
  ms::DispatchKernel([inputs, out]() {
    auto *op = out.data_ptr<int32_t>();
    for (size_t i = 0; i < out.numel(); i++) {
      int32_t sum = 0;
      for (const auto &input : inputs) {
        sum += input.data_ptr<int32_t>()[i];
      }
      op[i] = sum;
    }
  });
  return out;
}

std::vector<ms::Tensor> add_sub_v2(const ms::Tensor &x, const ms::Tensor &y) {
  std::vector<ms::Tensor> outputs{ms::Tensor(x.data_type(), x.shape()), ms::Tensor(x.data_type(), x.shape())};
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(outputs);
  ms::DispatchKernel([x, y, outputs]() {
    auto *xp = x.data_ptr<int32_t>();
    auto *yp = y.data_ptr<int32_t>();
    auto *add_out = outputs[0].data_ptr<int32_t>();
    auto *sub_out = outputs[1].data_ptr<int32_t>();
    for (size_t i = 0; i < x.numel(); i++) {
      add_out[i] = xp[i] + yp[i];
      sub_out[i] = xp[i] - yp[i];
    }
  });
  return outputs;
}

std::vector<ms::Tensor> add_mul_v2(const ms::Tensor &x, const ms::Tensor &y) {
  auto add_out = ms::Tensor(x.data_type(), x.shape());
  auto mul_out = ms::Tensor(x.data_type(), x.shape());
  ms::TensorToDevice(x, y);
  ms::TensorAllocate(add_out, mul_out);
  ms::DispatchKernel([x, y, add_out, mul_out]() {
    auto *xp = x.data_ptr<int32_t>();
    auto *yp = y.data_ptr<int32_t>();
    auto *add_ptr = add_out.data_ptr<int32_t>();
    auto *mul_ptr = mul_out.data_ptr<int32_t>();
    for (size_t i = 0; i < x.numel(); i++) {
      add_ptr[i] = xp[i] + yp[i];
      mul_ptr[i] = xp[i] * yp[i];
    }
  });
  return {add_out, mul_out};
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("add_v2", PYBOOST_CALLER(1, add_v2));
  m.def("add_scaled_v2", PYBOOST_CALLER(1, add_scaled_v2));
  m.def("sum_vector_v2", PYBOOST_CALLER(1, sum_vector_v2));
  m.def("add_sub_v2", PYBOOST_CALLER(2, add_sub_v2));
  m.def("add_mul_v2", PYBOOST_CALLER(2, add_mul_v2));
}

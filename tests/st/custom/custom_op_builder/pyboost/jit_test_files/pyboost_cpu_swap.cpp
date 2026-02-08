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

class CustomSwap : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;

  size_t CalcWorkspace() override { return inputs()[0].numel() * sizeof(int32_t); }

  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    int32_t *x_base_ptr = static_cast<int32_t *>(x.GetDataPtr());
    int32_t *y_base_ptr = static_cast<int32_t *>(y.GetDataPtr());
    int32_t *ws_base_ptr = static_cast<int32_t *>(workspace_ptr());
    for (size_t i = 0; i < x.numel(); i++) {
      ws_base_ptr[i] = x_base_ptr[i];
    }
    for (size_t i = 0; i < x.numel(); i++) {
      x_base_ptr[i] = y_base_ptr[i];
    }
    for (size_t i = 0; i < x.numel(); i++) {
      y_base_ptr[i] = ws_base_ptr[i];
    }
  }
};

void custom_swap(const ms::Tensor &x, const ms::Tensor &y) {
  std::make_shared<CustomSwap>("Swap")->Run({x, y}, {x, y});
}

auto pyboost_swap(const ms::Tensor &x, const ms::Tensor &y) {
  return ms::pynative::PyboostRunner::Call<0>(custom_swap, x, y);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("swap", &pyboost_swap, "swap x and y", pybind11::arg("x"), pybind11::arg("y"));
}

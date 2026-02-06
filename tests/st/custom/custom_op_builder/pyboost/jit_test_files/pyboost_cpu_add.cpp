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

class CustomAddUncontiguous : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    auto &z = outputs()[0];

    // Create TensorAccessors for efficient data access
    auto x_accessor = x.accessor<int32_t, 3>();
    auto y_accessor = y.accessor<int32_t, 3>();
    auto z_accessor = z.accessor<int32_t, 3>();

    // Get shape of the tensors (assume 3D tensors)
    const auto &x_shape = x.shape();

    // Iterate through the tensor elements using TensorAccessor with bracket access
    for (int64_t i = 0; i < x_shape[0]; ++i) {
      for (int64_t j = 0; j < x_shape[1]; ++j) {
        for (int64_t k = 0; k < x_shape[2]; ++k) {
          // the TensorAccessor support [] and () indices.
          z_accessor(i, j, k) = x_accessor[i][j][k] + y_accessor[i][j][k];
        }
      }
    }
  }
};

ms::Tensor add_uncontiguous(const ms::Tensor &x, const ms::Tensor &y) {
  // assume the shape of x and y is same
  auto out = ms::Tensor(x.data_type(), x.shape());
  if (x.is_contiguous() || y.is_contiguous()) {
    throw std::invalid_argument("For add_uncontiguous, the inputs should be uncontiguous tensor.");
  }
  auto runner = std::make_shared<CustomAddUncontiguous>("Add1");
  runner->Run({x, y}, {out});
  return out;
}

auto pyboost_add1(const ms::Tensor &x, const ms::Tensor &y) {
  x.SetNeedContiguous(false);
  y.SetNeedContiguous(false);
  return ms::pynative::PyboostRunner::Call<1>(add_uncontiguous, x, y);
}

class CustomAddContiguous : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    auto &z = outputs()[0];
    const int32_t *x_base_ptr = static_cast<const int32_t *>(x.GetDataPtr());
    const int32_t *y_base_ptr = static_cast<const int32_t *>(y.GetDataPtr());
    int32_t *z_base_ptr = static_cast<int32_t *>(z.GetDataPtr());
    for (size_t i = 0; i < x.numel(); i++) {
      z_base_ptr[i] = x_base_ptr[i] + y_base_ptr[i];
    }
  }
};

ms::Tensor add_contiguous(const ms::Tensor &x, const ms::Tensor &y) {
  // assume the shape of x and y is same
  auto out = ms::Tensor(x.data_type(), x.shape());
  if (!x.is_contiguous() || !y.is_contiguous()) {
    throw std::invalid_argument("For add_contiguous, the inputs should be contiguous tensor.");
  }
  auto runner = std::make_shared<CustomAddContiguous>("Add2");
  runner->Run({x, y}, {out});
  return out;
}

auto pyboost_add2(const ms::Tensor &x, const ms::Tensor &y) {
  return ms::pynative::PyboostRunner::Call<1>(add_contiguous, x, y);
}

class CustomAdd3 : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  CustomAdd3(const std::string &name) : PyboostRunner(name) {}
  size_t CalcWorkspace() override { return inputs()[0].numel() * sizeof(int32_t); }
  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    auto &z = inputs()[2];
    auto &out = outputs()[0];
    const int32_t *x_base_ptr = static_cast<const int32_t *>(x.GetDataPtr());
    const int32_t *y_base_ptr = static_cast<const int32_t *>(y.GetDataPtr());
    const int32_t *z_base_ptr = static_cast<const int32_t *>(z.GetDataPtr());
    int32_t *ws_base_ptr = static_cast<int32_t *>(workspace_ptr());
    int32_t *out_base_ptr = static_cast<int32_t *>(out.GetDataPtr());
    for (size_t i = 0; i < x.numel(); i++) {
      ws_base_ptr[i] = x_base_ptr[i] + y_base_ptr[i];
    }
    for (size_t i = 0; i < x.numel(); i++) {
      out_base_ptr[i] = z_base_ptr[i] + ws_base_ptr[i];
    }
  }
  static ms::Tensor Eval(const ms::Tensor &x, const ms::Tensor &y, const ms::Tensor &z) {
    // assume the shapes of x, y and z are same.
    auto out = ms::Tensor(x.data_type(), x.shape());
    auto runner = std::make_shared<CustomAdd3>("Add3");
    runner->Run({x, y, z}, {out});
    return out;
  }
};

class CustomAdd4 : public CustomAdd3 {
 public:
  CustomAdd4(const std::string &name) : CustomAdd3(name) {}
  static ms::Tensor Eval(const ms::Tensor &x, const ms::Tensor &y, const std::vector<ms::Tensor> &z) {
    // assume the shapes of x, y and z are same.
    auto out = ms::Tensor(x.data_type(), x.shape());
    auto runner = std::make_shared<CustomAdd4>("Add4");
    runner->Run({x, y, z[0]}, {out});
    return out;
  }
};

class CustomAdd5 : public CustomAdd3 {
 public:
  CustomAdd5(const std::string &name) : CustomAdd3(name) {}
  static ms::Tensor Eval(const ms::Tensor &x, const ms::Tensor &y, const std::vector<std::vector<ms::Tensor>> &z) {
    // assume the shapes of x, y and z are same.
    auto out = ms::Tensor(x.data_type(), x.shape());
    auto runner = std::make_shared<CustomAdd5>("Add5");
    runner->Run({x, y, z[0][0]}, {out});
    return out;
  }
};

class CustomAdd6 : public CustomAdd3 {
 public:
  CustomAdd6(const std::string &name) : CustomAdd3(name) {}
  static void Eval(const ms::Tensor &x, const ms::Tensor &y, const ms::Tensor &z, const std::vector<ms::Tensor> &out) {
    auto runner = std::make_shared<CustomAdd5>("Add6");
    runner->Run({x, y, z}, {out[0]});
  }
};

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("add_uncontiguous", &pyboost_add1, "add, support uncontiguous", pybind11::arg("x"), pybind11::arg("y"));
  m.def("add_contiguous", &pyboost_add2, "add, only support contiguous", pybind11::arg("x"), pybind11::arg("y"));
  m.def("add3", PYBOOST_CALLER(1, CustomAdd3::Eval));
  m.def("add4", PYBOOST_CALLER(1, CustomAdd4::Eval));
  m.def("add5", PYBOOST_CALLER(1, CustomAdd5::Eval));
  m.def("add6", PYBOOST_CALLER(0, CustomAdd6::Eval));
}

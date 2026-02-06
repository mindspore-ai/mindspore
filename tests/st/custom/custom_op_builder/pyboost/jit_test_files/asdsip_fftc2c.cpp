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

ms::Tensor InferFFTForward(const ms::Tensor &input) {
  ShapeVector out_tensor_shape(input.shape());
  return ms::Tensor(input.data_type(), out_tensor_shape);
}

ms::Tensor npu_fft_1d(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  ms::pynative::FFTParam param;
  param.fftXSize = n;
  param.fftYSize = 0;
  param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
  param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
  param.batchSize = batch_size;
  param.dimType = ms::pynative::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
  auto output = InferFFTForward(input);
  ms::pynative::RunAsdSipFFTOp("asdFftExecC2C", param, input, output);
  return output;
}

ms::Tensor npu_fft_2d(const ms::Tensor &input, int64_t x_size, int64_t y_size, int64_t batch_size) {
  ms::pynative::FFTParam param;
  param.fftXSize = x_size;
  param.fftYSize = y_size;
  param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
  param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
  param.batchSize = batch_size;
  auto output = InferFFTForward(input);
  ms::pynative::RunAsdSipFFTOp("asdFftExecC2C", param, input, output);
  return output;
}

ms::Tensor wrong_npu_fft_1d(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  ms::pynative::FFTParam param;
  param.fftXSize = n;
  param.fftYSize = 0;
  param.fftType = ms::pynative::asdFftType::ASCEND_FFT_C2C;
  param.direction = ms::pynative::asdFftDirection::ASCEND_FFT_FORWARD;
  param.batchSize = batch_size;
  param.dimType = ms::pynative::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;
  auto output = InferFFTForward(input);
  // asdfftexecc2c is wrong, use "asdFftExecC2C" instead.
  ms::pynative::RunAsdSipFFTOp("asdfftexecc2c", param, input, output);
  return output;
}

auto pyboost_npu_fft_1d(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(npu_fft_1d, input, n, batch_size);
}

auto pyboost_npu_fft_2d(const ms::Tensor &input, int64_t x_size, int64_t y_size, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(npu_fft_2d, input, x_size, y_size, batch_size);
}

auto wrong_pyboost_npu_fft_1d(const ms::Tensor &input, int64_t n, int64_t batch_size) {
  return ms::pynative::PyboostRunner::Call<1>(wrong_npu_fft_1d, input, n, batch_size);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("fft_1d", &pyboost_npu_fft_1d, "FFT C2C", pybind11::arg("input"), pybind11::arg("n"),
        pybind11::arg("batch_size"));
  m.def("fft_2d", &pyboost_npu_fft_2d, "FFT C2C", pybind11::arg("input"), pybind11::arg("x_size"),
        pybind11::arg("y_size"), pybind11::arg("batch_size"));
  m.def("wrong_fft_1d", &wrong_pyboost_npu_fft_1d, "Wrong FFT C2C", pybind11::arg("input"), pybind11::arg("n"),
        pybind11::arg("batch_size"));
}

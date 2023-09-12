/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/cast_kernels.h"
#include <map>
#include <complex>
#include <vector>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "./kernel_log.h"
#include "./kernel_errcode.h"
#include "proto/node_def.pb.h"
#include "common/tensor.h"
#include "proto/attr.pb.h"
namespace aicpu {

template <typename T, typename S>
uint32_t CastTask(const std::vector<uintptr_t> &ioAddrs, const size_t &input_size) {
  if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
    for (size_t i = 0; i < input_size; i++) {
      auto *input_addr = reinterpret_cast<T *>(ioAddrs[0]);
      auto *output_addr = reinterpret_cast<S *>(ioAddrs[1]);
      output_addr[i] = static_cast<S>(std::real(input_addr[i]));
    }
    return kAicpuKernelStateSucess;
  } else {
    constexpr int kRank = 2;
    Eigen::TensorMap<Eigen::Tensor<T, kRank, Eigen::RowMajor>> input_map(reinterpret_cast<T *>(ioAddrs[0]), 1,
                                                                         input_size);
    const auto &input = Eigen::Tensor<T, kRank, Eigen::RowMajor>(input_map);
    Eigen::TensorMap<Eigen::Tensor<S, kRank, Eigen::RowMajor>>(reinterpret_cast<S *>(ioAddrs[1]), 1, input_size) =
      input.template cast<S>();
    return kAicpuKernelStateSucess;
  }
}

uint32_t CastKernel::DoCompute() {
  std::map<int, std::map<int, std::function<uint32_t(std::vector<uintptr_t> &, size_t &)>>> calls;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_INT8] = CastTask<int8_t, int8_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_INT16] = CastTask<int8_t, int16_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_INT32] = CastTask<int8_t, int32_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_INT64] = CastTask<int8_t, int64_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_FLOAT16] = CastTask<int8_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_FLOAT32] = CastTask<int8_t, float>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_FLOAT64] = CastTask<int8_t, double>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_UINT8] = CastTask<int8_t, uint8_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_UINT16] = CastTask<int8_t, uint16_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_UINT32] = CastTask<int8_t, uint32_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_UINT64] = CastTask<int8_t, uint64_t>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_BOOL] = CastTask<int8_t, bool>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<int8_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_INT8][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<int8_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_INT8] = CastTask<int16_t, int8_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_INT16] = CastTask<int16_t, int16_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_INT32] = CastTask<int16_t, int32_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_INT64] = CastTask<int16_t, int64_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_FLOAT16] = CastTask<int16_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_FLOAT32] = CastTask<int16_t, float>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_FLOAT64] = CastTask<int16_t, double>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_UINT8] = CastTask<int16_t, uint8_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_UINT16] = CastTask<int16_t, uint16_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_UINT32] = CastTask<int16_t, uint32_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_UINT64] = CastTask<int16_t, uint64_t>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_BOOL] = CastTask<int16_t, bool>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<int16_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_INT16][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<int16_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_INT8] = CastTask<int32_t, int8_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_INT16] = CastTask<int32_t, int16_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_INT32] = CastTask<int32_t, int32_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_INT64] = CastTask<int32_t, int64_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_FLOAT16] = CastTask<int32_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_FLOAT32] = CastTask<int32_t, float>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_FLOAT64] = CastTask<int32_t, double>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_UINT8] = CastTask<int32_t, uint8_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_UINT16] = CastTask<int32_t, uint16_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_UINT32] = CastTask<int32_t, uint32_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_UINT64] = CastTask<int32_t, uint64_t>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_BOOL] = CastTask<int32_t, bool>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<int32_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_INT32][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<int32_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_INT8] = CastTask<int64_t, int8_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_INT16] = CastTask<int64_t, int16_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_INT32] = CastTask<int64_t, int32_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_INT64] = CastTask<int64_t, int64_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_FLOAT16] = CastTask<int64_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_FLOAT32] = CastTask<int64_t, float>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_FLOAT64] = CastTask<int64_t, double>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_UINT8] = CastTask<int64_t, uint8_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_UINT16] = CastTask<int64_t, uint16_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_UINT32] = CastTask<int64_t, uint32_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_UINT64] = CastTask<int64_t, uint64_t>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_BOOL] = CastTask<int64_t, bool>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<int64_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_INT64][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<int64_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_INT8] = CastTask<Eigen::half, int8_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_INT16] = CastTask<Eigen::half, int16_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_INT32] = CastTask<Eigen::half, int32_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_INT64] = CastTask<Eigen::half, int64_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_FLOAT16] = CastTask<Eigen::half, Eigen::half>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_FLOAT32] = CastTask<Eigen::half, float>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_FLOAT64] = CastTask<Eigen::half, double>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_UINT8] = CastTask<Eigen::half, uint8_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_UINT16] = CastTask<Eigen::half, uint16_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_UINT32] = CastTask<Eigen::half, uint32_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_UINT64] = CastTask<Eigen::half, uint64_t>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_BOOL] = CastTask<Eigen::half, bool>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<Eigen::half, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_FLOAT16][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<Eigen::half, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_INT8] = CastTask<float, int8_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_INT16] = CastTask<float, int16_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_INT32] = CastTask<float, int32_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_INT64] = CastTask<float, int64_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_FLOAT16] = CastTask<float, Eigen::half>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_FLOAT32] = CastTask<float, float>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_FLOAT64] = CastTask<float, double>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_UINT8] = CastTask<float, uint8_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_UINT16] = CastTask<float, uint16_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_UINT32] = CastTask<float, uint32_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_UINT64] = CastTask<float, uint64_t>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_BOOL] = CastTask<float, bool>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<float, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_FLOAT32][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<float, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_INT8] = CastTask<double, int8_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_INT16] = CastTask<double, int16_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_INT32] = CastTask<double, int32_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_INT64] = CastTask<double, int64_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_FLOAT16] = CastTask<double, Eigen::half>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_FLOAT32] = CastTask<double, float>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_FLOAT64] = CastTask<double, double>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_UINT8] = CastTask<double, uint8_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_UINT16] = CastTask<double, uint16_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_UINT32] = CastTask<double, uint32_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_UINT64] = CastTask<double, uint64_t>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_BOOL] = CastTask<double, bool>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<double, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_FLOAT64][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<double, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_INT8] = CastTask<uint8_t, int8_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_INT16] = CastTask<uint8_t, int16_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_INT32] = CastTask<uint8_t, int32_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_INT64] = CastTask<uint8_t, int64_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_FLOAT16] = CastTask<uint8_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_FLOAT32] = CastTask<uint8_t, float>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_FLOAT64] = CastTask<uint8_t, double>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_UINT8] = CastTask<uint8_t, uint8_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_UINT16] = CastTask<uint8_t, uint16_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_UINT32] = CastTask<uint8_t, uint32_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_UINT64] = CastTask<uint8_t, uint64_t>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_BOOL] = CastTask<uint8_t, bool>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<uint8_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_UINT8][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<uint8_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_INT8] = CastTask<uint16_t, int8_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_INT16] = CastTask<uint16_t, int16_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_INT32] = CastTask<uint16_t, int32_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_INT64] = CastTask<uint16_t, int64_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_FLOAT16] = CastTask<uint16_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_FLOAT32] = CastTask<uint16_t, float>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_FLOAT64] = CastTask<uint16_t, double>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_UINT8] = CastTask<uint16_t, uint8_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_UINT16] = CastTask<uint16_t, uint16_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_UINT32] = CastTask<uint16_t, uint32_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_UINT64] = CastTask<uint16_t, uint64_t>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_BOOL] = CastTask<uint16_t, bool>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<uint16_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_UINT16][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<uint16_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_INT8] = CastTask<uint32_t, int8_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_INT16] = CastTask<uint32_t, int16_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_INT32] = CastTask<uint32_t, int32_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_INT64] = CastTask<uint32_t, int64_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_FLOAT16] = CastTask<uint32_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_FLOAT32] = CastTask<uint32_t, float>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_FLOAT64] = CastTask<uint32_t, double>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_UINT8] = CastTask<uint32_t, uint8_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_UINT16] = CastTask<uint32_t, uint16_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_UINT32] = CastTask<uint32_t, uint32_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_UINT64] = CastTask<uint32_t, uint64_t>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_BOOL] = CastTask<uint32_t, bool>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<int32_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_UINT32][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<int32_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_INT8] = CastTask<uint64_t, int8_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_INT16] = CastTask<uint64_t, int16_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_INT32] = CastTask<uint64_t, int32_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_INT64] = CastTask<uint64_t, int64_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_FLOAT16] = CastTask<uint64_t, Eigen::half>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_FLOAT32] = CastTask<uint64_t, float>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_FLOAT64] = CastTask<uint64_t, double>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_UINT8] = CastTask<uint64_t, uint8_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_UINT16] = CastTask<uint64_t, uint16_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_UINT32] = CastTask<uint64_t, uint32_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_UINT64] = CastTask<uint64_t, uint64_t>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_BOOL] = CastTask<uint64_t, bool>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<uint64_t, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_UINT64][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<uint64_t, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_INT8] = CastTask<bool, int8_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_INT16] = CastTask<bool, int16_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_INT32] = CastTask<bool, int32_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_INT64] = CastTask<bool, int64_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_FLOAT16] = CastTask<bool, Eigen::half>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_FLOAT32] = CastTask<bool, float>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_FLOAT64] = CastTask<bool, double>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_UINT8] = CastTask<bool, uint8_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_UINT16] = CastTask<bool, uint16_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_UINT32] = CastTask<bool, uint32_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_UINT64] = CastTask<bool, uint64_t>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_BOOL] = CastTask<bool, bool>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_COMPLEX64] = CastTask<bool, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_BOOL][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<bool, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_INT8] =
    CastTask<std::complex<std::float_t>, int8_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_INT16] =
    CastTask<std::complex<std::float_t>, int16_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_INT32] =
    CastTask<std::complex<std::float_t>, int32_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_INT64] =
    CastTask<std::complex<std::float_t>, int64_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_FLOAT16] =
    CastTask<std::complex<std::float_t>, Eigen::half>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_FLOAT32] =
    CastTask<std::complex<std::float_t>, float>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_FLOAT64] =
    CastTask<std::complex<std::float_t>, double>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_UINT8] =
    CastTask<std::complex<std::float_t>, uint8_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_UINT16] =
    CastTask<std::complex<std::float_t>, uint16_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_UINT32] =
    CastTask<std::complex<std::float_t>, uint32_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_UINT64] =
    CastTask<std::complex<std::float_t>, uint64_t>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_BOOL] = CastTask<std::complex<std::float_t>, bool>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<std::complex<std::float_t>, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_COMPLEX64][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<std::complex<std::float_t>, std::complex<std::double_t>>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_INT8] =
    CastTask<std::complex<std::double_t>, int8_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_INT16] =
    CastTask<std::complex<std::double_t>, int16_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_INT32] =
    CastTask<std::complex<std::double_t>, int32_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_INT64] =
    CastTask<std::complex<std::double_t>, int64_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_FLOAT16] =
    CastTask<std::complex<std::double_t>, Eigen::half>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_FLOAT32] =
    CastTask<std::complex<std::double_t>, float>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_FLOAT64] =
    CastTask<std::complex<std::double_t>, double>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_UINT8] =
    CastTask<std::complex<std::double_t>, uint8_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_UINT16] =
    CastTask<std::complex<std::double_t>, uint16_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_UINT32] =
    CastTask<std::complex<std::double_t>, uint32_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_UINT64] =
    CastTask<std::complex<std::double_t>, uint64_t>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_BOOL] =
    CastTask<std::complex<std::double_t>, bool>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_COMPLEX64] =
    CastTask<std::complex<std::double_t>, std::complex<std::float_t>>;
  calls[::aicpuops::DataType::MS_COMPLEX128][::aicpuops::DataType::MS_COMPLEX128] =
    CastTask<std::complex<std::double_t>, std::complex<std::double_t>>;
  return calls[input_type_][output_type_](io_addrs_, input_size_);
}

uint32_t CastKernel::ParseKernelParam() {
  aicpuops::Tensor output_tensor = node_def_.outputs(0);
  output_type_ = static_cast<::aicpuops::DataType>(output_tensor.tensor_type());

  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  input_type_ = static_cast<::aicpuops::DataType>(input_tensor.tensor_type());

  aicpuops::TensorShape input_shape = input_tensor.tensor_shape();
  input_size_ = 1;
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    input_size_ *= input_shape.dim(i).size();
  }

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t Cast(void *param) {
  aicpu::CastKernel castKernel;
  return castKernel.Compute(param);
}
}

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

#include "cast.h"
#include <map>
#include <complex>
#include <vector>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "inc/kernel_log.h"
namespace aicpu {
namespace {
const char *kCast = "Cast";
}

template <typename T, typename S>
uint32_t CastTask(CpuKernelContext &ctx) {
  auto input = ctx.Input(0);
  auto input_size = input->NumElements();
  auto output = ctx.Output(0);
  auto *input_addr = reinterpret_cast<T *>(input->GetData());
  auto *output_addr = reinterpret_cast<S *>(output->GetData());
  if constexpr ((std::is_same_v<T, std::complex<float>>) || (std::is_same_v<T, std::complex<double>>)) {
    for (int i = 0; i < input_size; i++) {
      output_addr[i] = static_cast<S>(std::real(input_addr[i]));
    }
    return KERNEL_STATUS_OK;
  } else {
    constexpr int kRank = 2;
    Eigen::TensorMap<Eigen::Tensor<T, kRank, Eigen::RowMajor>> input_map(input_addr, 1, input_size);
    const auto &input = Eigen::Tensor<T, kRank, Eigen::RowMajor>(input_map);
    Eigen::TensorMap<Eigen::Tensor<S, kRank, Eigen::RowMajor>>(output_addr, 1, input_size) = input.template cast<S>();
    return KERNEL_STATUS_OK;
  }
}

uint32_t CastKernel::Compute(CpuKernelContext &ctx) {
  RETURN_IF_FAILURE(ParseKernelParam(ctx));
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &)>>> calls;
  calls[DT_INT8][DT_INT8] = CastTask<int8_t, int8_t>;
  calls[DT_INT8][DT_INT16] = CastTask<int8_t, int16_t>;
  calls[DT_INT8][DT_INT32] = CastTask<int8_t, int32_t>;
  calls[DT_INT8][DT_INT64] = CastTask<int8_t, int64_t>;
  calls[DT_INT8][DT_FLOAT16] = CastTask<int8_t, Eigen::half>;
  calls[DT_INT8][DT_FLOAT] = CastTask<int8_t, float>;
  calls[DT_INT8][DT_DOUBLE] = CastTask<int8_t, double>;
  calls[DT_INT8][DT_UINT8] = CastTask<int8_t, uint8_t>;
  calls[DT_INT8][DT_UINT16] = CastTask<int8_t, uint16_t>;
  calls[DT_INT8][DT_UINT32] = CastTask<int8_t, uint32_t>;
  calls[DT_INT8][DT_UINT64] = CastTask<int8_t, uint64_t>;
  calls[DT_INT8][DT_BOOL] = CastTask<int8_t, bool>;
  calls[DT_INT8][DT_COMPLEX64] = CastTask<int8_t, std::complex<std::float_t>>;
  calls[DT_INT8][DT_COMPLEX128] = CastTask<int8_t, std::complex<std::double_t>>;
  calls[DT_INT16][DT_INT8] = CastTask<int16_t, int8_t>;
  calls[DT_INT16][DT_INT16] = CastTask<int16_t, int16_t>;
  calls[DT_INT16][DT_INT32] = CastTask<int16_t, int32_t>;
  calls[DT_INT16][DT_INT64] = CastTask<int16_t, int64_t>;
  calls[DT_INT16][DT_FLOAT16] = CastTask<int16_t, Eigen::half>;
  calls[DT_INT16][DT_FLOAT] = CastTask<int16_t, float>;
  calls[DT_INT16][DT_DOUBLE] = CastTask<int16_t, double>;
  calls[DT_INT16][DT_UINT8] = CastTask<int16_t, uint8_t>;
  calls[DT_INT16][DT_UINT16] = CastTask<int16_t, uint16_t>;
  calls[DT_INT16][DT_UINT32] = CastTask<int16_t, uint32_t>;
  calls[DT_INT16][DT_UINT64] = CastTask<int16_t, uint64_t>;
  calls[DT_INT16][DT_BOOL] = CastTask<int16_t, bool>;
  calls[DT_INT16][DT_COMPLEX64] = CastTask<int16_t, std::complex<std::float_t>>;
  calls[DT_INT16][DT_COMPLEX128] = CastTask<int16_t, std::complex<std::double_t>>;
  calls[DT_INT32][DT_INT8] = CastTask<int32_t, int8_t>;
  calls[DT_INT32][DT_INT16] = CastTask<int32_t, int16_t>;
  calls[DT_INT32][DT_INT32] = CastTask<int32_t, int32_t>;
  calls[DT_INT32][DT_INT64] = CastTask<int32_t, int64_t>;
  calls[DT_INT32][DT_FLOAT16] = CastTask<int32_t, Eigen::half>;
  calls[DT_INT32][DT_FLOAT] = CastTask<int32_t, float>;
  calls[DT_INT32][DT_DOUBLE] = CastTask<int32_t, double>;
  calls[DT_INT32][DT_UINT8] = CastTask<int32_t, uint8_t>;
  calls[DT_INT32][DT_UINT16] = CastTask<int32_t, uint16_t>;
  calls[DT_INT32][DT_UINT32] = CastTask<int32_t, uint32_t>;
  calls[DT_INT32][DT_UINT64] = CastTask<int32_t, uint64_t>;
  calls[DT_INT32][DT_BOOL] = CastTask<int32_t, bool>;
  calls[DT_INT32][DT_COMPLEX64] = CastTask<int32_t, std::complex<std::float_t>>;
  calls[DT_INT32][DT_COMPLEX128] = CastTask<int32_t, std::complex<std::double_t>>;
  calls[DT_INT64][DT_INT8] = CastTask<int64_t, int8_t>;
  calls[DT_INT64][DT_INT16] = CastTask<int64_t, int16_t>;
  calls[DT_INT64][DT_INT32] = CastTask<int64_t, int32_t>;
  calls[DT_INT64][DT_INT64] = CastTask<int64_t, int64_t>;
  calls[DT_INT64][DT_FLOAT16] = CastTask<int64_t, Eigen::half>;
  calls[DT_INT64][DT_FLOAT] = CastTask<int64_t, float>;
  calls[DT_INT64][DT_DOUBLE] = CastTask<int64_t, double>;
  calls[DT_INT64][DT_UINT8] = CastTask<int64_t, uint8_t>;
  calls[DT_INT64][DT_UINT16] = CastTask<int64_t, uint16_t>;
  calls[DT_INT64][DT_UINT32] = CastTask<int64_t, uint32_t>;
  calls[DT_INT64][DT_UINT64] = CastTask<int64_t, uint64_t>;
  calls[DT_INT64][DT_BOOL] = CastTask<int64_t, bool>;
  calls[DT_INT64][DT_COMPLEX64] = CastTask<int64_t, std::complex<std::float_t>>;
  calls[DT_INT64][DT_COMPLEX128] = CastTask<int64_t, std::complex<std::double_t>>;
  calls[DT_FLOAT16][DT_INT8] = CastTask<Eigen::half, int8_t>;
  calls[DT_FLOAT16][DT_INT16] = CastTask<Eigen::half, int16_t>;
  calls[DT_FLOAT16][DT_INT32] = CastTask<Eigen::half, int32_t>;
  calls[DT_FLOAT16][DT_INT64] = CastTask<Eigen::half, int64_t>;
  calls[DT_FLOAT16][DT_FLOAT16] = CastTask<Eigen::half, Eigen::half>;
  calls[DT_FLOAT16][DT_FLOAT] = CastTask<Eigen::half, float>;
  calls[DT_FLOAT16][DT_DOUBLE] = CastTask<Eigen::half, double>;
  calls[DT_FLOAT16][DT_UINT8] = CastTask<Eigen::half, uint8_t>;
  calls[DT_FLOAT16][DT_UINT16] = CastTask<Eigen::half, uint16_t>;
  calls[DT_FLOAT16][DT_UINT32] = CastTask<Eigen::half, uint32_t>;
  calls[DT_FLOAT16][DT_UINT64] = CastTask<Eigen::half, uint64_t>;
  calls[DT_FLOAT16][DT_BOOL] = CastTask<Eigen::half, bool>;
  calls[DT_FLOAT16][DT_COMPLEX64] = CastTask<Eigen::half, std::complex<std::float_t>>;
  calls[DT_FLOAT16][DT_COMPLEX128] = CastTask<Eigen::half, std::complex<std::double_t>>;
  calls[DT_FLOAT][DT_INT8] = CastTask<float, int8_t>;
  calls[DT_FLOAT][DT_INT16] = CastTask<float, int16_t>;
  calls[DT_FLOAT][DT_INT32] = CastTask<float, int32_t>;
  calls[DT_FLOAT][DT_INT64] = CastTask<float, int64_t>;
  calls[DT_FLOAT][DT_FLOAT16] = CastTask<float, Eigen::half>;
  calls[DT_FLOAT][DT_FLOAT] = CastTask<float, float>;
  calls[DT_FLOAT][DT_DOUBLE] = CastTask<float, double>;
  calls[DT_FLOAT][DT_UINT8] = CastTask<float, uint8_t>;
  calls[DT_FLOAT][DT_UINT16] = CastTask<float, uint16_t>;
  calls[DT_FLOAT][DT_UINT32] = CastTask<float, uint32_t>;
  calls[DT_FLOAT][DT_UINT64] = CastTask<float, uint64_t>;
  calls[DT_FLOAT][DT_BOOL] = CastTask<float, bool>;
  calls[DT_FLOAT][DT_COMPLEX64] = CastTask<float, std::complex<std::float_t>>;
  calls[DT_FLOAT][DT_COMPLEX128] = CastTask<float, std::complex<std::double_t>>;
  calls[DT_DOUBLE][DT_INT8] = CastTask<double, int8_t>;
  calls[DT_DOUBLE][DT_INT16] = CastTask<double, int16_t>;
  calls[DT_DOUBLE][DT_INT32] = CastTask<double, int32_t>;
  calls[DT_DOUBLE][DT_INT64] = CastTask<double, int64_t>;
  calls[DT_DOUBLE][DT_FLOAT16] = CastTask<double, Eigen::half>;
  calls[DT_DOUBLE][DT_FLOAT] = CastTask<double, float>;
  calls[DT_DOUBLE][DT_DOUBLE] = CastTask<double, double>;
  calls[DT_DOUBLE][DT_UINT8] = CastTask<double, uint8_t>;
  calls[DT_DOUBLE][DT_UINT16] = CastTask<double, uint16_t>;
  calls[DT_DOUBLE][DT_UINT32] = CastTask<double, uint32_t>;
  calls[DT_DOUBLE][DT_UINT64] = CastTask<double, uint64_t>;
  calls[DT_DOUBLE][DT_BOOL] = CastTask<double, bool>;
  calls[DT_DOUBLE][DT_COMPLEX64] = CastTask<double, std::complex<std::float_t>>;
  calls[DT_DOUBLE][DT_COMPLEX128] = CastTask<double, std::complex<std::double_t>>;
  calls[DT_UINT8][DT_INT8] = CastTask<uint8_t, int8_t>;
  calls[DT_UINT8][DT_INT16] = CastTask<uint8_t, int16_t>;
  calls[DT_UINT8][DT_INT32] = CastTask<uint8_t, int32_t>;
  calls[DT_UINT8][DT_INT64] = CastTask<uint8_t, int64_t>;
  calls[DT_UINT8][DT_FLOAT16] = CastTask<uint8_t, Eigen::half>;
  calls[DT_UINT8][DT_FLOAT] = CastTask<uint8_t, float>;
  calls[DT_UINT8][DT_DOUBLE] = CastTask<uint8_t, double>;
  calls[DT_UINT8][DT_UINT8] = CastTask<uint8_t, uint8_t>;
  calls[DT_UINT8][DT_UINT16] = CastTask<uint8_t, uint16_t>;
  calls[DT_UINT8][DT_UINT32] = CastTask<uint8_t, uint32_t>;
  calls[DT_UINT8][DT_UINT64] = CastTask<uint8_t, uint64_t>;
  calls[DT_UINT8][DT_BOOL] = CastTask<uint8_t, bool>;
  calls[DT_UINT8][DT_COMPLEX64] = CastTask<uint8_t, std::complex<std::float_t>>;
  calls[DT_UINT8][DT_COMPLEX128] = CastTask<uint8_t, std::complex<std::double_t>>;
  calls[DT_UINT16][DT_INT8] = CastTask<uint16_t, int8_t>;
  calls[DT_UINT16][DT_INT16] = CastTask<uint16_t, int16_t>;
  calls[DT_UINT16][DT_INT32] = CastTask<uint16_t, int32_t>;
  calls[DT_UINT16][DT_INT64] = CastTask<uint16_t, int64_t>;
  calls[DT_UINT16][DT_FLOAT16] = CastTask<uint16_t, Eigen::half>;
  calls[DT_UINT16][DT_FLOAT] = CastTask<uint16_t, float>;
  calls[DT_UINT16][DT_DOUBLE] = CastTask<uint16_t, double>;
  calls[DT_UINT16][DT_UINT8] = CastTask<uint16_t, uint8_t>;
  calls[DT_UINT16][DT_UINT16] = CastTask<uint16_t, uint16_t>;
  calls[DT_UINT16][DT_UINT32] = CastTask<uint16_t, uint32_t>;
  calls[DT_UINT16][DT_UINT64] = CastTask<uint16_t, uint64_t>;
  calls[DT_UINT16][DT_BOOL] = CastTask<uint16_t, bool>;
  calls[DT_UINT16][DT_COMPLEX64] = CastTask<uint16_t, std::complex<std::float_t>>;
  calls[DT_UINT16][DT_COMPLEX128] = CastTask<uint16_t, std::complex<std::double_t>>;
  calls[DT_UINT32][DT_INT8] = CastTask<uint32_t, int8_t>;
  calls[DT_UINT32][DT_INT16] = CastTask<uint32_t, int16_t>;
  calls[DT_UINT32][DT_INT32] = CastTask<uint32_t, int32_t>;
  calls[DT_UINT32][DT_INT64] = CastTask<uint32_t, int64_t>;
  calls[DT_UINT32][DT_FLOAT16] = CastTask<uint32_t, Eigen::half>;
  calls[DT_UINT32][DT_FLOAT] = CastTask<uint32_t, float>;
  calls[DT_UINT32][DT_DOUBLE] = CastTask<uint32_t, double>;
  calls[DT_UINT32][DT_UINT8] = CastTask<uint32_t, uint8_t>;
  calls[DT_UINT32][DT_UINT16] = CastTask<uint32_t, uint16_t>;
  calls[DT_UINT32][DT_UINT32] = CastTask<uint32_t, uint32_t>;
  calls[DT_UINT32][DT_UINT64] = CastTask<uint32_t, uint64_t>;
  calls[DT_UINT32][DT_BOOL] = CastTask<uint32_t, bool>;
  calls[DT_UINT32][DT_COMPLEX64] = CastTask<int32_t, std::complex<std::float_t>>;
  calls[DT_UINT32][DT_COMPLEX128] = CastTask<int32_t, std::complex<std::double_t>>;
  calls[DT_UINT64][DT_INT8] = CastTask<uint64_t, int8_t>;
  calls[DT_UINT64][DT_INT16] = CastTask<uint64_t, int16_t>;
  calls[DT_UINT64][DT_INT32] = CastTask<uint64_t, int32_t>;
  calls[DT_UINT64][DT_INT64] = CastTask<uint64_t, int64_t>;
  calls[DT_UINT64][DT_FLOAT16] = CastTask<uint64_t, Eigen::half>;
  calls[DT_UINT64][DT_FLOAT] = CastTask<uint64_t, float>;
  calls[DT_UINT64][DT_DOUBLE] = CastTask<uint64_t, double>;
  calls[DT_UINT64][DT_UINT8] = CastTask<uint64_t, uint8_t>;
  calls[DT_UINT64][DT_UINT16] = CastTask<uint64_t, uint16_t>;
  calls[DT_UINT64][DT_UINT32] = CastTask<uint64_t, uint32_t>;
  calls[DT_UINT64][DT_UINT64] = CastTask<uint64_t, uint64_t>;
  calls[DT_UINT64][DT_BOOL] = CastTask<uint64_t, bool>;
  calls[DT_UINT64][DT_COMPLEX64] = CastTask<uint64_t, std::complex<std::float_t>>;
  calls[DT_UINT64][DT_COMPLEX128] = CastTask<uint64_t, std::complex<std::double_t>>;
  calls[DT_BOOL][DT_INT8] = CastTask<bool, int8_t>;
  calls[DT_BOOL][DT_INT16] = CastTask<bool, int16_t>;
  calls[DT_BOOL][DT_INT32] = CastTask<bool, int32_t>;
  calls[DT_BOOL][DT_INT64] = CastTask<bool, int64_t>;
  calls[DT_BOOL][DT_FLOAT16] = CastTask<bool, Eigen::half>;
  calls[DT_BOOL][DT_FLOAT] = CastTask<bool, float>;
  calls[DT_BOOL][DT_DOUBLE] = CastTask<bool, double>;
  calls[DT_BOOL][DT_UINT8] = CastTask<bool, uint8_t>;
  calls[DT_BOOL][DT_UINT16] = CastTask<bool, uint16_t>;
  calls[DT_BOOL][DT_UINT32] = CastTask<bool, uint32_t>;
  calls[DT_BOOL][DT_UINT64] = CastTask<bool, uint64_t>;
  calls[DT_BOOL][DT_BOOL] = CastTask<bool, bool>;
  calls[DT_BOOL][DT_COMPLEX64] = CastTask<bool, std::complex<std::float_t>>;
  calls[DT_BOOL][DT_COMPLEX128] = CastTask<bool, std::complex<std::double_t>>;
  calls[DT_COMPLEX64][DT_INT8] = CastTask<std::complex<std::float_t>, int8_t>;
  calls[DT_COMPLEX64][DT_INT16] = CastTask<std::complex<std::float_t>, int16_t>;
  calls[DT_COMPLEX64][DT_INT32] = CastTask<std::complex<std::float_t>, int32_t>;
  calls[DT_COMPLEX64][DT_INT64] = CastTask<std::complex<std::float_t>, int64_t>;
  calls[DT_COMPLEX64][DT_FLOAT16] = CastTask<std::complex<std::float_t>, Eigen::half>;
  calls[DT_COMPLEX64][DT_FLOAT] = CastTask<std::complex<std::float_t>, float>;
  calls[DT_COMPLEX64][DT_DOUBLE] = CastTask<std::complex<std::float_t>, double>;
  calls[DT_COMPLEX64][DT_UINT8] = CastTask<std::complex<std::float_t>, uint8_t>;
  calls[DT_COMPLEX64][DT_UINT16] = CastTask<std::complex<std::float_t>, uint16_t>;
  calls[DT_COMPLEX64][DT_UINT32] = CastTask<std::complex<std::float_t>, uint32_t>;
  calls[DT_COMPLEX64][DT_UINT64] = CastTask<std::complex<std::float_t>, uint64_t>;
  calls[DT_COMPLEX64][DT_BOOL] = CastTask<std::complex<std::float_t>, bool>;
  calls[DT_COMPLEX64][DT_COMPLEX64] = CastTask<std::complex<std::float_t>, std::complex<std::float_t>>;
  calls[DT_COMPLEX64][DT_COMPLEX128] = CastTask<std::complex<std::float_t>, std::complex<std::double_t>>;
  calls[DT_COMPLEX128][DT_INT8] = CastTask<std::complex<std::double_t>, int8_t>;
  calls[DT_COMPLEX128][DT_INT16] = CastTask<std::complex<std::double_t>, int16_t>;
  calls[DT_COMPLEX128][DT_INT32] = CastTask<std::complex<std::double_t>, int32_t>;
  calls[DT_COMPLEX128][DT_INT64] = CastTask<std::complex<std::double_t>, int64_t>;
  calls[DT_COMPLEX128][DT_FLOAT16] = CastTask<std::complex<std::double_t>, Eigen::half>;
  calls[DT_COMPLEX128][DT_FLOAT] = CastTask<std::complex<std::double_t>, float>;
  calls[DT_COMPLEX128][DT_DOUBLE] = CastTask<std::complex<std::double_t>, double>;
  calls[DT_COMPLEX128][DT_UINT8] = CastTask<std::complex<std::double_t>, uint8_t>;
  calls[DT_COMPLEX128][DT_UINT16] = CastTask<std::complex<std::double_t>, uint16_t>;
  calls[DT_COMPLEX128][DT_UINT32] = CastTask<std::complex<std::double_t>, uint32_t>;
  calls[DT_COMPLEX128][DT_UINT64] = CastTask<std::complex<std::double_t>, uint64_t>;
  calls[DT_COMPLEX128][DT_BOOL] = CastTask<std::complex<std::double_t>, bool>;
  calls[DT_COMPLEX128][DT_COMPLEX64] = CastTask<std::complex<std::double_t>, std::complex<std::float_t>>;
  calls[DT_COMPLEX128][DT_COMPLEX128] = CastTask<std::complex<std::double_t>, std::complex<std::double_t>>;
  return calls[input_type_][output_type_](ctx);
}

uint32_t CastKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto output_tensor = ctx.Output(0);
  output_type_ = output_tensor->GetDataType();
  auto input_tensor = ctx.Input(0);
  input_type_ = input_tensor->GetDataType();
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kCast, CastKernel);
}  // namespace aicpu

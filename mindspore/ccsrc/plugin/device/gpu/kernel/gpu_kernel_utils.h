/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_UTILS_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <numeric>
#include <string>
#include <vector>
#include "mindspore/core/utils/log_adapter.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
inline void MatrixTransposeND(const T *src, const std::vector<size_t> &host_shape, const std::vector<size_t> host_axis,
                              size_t *dev_shape, size_t *dev_axis, T *dst, cudaStream_t cuda_stream,
                              const std::string &kernel_name) {
  if (host_shape.size() != host_axis.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', size of host_shape and host_axis mismatch: " << host_shape.size()
                      << " != " << host_axis.size();
  }
  const size_t src_size = std::accumulate(host_shape.begin(), host_shape.end(), size_t(1), std::multiplies{});
  TransposeInfo info;
  for (size_t i = 0; i < host_shape.size(); ++i) {
    info.input_shape.push_back(static_cast<int64_t>(host_shape[i]));
    info.perm.push_back(static_cast<int32_t>(host_axis[i]));
  }
  (void)CalTranspose<T, true>(src_size, src, info, dst, cuda_stream);
}
template <>
inline void MatrixTransposeND(const cuComplex *src, const std::vector<size_t> &host_shape,
                              const std::vector<size_t> host_axis, size_t *dev_shape, size_t *dev_axis, cuComplex *dst,
                              cudaStream_t cuda_stream, const std::string &kernel_name) {
  auto converted_src = reinterpret_cast<const mindspore::utils::Complex<float> *>(src);
  auto converted_dst = reinterpret_cast<mindspore::utils::Complex<float> *>(dst);
  MatrixTransposeND(converted_src, host_shape, host_axis, dev_shape, dev_axis, converted_dst, cuda_stream, kernel_name);
}
template <>
inline void MatrixTransposeND(const cuDoubleComplex *src, const std::vector<size_t> &host_shape,
                              const std::vector<size_t> host_axis, size_t *dev_shape, size_t *dev_axis,
                              cuDoubleComplex *dst, cudaStream_t cuda_stream, const std::string &kernel_name) {
  auto converted_src = reinterpret_cast<const mindspore::utils::Complex<double> *>(src);
  auto converted_dst = reinterpret_cast<mindspore::utils::Complex<double> *>(dst);
  MatrixTransposeND(converted_src, host_shape, host_axis, dev_shape, dev_axis, converted_dst, cuda_stream, kernel_name);
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_UTILS_H_

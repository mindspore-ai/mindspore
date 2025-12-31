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
#include "utils/log_adapter.h"
#include "kernel/gpu/gpu_common.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/cast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr double SCALE_ZERO_THRESHOLD = 0.0;
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

template <typename S, typename T>
void CastKernelTensor(KernelTensor *source, KernelTensor *target, cudaStream_t stream, const std::string &kernel_name) {
  MS_EXCEPTION_IF_NULL(source);
  MS_EXCEPTION_IF_NULL(source->device_ptr());
  S *source_addr = reinterpret_cast<S *>(source->device_ptr());
  MS_EXCEPTION_IF_NULL(target);
  MS_EXCEPTION_IF_NULL(target->device_ptr());
  T *target_addr = reinterpret_cast<T *>(target->device_ptr());
  auto status = Cast(source->size(), source_addr, target_addr, stream);
  CHECK_CUDA_STATUS(status, kernel_name);
}

template <typename T>
inline T ComputeScalesBackward(const double scale, const int64_t src_size, const int64_t dst_size) {
  if (scale > SCALE_ZERO_THRESHOLD) {
    return static_cast<T>(scale);
  } else if (dst_size > 0) {
    return static_cast<T>(src_size) / dst_size;
  }
  return 0;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_GPU_KERNEL_UTILS_H_

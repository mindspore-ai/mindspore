/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_KERNELS_NORMALIZED_SCALEANDTRANSLATE_H_
#define AICPU_KERNELS_NORMALIZED_SCALEANDTRANSLATE_H_

#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {

class ScaleAndTranslateCpuKernel : public CpuKernel {
 public:
  ScaleAndTranslateCpuKernel() = default;
  ~ScaleAndTranslateCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t ScaleAndTranslateCheck(CpuKernelContext &ctx);

  template <typename T>
  static uint32_t ScaleAndTranslateCompute(CpuKernelContext &ctx);
};

class ScaleAndTranslateGradCpuKernel : public CpuKernel {
 public:
  ScaleAndTranslateGradCpuKernel() = default;
  ~ScaleAndTranslateGradCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  static uint32_t ScaleAndTranslateGradCheck(CpuKernelContext &ctx);

  template <typename T>
  static uint32_t ScaleAndTranslateGradCompute(CpuKernelContext &ctx);
};

struct Spans {
  // The maximum span size of any output pixel.
  int span_size;
  // int32 tensor of size [output_dim].
  Eigen::Tensor<int32_t, 1> *starts;
  // float tensor of size [output_dim, span_size].
  Eigen::Tensor<float, 1> *weights;
};

template <typename T>
struct GatherSpans {
  uint32_t operator()(aicpu::CpuKernelContext &context, int row_span_size,
                      Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts,
                      Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights, int col_span_size,
                      Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts,
                      Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights, typename TTypes<T, 4>::Tensor input_images,
                      Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_buffer,
                      typename TTypes<float, 4>::Tensor output_images);
};
}  // namespace aicpu
#endif

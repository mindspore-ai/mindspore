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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCALE_AND_TRANSLATE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCALE_AND_TRANSLATE_CPU_KERNEL_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/eigen/eigen_common_utils.h"
#include "plugin/device/cpu/kernel/utils/sampling_kernels.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t dim1 = 1;
constexpr size_t dim4 = 4;

struct Spans {
  // The maximum span size of any output pixel.
  int64_t span_size;
  // int64 tensor of size [output_dim].
  std::shared_ptr<Eigen::Tensor<int64_t, 1>> starts;
  // float tensor of size [output_dim, span_size].
  std::shared_ptr<Eigen::Tensor<float, 1>> weights;
};

template <typename T>
struct GatherSpans {
  uint32_t operator()(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                      int64_t row_span_size, Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts,
                      Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights, int64_t col_span_size,
                      Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts,
                      Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights,
                      typename TTypes<T, dim4>::Tensor input_images,
                      Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_buffer,
                      typename TTypes<float, dim4>::Tensor output_images);
};

class ScaleAndTranslateCpuKernelMod : public NativeCpuKernelMod {
 public:
  ScaleAndTranslateCpuKernelMod() = default;

  ~ScaleAndTranslateCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &others = std::map<uint32_t, tensor::TensorPtr>()) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  friend class ScaleAndTranslateGradCpuKernelMod;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using ScaleAndTranslateKernel = std::function<bool(
    ScaleAndTranslateCpuKernelMod *, const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  template <typename T>
  void GatherRows(int64_t span_size, const int64_t *begin, const float *weights, const T *image,
                  const int64_t in_height, const int64_t in_width, const int64_t out_height, const int64_t out_width,
                  const int64_t channels, float *output);
  template <typename T>
  void GatherColumns(int64_t span_size, const int64_t *begin, const float *weights, const T *image,
                     const int64_t in_height, const int64_t in_width, const int64_t out_height, const int64_t out_width,
                     const int64_t channels, float *output);
  template <typename Kernel>
  void ComputeSpansCore(const Kernel &kernel, const int64_t out_size, const int64_t in_size, const float scale,
                        const float translate, bool antialias, Spans *spans);
  bool ComputeSpans(const KernelType kernel_type, const int64_t out_size, const int64_t in_size, const float scale,
                    const float translate, const bool antialias, Spans *spans, const std::string kernel_name);
  template <typename T>
  uint32_t GatherSpans(int64_t size_row_span, Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_row,
                       Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights, int64_t size_col_span,
                       Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_col,
                       Eigen::TensorMap<Eigen::Tensor<float, dim1>> weights_col,
                       typename TTypes<T, dim4>::Tensor images,
                       Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_buffer,
                       typename TTypes<float, dim4>::Tensor resized_images);
  ScaleAndTranslateKernel kernel_func_;
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
  std::vector<int64_t> input2_shape_;
  std::vector<int64_t> input3_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input0_dtype_{kTypeUnknown};
  std::string kernel_type_{"lanczos3"};
  bool antialias_{true};
};

class ScaleAndTranslateGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  ScaleAndTranslateGradCpuKernelMod() = default;

  ~ScaleAndTranslateGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &others = std::map<uint32_t, tensor::TensorPtr>()) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using ScaleAndTranslateGradKernel =
    std::function<bool(ScaleAndTranslateGradCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &)>;
  ScaleAndTranslateGradKernel kernel_func_;

  void ComputeGradSpansCore(const Spans *spans, const int64_t forward_output_size, const int64_t forward_input_size,
                            Spans *grad_spans);
  bool ComputeGradSpans(const KernelType kernel_type, const int64_t output_size, const int64_t input_size,
                        const float scale, const float translate, const bool antialias, Spans *spans,
                        const std::string kernel_name);
  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
  std::vector<int64_t> input2_shape_;
  std::vector<int64_t> input3_shape_;
  std::vector<int64_t> output_shape_;
  std::string kernel_type_{"lanczos3"};
  bool antialias_{true};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCALE_AND_TRANSLATE_CPU_KERNEL_H_

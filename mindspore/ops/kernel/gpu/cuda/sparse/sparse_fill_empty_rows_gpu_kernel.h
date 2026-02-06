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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_H

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <complex>

#include "mindspore/ops/infer/sparse_fill_empty_rows.h"
#include "plugin/gpu/res_manager/symbol_interface/cuda_driver.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_fill_empty_rows.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

constexpr size_t kSparseFillEmptyRowsInputsNum = 4;
constexpr size_t kSparseFillEmptyRowsOutputsNum = 4;

namespace mindspore {
namespace kernel {
class SparseFillEmptyRowsGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseFillEmptyRowsGpuKernelMod() { ResetResource(); }
  ~SparseFillEmptyRowsGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void ResetResource() noexcept;

 protected:
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  template <typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using SparseFillEmptyRowsFunc =
    std::function<bool(SparseFillEmptyRowsGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseFillEmptyRowsFunc>> func_list_;
  SparseFillEmptyRowsFunc kernel_func_;
  void *cuda_stream_{nullptr};
  bool is_null_input_{false};
  size_t input_indice_size_{0};
  size_t input_values_size_{0};
  size_t input_dense_shape_size_{0};
  size_t input_default_values_size_{0};
  size_t output_indices_size_{0};
  size_t output_values_size_{0};
  size_t output_empty_row_indicator_size_{0};
  size_t output_reverse_index_map_size_{0};
  size_t output_elements1_{0};
  size_t output_elements2_{0};
  size_t output_elements3_{0};
  size_t output_elements4_{0};
  size_t dense_row{0};
  size_t real_output_size_;  // Dynamic shape related.
  std::vector<int64_t> input_dense_shape_shapes_;
  std::vector<int64_t> input_default_values_shapes_;
  std::vector<int64_t> input_indices_shapes_;
  std::vector<int64_t> input_values_shapes_;
  std::vector<int64_t> output_indices_shapes_;
  std::vector<int64_t> output_values_shapes_;
  std::vector<int64_t> output_empty_row_indicator_shape_;
  std::vector<int64_t> output_reverse_index_map_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_H

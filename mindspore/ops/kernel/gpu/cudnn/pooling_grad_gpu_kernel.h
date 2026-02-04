/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include <algorithm>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
class PoolingGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit PoolingGradGpuKernelMod(const std::string &kernel_name) : kernel_name_(kernel_name) { InitResource(); }
  ~PoolingGradGpuKernelMod() override { DestroyResource(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  void DestroyResource() noexcept override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  void InitResource() override;
  void InitSizeLists();

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  bool InitShape(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs, int *dimA,
                 int *strideAin, int *dimAy, int *strideAiny, int *dimAdy, int *strideAdy, int *dimAout,
                 int *strideAout, int nbDims);
  void SetPad();
  void SetPad3D();
  void SetPoolingMode();
  std::vector<int64_t> GetEdgeKernelSize();
  void SetFirstInputIndex(size_t input_num);
  using PoolingGradFunc =
    std::function<bool(PoolingGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  PoolingGradFunc kernel_func_;
  static std::map<std::string, std::vector<std::pair<KernelAttr, PoolingGradGpuKernelMod::PoolingGradFunc>>>
    kernel_attr_map_;

  cudnnHandle_t cudnn_handle_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnTensorDescriptor_t y_descriptor_;
  cudnnTensorDescriptor_t dy_descriptor_;
  cudnnTensorDescriptor_t x_descriptor_;
  cudnnTensorDescriptor_t dx_descriptor_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  cudnnPoolingMode_t pooling_mode_;

  std::vector<int> stride_;
  std::vector<int64_t> stride_me_;
  std::vector<int64_t> window_me_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> edge_kernel_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> pad_list_;
  std::string kernel_name_;
  PadMode pad_mode_;
  Format format_attr_ = Format::NCHW;

  int old_depth_{0};
  int old_height_{0};
  int old_width_{0};
  int pad_depth_{0};
  int pad_height_{0};
  int pad_width_{0};
  int pad_front_{0};
  int pad_top_{0};
  int pad_left_{0};
  int n_{0};
  int c_{0};
  float pad_value_{0.0};
  bool is_null_input_{false};
  bool include_{false};
  bool ceil_mode_{false};
  int64_t divisor_override_{0};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t workspace_size_{0};
  void *cuda_stream_{nullptr};
  // For the dynamic shape of AvgPool3DGrad with two input, index of first valid input data is 1,
  // and in other cases, the index is 0.
  size_t first_input_index_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_

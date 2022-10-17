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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_UNSORTED_SEGMENT_PROD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_UNSORTED_SEGMENT_PROD_GPU_KERNEL_H_

#include <utility>
#include <map>
#include <vector>
#include <algorithm>
#include "mindspore/core/abstract/utils.h"
#include "mindspore/ccsrc/kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unsorted_segment_prod.cuh"

namespace mindspore {
namespace kernel {
class UnsortedSegmentProdGpuKernelMod : public NativeGpuKernelMod,
                                        public MatchKernelHelper<UnsortedSegmentProdGpuKernelMod> {
 public:
  UnsortedSegmentProdGpuKernelMod() {}
  ~UnsortedSegmentProdGpuKernelMod() {}

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

 private:
  size_t input_dim0_ = 1;
  size_t input_dim1_ = 1;
  size_t output_dim0_ = 1;
  size_t output_dim1_ = 1;
  size_t ids_unit_size_ = 0; /* size of S */
  int64_t batch_rank_ = 0;
  int64_t batch_size_ = 1;
  int64_t in_stride_ = 1;
  int64_t ids_stride_ = 1;
  int64_t out_stride_ = 1;
  int64_t num_segments_ = 1;
  size_t loop_size_ = 0;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_UNSORTED_SEGMENT_PROD_GPU_KERNEL_H_

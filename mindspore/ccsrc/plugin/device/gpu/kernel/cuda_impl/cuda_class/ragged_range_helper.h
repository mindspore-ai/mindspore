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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RAGGED_RANGE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RAGGED_RANGE_HELPER_H_
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/ragged_range_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T, typename TSPLITS>
class RaggedRangeHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit RaggedRangeHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    broadcast_starts_ = false;
    broadcast_limits_ = false;
    broadcast_deltas_ = false;
  }

  virtual ~RaggedRangeHelperGpuKernel() = default;

  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 3;
    constexpr size_t OUTPUT_NESTED_SPLITS_NUM = 1;
    constexpr size_t OUTPUT_DENSE_VALUES_NUM = 1;
    constexpr size_t WORKSPACE_NUM = 1;
    const std::vector<std::vector<int64_t>> nested_splits_shapes{output_shapes[0]};
    const std::vector<std::vector<int64_t>> dense_values_shapes{output_shapes[1]};
    const std::vector<std::vector<int64_t>> range_sizes_shapes{std::vector<int64_t>{output_shapes[0][0] - 1}};
    int in_flag, nested_splits_out_flag, dense_values_out_flag;
    int workspace_flag;
    ResetResource();

    if ((in_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_)) ==
        -1) {
      return in_flag;
    }
    if ((workspace_flag = CalShapesSizeInBytes<TSPLITS>(range_sizes_shapes, WORKSPACE_NUM, kernel_name_,
                                                        "range_sizes_shape", &work_size_list_)) == -1) {
      return workspace_flag;
    }
    if ((nested_splits_out_flag =
           CalShapesSizeInBytes<TSPLITS>(nested_splits_shapes, OUTPUT_NESTED_SPLITS_NUM, kernel_name_,
                                         "rt_nested_splits_shape", &output_size_list_)) == -1) {
      return nested_splits_out_flag;
    }
    if ((dense_values_out_flag = CalShapesSizeInBytes<T>(dense_values_shapes, OUTPUT_DENSE_VALUES_NUM, kernel_name_,
                                                         "rt_dense_values_shape", &output_size_list_)) == -1) {
      return dense_values_out_flag;
    }

    size_t starts_dim = input_shapes[0].size();
    size_t limits_dim = input_shapes[1].size();
    size_t deltas_dim = input_shapes[2].size();
    broadcast_starts_ = starts_dim == 0;
    broadcast_limits_ = limits_dim == 0;
    broadcast_deltas_ = deltas_dim == 0;
    const std::vector<int64_t> &starts_shape = input_shapes[0];
    const std::vector<int64_t> &limits_shape = input_shapes[1];
    const std::vector<int64_t> &deltas_shape = input_shapes[2];
    if (!broadcast_starts_) {
      in_sizes_.push_back(starts_shape[0]);
    }
    if (!broadcast_limits_) {
      in_sizes_.push_back(limits_shape[0]);
    }
    if (!broadcast_deltas_) {
      in_sizes_.push_back(deltas_shape[0]);
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    constexpr size_t startsAddrIdx = 0;
    constexpr size_t limitsAddrIdx = 1;
    constexpr size_t deltasAddrIdx = 2;
    constexpr size_t nestedSplitsAddrIdx = 0;
    constexpr size_t denseValuesAddrIdx = 1;
    constexpr size_t rangeSizesAddrIdx = 0;
    size_t nrows = static_cast<size_t>(in_sizes_.empty() ? 1 : in_sizes_[0]);
    T *starts_addr;
    T *limits_addr;
    T *deltas_addr;
    TSPLITS *rt_nested_splits_addr;
    T *rt_dense_values_addr;
    TSPLITS *range_sizes_addr;

    (void)GetDeviceAddress<T>(input_ptrs, startsAddrIdx, kernel_name_, &starts_addr);
    (void)GetDeviceAddress<T>(input_ptrs, limitsAddrIdx, kernel_name_, &limits_addr);
    (void)GetDeviceAddress<T>(input_ptrs, deltasAddrIdx, kernel_name_, &deltas_addr);
    (void)GetDeviceAddress<TSPLITS>(output_ptrs, nestedSplitsAddrIdx, kernel_name_, &rt_nested_splits_addr);
    (void)GetDeviceAddress<T>(output_ptrs, denseValuesAddrIdx, kernel_name_, &rt_dense_values_addr);
    (void)GetDeviceAddress<TSPLITS>(work_ptrs, rangeSizesAddrIdx, kernel_name_, &range_sizes_addr);

    CalRaggedRange(starts_addr, limits_addr, deltas_addr, rt_nested_splits_addr, rt_dense_values_addr, range_sizes_addr,
                   nrows, broadcast_starts_, broadcast_limits_, broadcast_deltas_, device_id_,
                   reinterpret_cast<cudaStream_t>(cuda_stream));

    return 0;
  }

  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    work_size_list_.clear();
    broadcast_starts_ = false;
    broadcast_limits_ = false;
    broadcast_deltas_ = false;
    in_sizes_.clear();
  }

 private:
  bool broadcast_starts_;
  bool broadcast_limits_;
  bool broadcast_deltas_;
  std::vector<int> in_sizes_;
};
}  // namespace cukernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RAGGED_RANGE_HELPER_H_

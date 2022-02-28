/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCAST_GRADIENT_ARGS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCAST_GRADIENT_ARGS_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 2;
template <typename T, typename S>
class DynamicBroadcastGradientArgsGpuKernelMod : public NativeGpuKernelMod {
 public:
  DynamicBroadcastGradientArgsGpuKernelMod() : r0_size_(0), r1_size_(0) { ResetResource(); }
  ~DynamicBroadcastGradientArgsGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto s0_addr = GetDeviceAddress<T>(inputs, 0);
    auto s1_addr = GetDeviceAddress<T>(inputs, 1);
    auto r0_addr = GetDeviceAddress<S>(outputs, 0);
    auto r1_addr = GetDeviceAddress<S>(outputs, 1);
    std::vector<T> x0_value(input_size_list_[0] / sizeof(T), 0);
    std::vector<T> x1_value(input_size_list_[1] / sizeof(T), 0);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(&x0_value[0], s0_addr, input_size_list_[0], cudaMemcpyDeviceToHost, cuda_stream),
      "DynamicBroadcastGradientArgs copy s0 value failed");
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(&x1_value[0], s1_addr, input_size_list_[1], cudaMemcpyDeviceToHost, cuda_stream),
      "DynamicBroadcastGradientArgs copy s1 value failed");
    auto grad_reduce_idx = CalOut({x0_value, x1_value});
    r0_size_ = SetOuputValue(r0_addr, grad_reduce_idx[0], x0_value.size(), cuda_stream);
    r1_size_ = SetOuputValue(r1_addr, grad_reduce_idx[1], x1_value.size(), cuda_stream);

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "DynamicBroadcastGradiendArgs needs " << kInputNum << " inputs, but get " << input_num;
    }
    auto s0_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto s1_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto r0_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    auto r1_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 1);
    if (s0_shape.size() != 1 || s1_shape.size() != 1) {
      MS_LOG(EXCEPTION) << "Inputs must be [1-D], but get " << s0_shape.size() << "-D and " << s1_shape.size() << "-D.";
    }

    auto s0_size = std::accumulate(s0_shape.begin(), s0_shape.end(), sizeof(T), std::multiplies<size_t>());
    auto s1_size = std::accumulate(s1_shape.begin(), s1_shape.end(), sizeof(T), std::multiplies<size_t>());

    input_size_list_.push_back(s0_size);
    input_size_list_.push_back(s1_size);
    output_size_list_.push_back(r0_shape[0] * sizeof(S));
    output_size_list_.push_back(r1_shape[0] * sizeof(S));
    return true;
  }
  void ResetResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }
  void UpdateOp() override {
    std::vector<size_t> r0_shape{r0_size_};
    std::vector<size_t> r1_shape{r1_size_};
    common::AnfAlgo::SetOutputInferTypeAndShape({TypeId::kNumberTypeInt64, TypeId::kNumberTypeInt64},
                                                {r0_shape, r1_shape}, kernel_node_.lock().get());
    MS_LOG(DEBUG) << "Run PostExecute for DynamicBroadcastGradientArgs, real r0 shape is " << r0_shape
                  << ", r1 shape is " << r1_shape;
  }

 protected:
  void InitSizeLists() override{};

 private:
  std::vector<std::vector<T>> CalOut(const std::vector<std::vector<T>> &input_shapes) {
    std::vector<std::vector<T>> grad_reduce_idx(kInputNum);
    bool all_equal = true;
    size_t max_rank = 0;
    for (size_t i = 0; i < kInputNum; i++) {
      if (input_shapes[i] != input_shapes[0]) {
        all_equal = false;
      }
      if (input_shapes[i].size() > max_rank) {
        max_rank = input_shapes[i].size();
      }
    }
    if (all_equal) {
      return grad_reduce_idx;
    }
    // Reverse shapes
    std::vector<std::vector<T>> reverse_shapes(kInputNum);
    for (size_t i = 0; i < kInputNum; i++) {
      reverse_shapes[i] = input_shapes[i];
      std::reverse(reverse_shapes[i].begin(), reverse_shapes[i].end());
      if (reverse_shapes[i].size() < max_rank) {
        reverse_shapes[i].resize(max_rank, 1);
      }
    }
    grad_reduce_idx = GetGradIndex(reverse_shapes, max_rank);
    return grad_reduce_idx;
  }

  void AddGradReduceIdx(std::vector<std::vector<T>> *grad_reduce_idx, std::vector<bool> cur_one, bool none_one,
                        const size_t max_rank, size_t j) {
    MS_EXCEPTION_IF_NULL(grad_reduce_idx);
    for (size_t i = 0; i < kInputNum; i++) {
      if (cur_one[i] && !none_one) {
        (void)(*grad_reduce_idx)[i].emplace_back(SizeToLong(max_rank - 1 - j));
      }
    }
  }

  std::vector<std::vector<T>> GetGradIndex(const std::vector<std::vector<T>> &revers_shapes, const size_t max_rank) {
    std::vector<std::vector<T>> grad_reduce_index(kInputNum);
    std::vector<bool> pre_one(kInputNum);
    std::vector<bool> cur_one(kInputNum);
    for (size_t i = 0; i < kInputNum; i++) {
      pre_one[i] = false;
      cur_one[i] = false;
    }
    bool set_one = false;
    for (size_t j = 0; j < max_rank; j++) {
      int out_dim = -1;
      bool out_dim_set = false;
      bool none_one = true;
      for (size_t i = 0; i < kInputNum; i++) {
        if (revers_shapes[i][j] == 1) {
          cur_one[i] = true;
          none_one = false;
        } else {
          cur_one[i] = false;
          if (!out_dim_set || revers_shapes[i][j] == static_cast<T>(out_dim)) {
            out_dim = static_cast<int>(revers_shapes[i][j]);
            out_dim_set = true;
          } else {
            MS_LOG(EXCEPTION) << "Can not broadcast inputs[0] and inputs[1].";
          }
        }
      }
      if (!out_dim_set) {
        for (size_t i = 0; i < kInputNum; i++) {
          (void)grad_reduce_index[i].emplace_back(max_rank - 1 - j);
        }
        continue;
      } else if (std::equal(cur_one.begin(), cur_one.end(), pre_one.begin()) && set_one) {
        AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
      } else {
        AddGradReduceIdx(&grad_reduce_index, cur_one, none_one, max_rank, j);
      }
      set_one = true;
      for (size_t i = 0; i < kInputNum; i++) {
        pre_one[i] = cur_one[i];
      }
    }
    return grad_reduce_index;
  }
  size_t SetOuputValue(S *addr, const std::vector<T> grad_reduce_idx, size_t input_num, cudaStream_t stream) {
    std::vector<S> output;
    size_t index_num = grad_reduce_idx.size();
    for (size_t i = 0; i < index_num; i++) {
      output.push_back(static_cast<S>(grad_reduce_idx[index_num - 1 - i]));
    }
    size_t out_size = index_num;
    if (index_num == 0) {
      out_size = input_num;
      for (size_t i = 0; i < input_num; i++) {
        output.push_back(static_cast<S>(i));
      }
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(addr, &output[0], out_size * sizeof(S), cudaMemcpyHostToDevice, stream),
                               "DynamicBroadcastGradientArgs copy output failed");
    return out_size;
  }
  size_t r0_size_;
  size_t r1_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_BRAODCAST_GRADIENT_ARGS_GPU_KERNEL_H_

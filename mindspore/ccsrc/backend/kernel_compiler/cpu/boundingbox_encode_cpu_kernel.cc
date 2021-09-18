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

#include "backend/kernel_compiler/cpu/boundingbox_encode_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void BoundingBoxEncodeCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != INPUT_NUMS) {
    MS_LOG(ERROR) << "Input num is " << input_num << ", but BoundingBoxEncode needs 2 inputs.";
  }

  const size_t coordinate_size = 4;
  if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<ValueTuple>() ||
      AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<ValueList>()) {
    means_ = AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, "means");
  } else if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("means")->isa<FloatImm>()) {
    float mean = AnfAlgo::GetNodeAttr<float>(kernel_node, "means");
    for (size_t i = 0; i < coordinate_size; i++) {
      means_.emplace_back(mean);
    }
  } else {
    MS_LOG(EXCEPTION) << "Attribute means type is invalid.";
  }

  if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<ValueTuple>() ||
      AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<ValueList>()) {
    stds_ = AnfAlgo::GetNodeAttr<std::vector<float>>(kernel_node, "stds");
  } else if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("stds")->isa<FloatImm>()) {
    float std = AnfAlgo::GetNodeAttr<float>(kernel_node, "stds");
    for (size_t i = 0; i < coordinate_size; i++) {
      stds_.emplace_back(std);
    }
  } else {
    MS_LOG(EXCEPTION) << "Attribute stds type is invalid.";
  }

  if (means_.size() < coordinate_size || stds_.size() < coordinate_size) {
    MS_LOG(EXCEPTION) << "The size of means or stds is less than 4.";
  }
}

template <typename T>
bool BoundingBoxEncodeCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto anchor_box = reinterpret_cast<T *>(inputs[0]->addr);
  auto groundtruth_box = reinterpret_cast<T *>(inputs[1]->addr);
  auto deltas = reinterpret_cast<T *>(outputs[0]->addr);

  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(ERROR) << "Anchor box size must be equal to groundtruth box size: " << inputs[1]->size << ", but got"
                  << inputs[0]->size;
    return false;
  }

  const size_t coordinate = 4;
  const size_t block_size = inputs[0]->size / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "The size of the box must be a multiple of 4.";
    return false;
  }

  size_t elem_num = block_size / coordinate;
  auto task = [this, &anchor_box, &groundtruth_box, &deltas](size_t start, size_t end) {
    constexpr size_t X_INDEX = 0;
    constexpr size_t Y_INDEX = 1;
    constexpr size_t W_INDEX = 2;
    constexpr size_t H_INDEX = 3;
    const T HALF = static_cast<T>(0.5);
    const T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      const size_t left_x = i * 4;
      const size_t left_y = i * 4 + 1;
      const size_t right_x = i * 4 + 2;
      const size_t right_y = i * 4 + 3;

      T px = (anchor_box[left_x] + anchor_box[right_x]) * HALF;
      T py = (anchor_box[left_y] + anchor_box[right_y]) * HALF;
      T pw = anchor_box[right_x] - anchor_box[left_x] + ONE;
      T ph = anchor_box[right_y] - anchor_box[left_y] + ONE;

      T gx = (groundtruth_box[left_x] + groundtruth_box[right_x]) * HALF;
      T gy = (groundtruth_box[left_y] + groundtruth_box[right_y]) * HALF;
      T gw = groundtruth_box[right_x] - groundtruth_box[left_x] + ONE;
      T gh = groundtruth_box[right_y] - groundtruth_box[left_y] + ONE;

      T dx = (gx - px) / pw;
      T dy = (gy - py) / ph;
      T dw = log(gw / pw);
      T dh = log(gh / ph);

      deltas[left_x] = (dx - static_cast<T>(means_[X_INDEX])) / static_cast<T>(stds_[X_INDEX]);
      deltas[left_y] = (dy - static_cast<T>(means_[Y_INDEX])) / static_cast<T>(stds_[Y_INDEX]);
      deltas[right_x] = (dw - static_cast<T>(means_[W_INDEX])) / static_cast<T>(stds_[W_INDEX]);
      deltas[right_y] = (dh - static_cast<T>(means_[H_INDEX])) / static_cast<T>(stds_[H_INDEX]);
    }
  };
  CPUKernelUtils::ParallelFor(task, elem_num);

  return true;
}
}  // namespace kernel
}  // namespace mindspore

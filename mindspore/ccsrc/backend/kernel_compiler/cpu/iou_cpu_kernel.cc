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
#include "backend/kernel_compiler/cpu/iou_cpu_kernel.h"

#include <cmath>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {

template <typename T>
void IOUCPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto anchor_boxes_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (anchor_boxes_shape.size() != 2 || anchor_boxes_shape[1] != 4) {
    MS_LOG(EXCEPTION) << "The anchor_boxes shape should be [N, 4].";
  }
  anchor_boxes_size_ = anchor_boxes_shape[0];
  auto gt_boxes_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (gt_boxes_shape.size() != 2 || gt_boxes_shape[1] != 4) {
    MS_LOG(EXCEPTION) << "The gt_boxes shape should be [N, 4].";
  }
  gt_boxes_size_ = gt_boxes_shape[0];
  iou_size_ = anchor_boxes_size_ * gt_boxes_size_;
  std::string iou_mode = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  if (iou_mode != "iou" && iou_mode != "iof") {
    MS_LOG(EXCEPTION) << "IOU mode should be 'iou', 'iof'.";
  }
  if (iou_mode == "iof") {
    mode_ = 1;
  }
}

template <typename T>
bool IOUCPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                             const std::vector<kernel::AddressPtr> & /*workspace*/,
                             const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.size() != 2) {
    MS_LOG(EXCEPTION) << "Input number is " << inputs.size() << ", but IOU needs 2 inputs.";
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "Output number is " << outputs.size() << ", but IOU needs 1 outputs.";
  }
  auto anchor_boxes = reinterpret_cast<T *>(inputs[0]->addr);
  auto gt_boxes = reinterpret_cast<T *>(inputs[1]->addr);
  auto iou_score = reinterpret_cast<T *>(outputs[0]->addr);

  // multithreading
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int idx1 = i % anchor_boxes_size_ * 4;
      int idx2 = i / anchor_boxes_size_ * 4;
      T I_x0 = std::max(anchor_boxes[idx1], gt_boxes[idx2]);
      T I_y0 = std::max(anchor_boxes[idx1 + 1], gt_boxes[idx2 + 1]);
      T I_x1 = std::min(anchor_boxes[idx1 + 2], gt_boxes[idx2 + 2]);
      T I_y1 = std::min(anchor_boxes[idx1 + 3], gt_boxes[idx2 + 3]);
      T overlaps = std::max(T(0), (I_x1 - I_x0 + T(1)) * (I_y1 - I_y0 + T(1)));
      T area1 =
        (anchor_boxes[idx1 + 2] - anchor_boxes[idx1] + T(1)) * (anchor_boxes[idx1 + 3] - anchor_boxes[idx1 + 1] + T(1));
      T area2 = (gt_boxes[idx2 + 2] - gt_boxes[idx2] + T(1)) * (gt_boxes[idx2 + 3] - gt_boxes[idx2 + 1] + T(1));
      if (mode_ == 0) {
        iou_score[i] = overlaps / (area1 + area2 - overlaps + T(1e-10));
      } else {
        iou_score[i] = overlaps / (area2 + T(1e-10));
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, iou_size_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore

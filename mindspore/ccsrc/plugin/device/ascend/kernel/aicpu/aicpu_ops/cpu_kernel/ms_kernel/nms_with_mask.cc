/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include "nms_with_mask.h"
#include <numeric>
#include "Eigen/Core"
#include "utils/kernel_util.h"

namespace {
const int32_t kInputNum = 1;
const int32_t kOutputNum = 3;
const int kColNum5 = 5;
const int kColNum8 = 8;
const char *kNMSWithMask = "NMSWithMask";
}  // namespace

namespace aicpu {
uint32_t NMSWithMaskCpuKernel::Compute(CpuKernelContext &ctx) {
  // check param
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "NMSWithMask check input or output is failed");
  AttrValue *iou_threshold = ctx.GetAttr("iou_threshold");
  KERNEL_CHECK_FALSE((iou_threshold != nullptr), KERNEL_STATUS_PARAM_INVALID, "Get attr [iou_threshold] failed.");
  iou_value_ = iou_threshold->GetFloat();

  Tensor *input_data = ctx.Input(0);
  auto data_type = input_data->GetDataType();
  KERNEL_CHECK_FALSE((data_type == DT_FLOAT || data_type == DT_FLOAT16), KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] data type[%s] is unsupported", DTypeStr(data_type).c_str());
  auto input_shape = input_data->GetTensorShape()->GetDimSizes();
  num_input_ = input_shape[0];  //  Get N values in  [N, 5] data.
  box_size_ = input_shape[1];
  if (box_size_ != kColNum5 && box_size_ != kColNum8) {
    KERNEL_LOG_INFO("NMSWithMask the col number of input[0] must be [%d] or [%d], but got [%d]!", kColNum5, kColNum8,
                    box_size_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t res;
  switch (data_type) {
    case DT_FLOAT16:
      res = DoCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      res = DoCompute<float>(ctx);
      break;
    default:
      KERNEL_LOG_INFO("NMSWithMask input[0] only support type[DT_FLOAT16, DT_FLOAT], but got type[%s]",
                      DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return res;
}

template <typename T>
uint32_t NMSWithMaskCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(OUTPUT)->GetData());
  auto sel_idx = reinterpret_cast<int *>(ctx.Output(SEL_IDX)->GetData());
  auto sel_boxes = reinterpret_cast<bool *>(ctx.Output(SEL_BOXES)->GetData());
  std::fill(&sel_idx[0], &sel_idx[num_input_], 0);
  std::fill(&sel_boxes[0], &sel_boxes[num_input_], false);

  const int box_size = box_size_;
  const auto comp = [input, box_size](const size_t a, const size_t b) {
    const size_t index_a = a * box_size + 4;
    const size_t index_b = b * box_size + 4;
    if (input[index_b] == input[index_a]) {
      return a < b;
    };
    return input[index_b] < input[index_a];
  };
  std::vector<int> order(num_input_);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), comp);

  std::vector<T> areas(num_input_);
  for (int64_t i = 0; i < num_input_; i++) {
    areas[i] =
      (input[i * box_size_ + 2] - input[i * box_size_]) * (input[i * box_size_ + 3] - input[i * box_size_ + 1]);
  }

  int64_t num_to_keep = 0;
  for (int64_t _i = 0; _i < num_input_; _i++) {
    auto i = order[_i];
    if (sel_boxes[i] == 1) continue;
    sel_idx[num_to_keep++] = i;
    auto ix1 = input[i * box_size_];
    auto iy1 = input[i * box_size_ + 1];
    auto ix2 = input[i * box_size_ + 2];
    auto iy2 = input[i * box_size_ + 3];

    for (int64_t _j = _i + 1; _j < num_input_; _j++) {
      auto j = order[_j];
      if (sel_boxes[j] == 1) continue;
      auto xx1 = std::max(ix1, input[j * box_size_]);
      auto yy1 = std::max(iy1, input[j * box_size_ + 1]);
      auto xx2 = std::min(ix2, input[j * box_size_ + 2]);
      auto yy2 = std::min(iy2, input[j * box_size_ + 3]);

      auto w = std::max(static_cast<T>(0), xx2 - xx1);
      auto h = std::max(static_cast<T>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (areas[i] + areas[j] - inter);
      if (static_cast<float>(ovr) > iou_value_) {
        sel_boxes[j] = 1;
      }
    }
  }

  for (int k = 0; k < num_input_; ++k) {
    for (int j = 0; j < box_size_; ++j) {
      if (k < num_to_keep) {
        output[k * kColNum5 + j] = input[sel_idx[k] * box_size_ + j];
        sel_boxes[k] = true;
      } else {
        output[k * kColNum5 + j] = static_cast<T>(0);
        sel_boxes[k] = false;
      }
    }
  }

  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNMSWithMask, NMSWithMaskCpuKernel);
}  // namespace aicpu

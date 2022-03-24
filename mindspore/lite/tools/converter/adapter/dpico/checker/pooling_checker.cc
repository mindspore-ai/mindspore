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

#include "checker/pooling_checker.h"
#include <vector>
#include <string>
#include <algorithm>
#include "common/anf_util.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxGroupNum = 2048;
constexpr int kMaxPoolUpperBound = 2048;
constexpr int kMaxPadSize = 7;
constexpr int kMaxKernelSize = 255;
constexpr int kMaxStrideSize = 255;
constexpr int kMaxDilationSize = 5;
constexpr int kMaxDilationAndKernelProd = 15;
constexpr float kCoefficient = 16.0;

bool CheckPad(const api::PrimitivePtr &primitive) {
  auto pad_ptr = primitive->GetAttr(ops::kPad);
  if (pad_ptr != nullptr) {
    auto pad_data = api::GetValue<std::vector<int64_t>>(pad_ptr);
    if (pad_data.size() > kDims3) {
      if (pad_data[0] < 0 || pad_data[0] > kMaxPadSize || pad_data[1] < 0 || pad_data[1] > kMaxPadSize ||
          pad_data[kAxis2] < 0 || pad_data[kAxis2] > kMaxPadSize || pad_data[kAxis3] < 0 ||
          pad_data[kAxis3] > kMaxPadSize) {
        MS_LOG(WARNING) << "pad should in range [1,7]";
        return false;
      }
    }
  }
  return true;
}
bool CheckKernelSize(const api::PrimitivePtr &primitive) {
  auto is_global_ptr = primitive->GetAttr(ops::kGlobal);
  auto kernel_ptr = primitive->GetAttr(ops::kKernelSize);
  if (kernel_ptr != nullptr) {
    auto kernel_data = api::GetValue<std::vector<int64_t>>(kernel_ptr);
    if (is_global_ptr != nullptr) {
      auto is_global = api::GetValue<bool>(is_global_ptr);
      if (!is_global && (kernel_data[0] < 1 || kernel_data[0] > kMaxKernelSize || kernel_data[1] < 1 ||
                         kernel_data[1] > kMaxKernelSize)) {
        MS_LOG(WARNING) << "kernel should in range [1,255]";
        return false;
      }
    }
  }
  return true;
}
bool CheckStride(const api::PrimitivePtr &primitive) {
  auto stride_ptr = primitive->GetAttr(ops::kStrides);
  if (stride_ptr != nullptr) {
    auto stride_data = api::GetValue<std::vector<int64_t>>(stride_ptr);
    if (stride_data[0] < 1 || stride_data[0] > kMaxStrideSize || stride_data[1] < 1 ||
        stride_data[1] > kMaxStrideSize) {
      MS_LOG(WARNING) << "stride should in range [1,255]";
      return false;
    }
  }
  return true;
}
bool CheckRoundMode(const api::PrimitivePtr &primitive) {
  auto round_mode_ptr = primitive->GetAttr(ops::kRoundMode);
  if (round_mode_ptr != nullptr) {
    auto round_mode_data = api::GetValue<int64_t>(round_mode_ptr);
    if (round_mode_data != 0 && round_mode_data != 1) {
      MS_LOG(WARNING) << "round mode only supports CEIL or FLOOR";
      return false;
    }
  }
  return true;
}
bool CheckAttr(const api::PrimitivePtr &primitive, int64_t input_w) {
  auto kernel_ptr = primitive->GetAttr(ops::kKernelSize);
  auto stride_ptr = primitive->GetAttr(ops::kStride);
  if (stride_ptr != nullptr && kernel_ptr != nullptr) {
    auto kernel_data = api::GetValue<std::vector<int64_t>>(kernel_ptr);
    auto stride_data = api::GetValue<std::vector<int64_t>>(stride_ptr);
    if (kernel_data[0] > kMaxPoolUpperBound / (input_w / (kCoefficient * stride_data[1]) * stride_data[1])) {
      MS_LOG(WARNING) << "kernel and stride should satisfy kernel_h <= 2048 / (w / (16 * stride) * stride) ";
      return false;
    }
  }
  return true;
}
}  // namespace

bool PoolingChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_shape) == RET_OK) {
    if (input_shape.size() != kDims4) {
      MS_LOG(ERROR) << "Error Pooling input, which should be 4 dims.";
      return false;
    }
    int64_t input_w;
    if (GetWidth(input_shape, format, &input_w) != RET_OK) {
      MS_LOG(ERROR) << "get input_w failed";
      return false;
    }
    if (input_w > kMaxInputWOf4Dims) {
      MS_LOG(WARNING) << "input_w " << input_w << " is greater than " << kMaxInputWOf4Dims << " "
                      << op->fullname_with_scope();
      return false;
    }
    if (!CheckAttr(primitive, input_w)) {
      return false;
    }
  }

  return CheckPad(primitive) && CheckRoundMode(primitive) && CheckKernelSize(primitive) && CheckStride(primitive);
}

OpCheckerRegistrar g_AvgPoolFusionChecker("AvgPoolFusion", new PoolingChecker());
OpCheckerRegistrar g_MaxPoolFusionChecker("MaxPoolFusion", new PoolingChecker());
}  // namespace dpico
}  // namespace mindspore

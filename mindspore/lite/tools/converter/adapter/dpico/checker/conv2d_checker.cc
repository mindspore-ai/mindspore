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

#include "checker/conv2d_checker.h"
#include <vector>
#include <string>
#include "common/anf_util.h"
#include "common/op_enum.h"
#include "common/check_base.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr int kMaxGroupNum = 2048;
constexpr int kMaxPadSize = 7;
constexpr int kMaxKernelSize = 15;
constexpr int kMaxStrideSize = 7;
constexpr int kMaxDilationSize = 5;
constexpr int kMaxDilationAndKernelProd = 15;
constexpr int kConvUpperBound = 2048;
constexpr float kCoefficient = 16.0;

bool CheckOutChannel(const api::PrimitivePtr &primitive) {
  auto out_channel_ptr = primitive->GetAttr(ops::kOutChannel);
  if (out_channel_ptr != nullptr) {
    auto out_channel_data = api::GetValue<int64_t>(out_channel_ptr);
    if (out_channel_data < 1 || out_channel_data > kMaxNumOutput) {
      MS_LOG(WARNING) << "out_channel:" << out_channel_data << " is unsupported by dpico.";
      return false;
    }
  }
  return true;
}
bool CheckGroup(const api::PrimitivePtr &primitive) {
  auto group_ptr = primitive->GetAttr(ops::kGroup);
  if (group_ptr != nullptr) {
    auto group_data = api::GetValue<int64_t>(group_ptr);
    if (group_data < 1 || group_data > kMaxGroupNum) {
      MS_LOG(WARNING) << "group:" << group_data << " is unsupported by dpico.";
      return false;
    }
  }
  return true;
}
bool CheckPadList(const api::PrimitivePtr &primitive) {
  auto pad_ptr = primitive->GetAttr(ops::kPadList);
  if (pad_ptr != nullptr) {
    auto pad_data = api::GetValue<std::vector<int64_t>>(pad_ptr);
    if (pad_data.size() > kDims3) {
      if (pad_data[0] < 0 || pad_data[0] > kMaxPadSize || pad_data[1] < 0 || pad_data[1] > kMaxPadSize ||
          pad_data[kAxis2] < 0 || pad_data[kAxis2] > kMaxPadSize || pad_data[kAxis3] < 0 ||
          pad_data[kAxis3] > kMaxPadSize) {
        MS_LOG(WARNING) << "pad size exceed the maximum pad size:" << kMaxPadSize;
        return false;
      }
    }
  }
  return true;
}
bool CheckKernelSize(const api::PrimitivePtr &primitive) {
  auto kernel_ptr = primitive->GetAttr(ops::kKernelSize);
  if (kernel_ptr != nullptr) {
    auto kernel_data = api::GetValue<std::vector<int64_t>>(kernel_ptr);
    if (kernel_data.size() > 1) {
      if (kernel_data[0] < 1 || kernel_data[0] > kMaxKernelSize || kernel_data[1] < 1 ||
          kernel_data[1] > kMaxKernelSize) {
        MS_LOG(WARNING) << "kernel size exceed the maximum kernel size:" << kMaxKernelSize;
        return false;
      }
    }
  }
  return true;
}
bool CheckStride(const api::PrimitivePtr &primitive) {
  auto stride_ptr = primitive->GetAttr(ops::kStride);
  if (stride_ptr != nullptr) {
    auto stride_data = api::GetValue<std::vector<int64_t>>(stride_ptr);
    if (stride_data.size() > 1) {
      if (stride_data[0] < 1 || stride_data[0] > kMaxStrideSize || stride_data[1] < 1 ||
          stride_data[1] > kMaxStrideSize) {
        MS_LOG(WARNING) << "stride[0]:" << stride_data[0] << " or stride[1]:" << stride_data[1]
                        << "is out of range [1,7]" << kMaxStrideSize;
        return false;
      }
    }
  }
  return true;
}
bool CheckDilation(const api::PrimitivePtr &primitive) {
  auto dilation_ptr = primitive->GetAttr(ops::kDilation);
  if (dilation_ptr != nullptr) {
    auto dilation_data = api::GetValue<std::vector<int64_t>>(dilation_ptr);
    if (dilation_data.size() > 1) {
      if (dilation_data[0] < 1 || dilation_data[0] > kMaxDilationSize || dilation_data[1] < 1 ||
          dilation_data[1] > kMaxDilationSize) {
        MS_LOG(WARNING) << "dilation[0]:" << dilation_data[0] << " or dilation[1]:" << dilation_data[1]
                        << "is out of range [1,5]" << kMaxDilationSize;
        return false;
      }
    }
  }
  return true;
}
bool CheckAttr(const api::CNodePtr &op, const api::PrimitivePtr &primitive, int64_t input_w) {
  auto dilation_ptr = primitive->GetAttr(ops::kDilation);
  auto kernel_ptr = primitive->GetAttr(ops::kKernelSize);
  auto stride_ptr = primitive->GetAttr(ops::kStride);
  auto output_paddings_ptr = primitive->GetAttr(ops::kOutputPaddings);

  if (dilation_ptr != nullptr && kernel_ptr != nullptr) {
    auto kernel_data = api::GetValue<std::vector<int64_t>>(kernel_ptr);
    auto dilation_data = api::GetValue<std::vector<int64_t>>(dilation_ptr);
    if ((kernel_data[0] - 1) * dilation_data[0] + 1 > kMaxDilationAndKernelProd ||
        (kernel_data[1] - 1) * dilation_data[1] + 1 > kMaxDilationAndKernelProd) {
      MS_LOG(WARNING) << "kernel should satisfy ((kernel - 1) * dilation + 1) less than 15";
      return false;
    }
  }
  if (stride_ptr != nullptr && kernel_ptr != nullptr) {
    auto kernel_data = api::GetValue<std::vector<int64_t>>(kernel_ptr);
    auto stride_data = api::GetValue<std::vector<int64_t>>(stride_ptr);
    if (CheckPrimitiveType(op, api::MakeShared<ops::Conv2DFusion>())) {
      MS_CHECK_TRUE_MSG(kCoefficient * stride_data[1] != 0, false, "kCoefficient * stride_data[1] should not be 0.");
      if (kernel_data[0] > kConvUpperBound / (input_w / (kCoefficient * stride_data[1]) * stride_data[1])) {
        MS_LOG(WARNING) << "kernel and stride should satisfy kernel_h <= 2048 / (w / (16 * stride) * stride) "
                        << op->fullname_with_scope();
        return false;
      }
    } else if (CheckPrimitiveType(op, api::MakeShared<ops::Conv2dTransposeFusion>())) {
      MS_CHECK_TRUE_MSG(input_w != 0, false, "input_w should not be 0.");
      if (kernel_data[0] > (kMaxNumOutput / input_w - 1) * stride_data[0] + 1) {
        MS_LOG(WARNING) << "kernel and stride should satisfy kernel_h <= (32768 / w - 1) * stride + 1 "
                        << op->fullname_with_scope();
        return false;
      }
    }
  }

  if (output_paddings_ptr != nullptr && CheckPrimitiveType(op, api::MakeShared<ops::Conv2dTransposeFusion>())) {
    auto output_paddings = api::GetValue<std::vector<int64_t>>(output_paddings_ptr);
    if (std::find_if_not(output_paddings.begin(), output_paddings.end(),
                         [](int64_t output_padding) { return output_padding == 0; }) != output_paddings.end()) {
      MS_LOG(WARNING) << "output_padding attr only support 0 by dpico. " << op->fullname_with_scope();
      return false;
    }
  }
  return true;
}
}  // namespace

bool Conv2DFusionChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (GetInputShapeFromCNode(op, kInputIndex1, &input_shape) == RET_OK && !input_shape.empty()) {
    if (input_shape.size() != kDims4) {
      MS_LOG(ERROR) << "Error conv2d input, which size should be 4";
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
    if (!CheckAttr(op, primitive, input_w)) {
      return false;
    }
  }
  return CheckOutChannel(primitive) && CheckGroup(primitive) && CheckPadList(primitive) && CheckKernelSize(primitive) &&
         CheckStride(primitive) && CheckDilation(primitive);
}

OpCheckerRegistrar g_Conv2DFusionChecker("Conv2DFusion", new Conv2DFusionChecker());
OpCheckerRegistrar g_Conv2dTransposeFusionChecker("Conv2dTransposeFusion", new Conv2DFusionChecker());
}  // namespace dpico
}  // namespace mindspore

/**
 * Copyright(c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unminimum required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "minimum.h"

#include "Eigen/Dense"
#include "cmath"
#include "cpu_kernel_utils.h"
#include "iostream"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const char *kMinimum = "Minimum";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 7 * 1024;
const int64_t kParallelDataNumSameShapeMid = 35 * 1024;

#define MINIMUM_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = MinimumCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Minimum kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t MinimumCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Minimum check input and output number failed.");
  KERNEL_HANDLE_ERROR(MinimumParamCheck(ctx), "Minimum check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();

  switch (data_type) {
    MINIMUM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    MINIMUM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    MINIMUM_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    MINIMUM_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MINIMUM_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Minimum kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t MinimumCpuKernel::MinimumParamCheck(CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *input_1 = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  DataType input0_type = input_0->GetDataType();
  DataType input1_type = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] need be same with "
                     "input1 [%s].",
                     DTypeStr(input0_type).c_str(), DTypeStr(input1_type).c_str())
  KERNEL_LOG_DEBUG(
    "MinimumCpuKernel[%s], input0: size[%llu];"
    "input1: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), input_1->GetDataSize(), output->GetDataSize());
  return KERNEL_STATUS_OK;
}

template <typename T>
void MinimumCpuKernel::SpecialComputeSameShape(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16) {
  auto input1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ignore_nan = false;
  auto ignore_nan_attr = ctx.GetAttr("ignore_nan");
  ignore_nan = (ignore_nan_attr == nullptr) ? false : ignore_nan_attr->GetBool();
  for (int64_t i = start; i < end; ++i) {
    if (ignore_nan == false && is_float16 == true) {
      if (Eigen::numext::isnan(*(input1 + i))) {
        *(output + i) = *(input1 + i);
      } else if (Eigen::numext::isnan(*(input2 + i))) {
        *(output + i) = *(input2 + i);
      } else {
        *(output + i) = *(input1 + i) < *(input2 + i) ? *(input1 + i) : *(input2 + i);
      }
    }
    if (ignore_nan == false && is_float16 == false) {
      if (isnan(*(input1 + i))) {
        *(output + i) = *(input1 + i);
      } else if (isnan(*(input2 + i))) {
        *(output + i) = *(input2 + i);
      } else {
        *(output + i) = *(input1 + i) < *(input2 + i) ? *(input1 + i) : *(input2 + i);
      }
    }
    if (ignore_nan == true && is_float16 == true) {
      if (Eigen::numext::isnan(*(input1 + i))) {
        *(output + i) = *(input2 + i);
      } else if (Eigen::numext::isnan(*(input2 + i))) {
        *(output + i) = *(input1 + i);
      } else {
        *(output + i) = *(input1 + i) < *(input2 + i) ? *(input1 + i) : *(input2 + i);
      }
    }
    if (ignore_nan == true && is_float16 == false) {
      if (isnan(*(input1 + i))) {
        *(output + i) = *(input2 + i);
      } else if (isnan(*(input2 + i))) {
        *(output + i) = *(input1 + i);
      } else {
        *(output + i) = *(input1 + i) < *(input2 + i) ? *(input1 + i) : *(input2 + i);
      }
    }
  }
}
template <typename T>
void MinimumCpuKernel::SpecialComputeXOneElement(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16) {
  auto input1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ignore_nan = false;
  auto ignore_nan_attr = ctx.GetAttr("ignore_nan");
  ignore_nan = (ignore_nan_attr == nullptr) ? false : ignore_nan_attr->GetBool();
  for (int64_t i = start; i < end; ++i) {
    if (ignore_nan == false && is_float16 == true) {
      if (Eigen::numext::isnan(*input1)) {
        *(output + i) = *input1;
      } else if (Eigen::numext::isnan(*(input2 + i))) {
        *(output + i) = *(input2 + i);
      } else {
        *(output + i) = *input1 < *(input2 + i) ? *input1 : *(input2 + i);
      }
    }
    if (ignore_nan == false && is_float16 == false) {
      if (isnan(*input1)) {
        *(output + i) = *input1;
      } else if (isnan(*(input2 + i))) {
        *(output + i) = *(input2 + i);
      } else {
        *(output + i) = *input1 < *(input2 + i) ? *input1 : *(input2 + i);
      }
    }
    if (ignore_nan == true && is_float16 == true) {
      if (Eigen::numext::isnan(*input1)) {
        *(output + i) = *(input2 + i);
      } else if (Eigen::numext::isnan(*(input2 + i))) {
        *(output + i) = *input1;
      } else {
        *(output + i) = *input1 < *(input2 + i) ? *input1 : *(input2 + i);
      }
    }
    if (ignore_nan == true && is_float16 == false) {
      if (isnan(*input1)) {
        *(output + i) = *(input2 + i);
      } else if (isnan(*(input2 + i))) {
        *(output + i) = *input1;
      } else {
        *(output + i) = *input1 < *(input2 + i) ? *input1 : *(input2 + i);
      }
    }
  }
}
template <typename T>
void MinimumCpuKernel::SpecialComputeYOneElement(int64_t start, int64_t end, CpuKernelContext &ctx, bool is_float16) {
  auto input1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ignore_nan = false;
  auto ignore_nan_attr = ctx.GetAttr("ignore_nan");
  ignore_nan = (ignore_nan_attr == nullptr) ? false : ignore_nan_attr->GetBool();
  for (int64_t i = start; i < end; ++i) {
    if (ignore_nan == false && is_float16 == true) {
      if (Eigen::numext::isnan(*(input1 + i))) {
        *(output + i) = *(input1 + i);
      } else if (Eigen::numext::isnan(*input2)) {
        *(output + i) = *input2;
      } else {
        *(output + i) = *(input1 + i) < *input2 ? *(input1 + i) : *input2;
      }
    }
    if (ignore_nan == false && is_float16 == false) {
      if (isnan(*(input1 + i))) {
        *(output + i) = *(input1 + i);
      } else if (isnan(*input2)) {
        *(output + i) = *input2;
      } else {
        *(output + i) = *(input1 + i) < *input2 ? *(input1 + i) : *input2;
      }
    }
    if (ignore_nan == true && is_float16 == true) {
      if (Eigen::numext::isnan(*(input1 + i))) {
        *(output + i) = *input2;
      } else if (Eigen::numext::isnan(*input2)) {
        *(output + i) = *(input1 + i);
      } else {
        *(output + i) = *(input1 + i) < *input2 ? *(input1 + i) : *input2;
      }
    }
    if (ignore_nan == true && is_float16 == false) {
      if (isnan(*(input1 + i))) {
        *(output + i) = *input2;
      } else if (isnan(*input2)) {
        *(output + i) = *(input1 + i);
      } else {
        *(output + i) = *(input1 + i) < *input2 ? *(input1 + i) : *input2;
      }
    }
  }
}

template <typename T>
void MinimumCpuKernel::SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext &ctx) {
  bool is_float16 = false;
  if (std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value) {
    is_float16 = false;
  } else {
    is_float16 = true;
  }
  switch (type) {
    case BcastShapeType::SAME_SHAPE:
      SpecialComputeSameShape<T>(start, end, ctx, is_float16);
      break;
    case BcastShapeType::X_ONE_ELEMENT:
      SpecialComputeXOneElement<T>(start, end, ctx, is_float16);
      break;
    case BcastShapeType::Y_ONE_ELEMENT:
      SpecialComputeYOneElement<T>(start, end, ctx, is_float16);
      break;
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
uint32_t MinimumCpuKernel::NoBcastCompute(CpuKernelContext &ctx) {
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = in0_elements_nums == in1_elements_nums
                          ? BcastShapeType::SAME_SHAPE
                          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_minimum = [&](int64_t start, int64_t end) { SpecialCompute<T>(type, start, end, ctx); };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_minimum),
                        "Minimum Compute failed.");
  } else {
    SpecialCompute<T>(type, 0, data_num, ctx);
  }

  return KERNEL_STATUS_OK;
}
template <typename T>
void MinimumCpuKernel::BcastComputeMultiKernel(int64_t start, int64_t end, CpuKernelContext &ctx, Bcast &bcast,
                                               bool is_float16) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ignore_nan = false;
  auto ignore_nan_attr = ctx.GetAttr("ignore_nan");
  ignore_nan = (ignore_nan_attr == nullptr) ? false : ignore_nan_attr->GetBool();
  for (int64_t i = start; i < end; ++i) {
    if (ignore_nan == false && is_float16 == true) {
      if (Eigen::numext::isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else if (Eigen::numext::isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
    if (ignore_nan == false && is_float16 == false) {
      if (isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else if (isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
    if (ignore_nan == true && is_float16 == true) {
      if (Eigen::numext::isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else if (Eigen::numext::isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }

    if (ignore_nan == true && is_float16 == false) {
      if (isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else if (isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
  }
}

template <typename T>
void MinimumCpuKernel::BcastComputeOneKernel(CpuKernelContext &ctx, Bcast &bcast, bool is_float16) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto ignore_nan = false;
  auto ignore_nan_attr = ctx.GetAttr("ignore_nan");
  ignore_nan = (ignore_nan_attr == nullptr) ? false : ignore_nan_attr->GetBool();
  int64_t data_num = ctx.Output(0)->NumElements();
  for (int64_t i = 0; i < data_num; ++i) {
    if (ignore_nan == false && is_float16 == true) {
      if (Eigen::numext::isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else if (Eigen::numext::isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
    if (ignore_nan == false && is_float16 == false) {
      if (isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else if (isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
    if (ignore_nan == true && is_float16 == true) {
      if (Eigen::numext::isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else if (Eigen::numext::isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
    if (ignore_nan == true && is_float16 == false) {
      if (isnan(*(in0 + bcast.GetBroadcastXIndex(i)))) {
        *(out + i) = *(in1 + bcast.GetBroadcastYIndex(i));
      } else if (isnan(*(in1 + bcast.GetBroadcastYIndex(i)))) {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i));
      } else {
        *(out + i) = *(in0 + bcast.GetBroadcastXIndex(i)) < *(in1 + bcast.GetBroadcastYIndex(i))
                       ? *(in0 + bcast.GetBroadcastXIndex(i))
                       : *(in1 + bcast.GetBroadcastYIndex(i));
      }
    }
  }
}

template <typename T>
uint32_t MinimumCpuKernel::BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  int64_t data_num = ctx.Output(0)->NumElements();
  bool is_float16 = false;
  if (std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value || std::is_same<T, float>::value ||
      std::is_same<T, double>::value) {
    is_float16 = false;
  } else {
    is_float16 = true;
  }
  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_minimum = [&](int64_t start, int64_t end) {
      BcastComputeMultiKernel<T>(start, end, ctx, bcast, is_float16);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_minimum),
                        "Minimum Compute failed.");
  } else {
    BcastComputeOneKernel<T>(ctx, bcast, is_float16);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MinimumCpuKernel::MinimumCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
  int64_t input0_elements_nums = input0_tensor->NumElements();

  Tensor *input1_tensor = ctx.Input(1);
  auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
  int64_t input1_elements_nums = input1_tensor->NumElements();
  bool isNeedBcast = (input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1);
  if (isNeedBcast) {
    return NoBcastCompute<T>(ctx);
  } else {
    Bcast bcast(input0_shape, input1_shape);
    if (!bcast.IsValid()) {
      KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }

    return BcastCompute<T>(ctx, bcast);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMinimum, MinimumCpuKernel);
}  // namespace aicpu

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

#include "igamma.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "igamma_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kigamma = "Igamma";
constexpr int64_t kParallelDataNums = 256;

#define SWITCH_PARALLEL(SHARD, data_num, max_core_num)                                                    \
  if ((data_num) <= kParallelDataNums) {                                                                  \
    SHARD(0, data_num);                                                                                   \
  } else {                                                                                                \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, (data_num), (data_num) / (max_core_num), SHARD), \
                        "Igamma SHARD Compute failed.");                                                  \
  }

#define IGAMMA_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                        \
    uint32_t result = IgammaCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                    \
      KERNEL_LOG_ERROR("Igamma kernel compute failed."); \
      return result;                                     \
    }                                                    \
    break;                                               \
  }

}  // namespace

namespace aicpu {
uint32_t IgammaCpuKernel::Compute(CpuKernelContext &ctx) {
  // check param number
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Igamma check input and output number failed.");

  KERNEL_HANDLE_ERROR(IgammaCheck(ctx), "Igamma check params failed.");

  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    IGAMMA_COMPUTE_CASE(DT_FLOAT, float, ctx)
    IGAMMA_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Igamma kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t IgammaCpuKernel::IgammaCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(kFirstInputIndex);
  auto input_1 = ctx.Input(kSecondInputIndex);
  auto output = ctx.Output(0);

  // check input datatype
  DataType input0_datatype = input_0->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == DT_DOUBLE || input0_datatype == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] data type must DT_FLOAT or DT_DOUBLE,"
                     "but got data type[%s].",
                     DTypeStr(input0_datatype).c_str());

  DataType input1_datatype = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == input1_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input1 [%s] need be same with "
                     "input0 [%s].",
                     DTypeStr(input1_datatype).c_str(), DTypeStr(input0_datatype).c_str())

  // check output dtype
  DataType output_datatype = output->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == output_datatype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output [%s] need be same with "
                     "input0 [%s].",
                     DTypeStr(output_datatype).c_str(), DTypeStr(input0_datatype).c_str())

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t BcastCompute(CpuKernelContext &ctx, Bcast &bcast) {
  auto input_x1 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_x2 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Output(0)->NumElements();

  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  if (max_core_num == 0) {
    max_core_num = 1;
  }

  if (data_num < kParallelDataNums) {
    auto cur_output = output_y;
    for (int64_t i = 0; i < data_num; i++) {
      T *x1_index = input_x1 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
      T *x2_index = input_x2 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
      *cur_output = IgammaSingle<T>(*x1_index, *x2_index);
      cur_output = cur_output + 1;
    }
  } else {
    auto shard_igamma = [&](size_t start, size_t end) {
      auto cur_output = output_y + start;
      for (size_t i = start; i < end; i++) {
        T *x1_index = input_x1 + bcast.GetBroadcastXIndex(i);  // i-th value of input0
        T *x2_index = input_x2 + bcast.GetBroadcastYIndex(i);  // i-th value of input1
        *cur_output = IgammaSingle<T>(*x1_index, *x2_index);
        cur_output = cur_output + 1;
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_igamma),
                        "Igamma SHARD Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

/* special compute is used in the following situations.
 * 1. the shapes of input1 and input2 are the same
 * 2. input1 is a 1D tensor with only one element or input1 is scalar
 * 3. input2 is a 1D tensor with only one element or input2 is scalar
 * 4. the shapes of input1 and input2 are different
 **/
template <typename T>
void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, const T *input1, const T *input2, T *output) {
  switch (type) {
    case BcastShapeType::SAME_SHAPE: {
      auto cur_input1 = input1 + start;
      auto cur_input2 = input2 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaSingle<T>(*cur_input1, *cur_input2);
        output = output + 1;
        cur_input1 = cur_input1 + 1;
        cur_input2 = cur_input2 + 1;
      }
      break;
    }
    case BcastShapeType::X_ONE_ELEMENT: {
      auto cur_input2 = input2 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaSingle<T>(*input1, *cur_input2);
        output = output + 1;
        cur_input2 = cur_input2 + 1;
      }
      break;
    }
    case BcastShapeType::Y_ONE_ELEMENT: {
      auto cur_input1 = input1 + start;
      for (int64_t i = start; i < end; ++i) {
        *output = IgammaSingle<T>(*cur_input1, *input2);
        output = output + 1;
        cur_input1 = cur_input1 + 1;
      }
      break;
    }
    default:
      KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
      break;
  }
}

template <typename T>
uint32_t NoBcastCompute(CpuKernelContext &ctx) {
  auto in0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto in1 = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  int64_t in0_elements_nums = ctx.Input(0)->NumElements();
  int64_t in1_elements_nums = ctx.Input(1)->NumElements();
  int64_t data_num = ctx.Output(0)->NumElements();
  BcastShapeType type = in0_elements_nums == in1_elements_nums
                          ? BcastShapeType::SAME_SHAPE
                          : (in0_elements_nums == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT);

  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

  if (max_core_num > data_num) {
    max_core_num = data_num;
  }

  if (data_num < kParallelDataNums) {
    SpecialCompute<T>(type, 0, data_num, in0, in1, out);
  } else {
    auto shard_igamma = [&](int64_t start, int64_t end) { SpecialCompute<T>(type, start, end, in0, in1, out + start); };

    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shard_igamma),
                        "Igamma SHARD Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t IgammaCpuKernel::IgammaCompute(CpuKernelContext &ctx) {
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

REGISTER_CPU_KERNEL(kigamma, IgammaCpuKernel);
}  // namespace aicpu
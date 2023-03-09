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
#include "cpu_kernel/ms_kernel/adaptive_avg_pool_2d.h"

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *kAdaptiveAvgPool2d = "AdaptiveAvgPool2D";
constexpr uint32_t kInputNum = 1;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 4 * 1024;
constexpr int64_t kthree = 3;
constexpr int64_t kneg_three = -3;
constexpr int64_t kfour = 4;
constexpr int64_t ktwo = 2;
constexpr int64_t kneg_two = -2;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  SCALAR_T *input_data = nullptr;
  SCALAR_T *output_data = nullptr;

  int64_t size_b = 1;
  int64_t size_d = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;

  int64_t out_size_h = 0;
  int64_t out_size_w = 0;

  int64_t in_stride_d = 0;
  int64_t in_stride_h = 0;
  int64_t in_stride_w = 0;
};

#define SWITCH_PARALLEL(SHARD, end_num, num)                                 \
  if ((num) <= kParallelDataNums) {                                          \
    for (size_t i = 0; i < size_t(end_num); i++) {                           \
      SHARD(i, i + 1);                                                       \
    }                                                                        \
  } else {                                                                   \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, 1, SHARD), \
                        "AdaptiveAvgPool2D #SHARD Compute failed.");         \
  }
}  // namespace

namespace aicpu {
template <typename SCALAR_T>
SCALAR_T ComputeSum(int64_t span_h, int64_t span_w, SCALAR_T *in_point, AdaptiveCalcArgs<SCALAR_T> &args) {
  SCALAR_T sum = static_cast<SCALAR_T>(0.);
  for (int in_h = 0; in_h < span_h; in_h++) {
    for (int in_w = 0; in_w < span_w; in_w++) {
      SCALAR_T val = *(in_point + in_h * args.in_stride_h + in_w * args.in_stride_w);

      sum += static_cast<SCALAR_T>(val);
    }
  }
  return sum;
}

template <typename SCALAR_T>
void ComputeSingleThread(int64_t start, int64_t end, AdaptiveCalcArgs<SCALAR_T> args) {
  for (auto d = start; d < end; d++) {
    /* loop over output */
    for (int64_t out_h = 0; out_h < args.out_size_h; out_h++) {
      int in_start_h = StartIndex(out_h, args.out_size_h, args.in_size_h);
      int in_end_h = EndIndex(out_h, args.out_size_h, args.in_size_h);
      int span_h = in_end_h - in_start_h;

      for (int64_t out_w = 0; out_w < args.out_size_w; out_w++) {
        int in_start_w = StartIndex(out_w, args.out_size_w, args.in_size_w);
        int in_end_w = EndIndex(out_w, args.out_size_w, args.in_size_w);
        int span_w = in_end_w - in_start_w;

        // local pointers
        SCALAR_T *in_point =
          args.input_data + d * args.in_stride_d + in_start_h * args.in_stride_h + in_start_w * args.in_stride_w;
        SCALAR_T *out_point =
          args.output_data + d * args.out_size_h * args.out_size_w + out_h * args.out_size_w + out_w;

        /* compute local average */
        /* set output to local average */
        *out_point = SCALAR_T(ComputeSum(span_h, span_w, in_point, args) / static_cast<SCALAR_T>(span_h * span_w));
      }
    }
  }
}

template <typename SCALAR_T>
uint32_t AdaptiveAvgPool2dOutFrame(CpuKernelContext &ctx, AdaptiveCalcArgs<SCALAR_T> args, int64_t num) {
  auto shard_frame = [&](int64_t start, int64_t end) { ComputeSingleThread(start, end, args); };
  SWITCH_PARALLEL(shard_frame, args.size_d, num);
  return KERNEL_STATUS_OK;
}

template <typename SCALAR_T>
uint32_t AdaptiveAvgPool2dOutTemplate(CpuKernelContext &ctx) {
  Tensor &input = *(ctx.Input(kFirstInputIndex));
  auto input_shape_ptr = input.GetTensorShape();
  int32_t input_dims = input_shape_ptr->GetDims();
  KERNEL_CHECK_NULLPTR(input_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input 0 shape failed.");

  KERNEL_CHECK_FALSE((input_dims == kthree || input_dims == kfour), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [3D] or [4D] (batch mode) tensor expected for input 0.");

  for (int32_t i = 0; i < input_dims; i++) {
    KERNEL_CHECK_FALSE((input_shape_ptr->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "AdaptiveAvgPool2D: expected input to have non-empty spatial "
                       "dimensions, "
                       "but input 0 has sizes [%d] with dimension [%d] being empty.",
                       input_dims, i);
  }

  AdaptiveCalcArgs<SCALAR_T> args;
  // sizes
  std::vector<int64_t> input_dim_sizes = input_shape_ptr->GetDimSizes();
  args.size_d = input_dim_sizes.end()[kneg_three];
  args.in_size_h = input_dim_sizes.end()[kneg_two];
  args.in_size_w = input_dim_sizes.end()[-1];

  // strides
  args.in_stride_w = 1;
  args.in_stride_h = args.in_size_w;
  args.in_stride_d = args.in_stride_h * args.in_size_h;

  // output sizes
  AttrValue *attr = ctx.GetAttr("output_size");
  std::vector<int64_t> output_size_data = attr->GetListInt();
  if (output_size_data.size() == ktwo) {
    args.out_size_h = output_size_data[0] > 0 ? output_size_data[0] : input_dim_sizes.end()[-2];
    args.out_size_w = output_size_data[1] > 0 ? output_size_data[1] : input_dim_sizes.end()[-1];
  } else if (output_size_data.size() == 1) {
    KERNEL_CHECK_FALSE((output_size_data[0] >= 0), KERNEL_STATUS_PARAM_INVALID,
                       "AdaptiveAvgPool2D: output_size value should be non-negative");
    args.out_size_h = output_size_data[0];
    args.out_size_w = output_size_data[0];
  } else {
    KERNEL_LOG_ERROR("output_size length should be 1 OR 2, but got [%d]", output_size_data.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // indices will contain i,j locations for each output point
  args.input_data = static_cast<SCALAR_T *>(input.GetData());
  args.output_data = static_cast<SCALAR_T *>(ctx.Output(kFirstOutputIndex)->GetData());
  int64_t num = input.NumElements();
  // resize output
  if (input_dims == kthree) {
    AdaptiveAvgPool2dOutFrame<SCALAR_T>(ctx, args, num);
  } else {
    auto shard_template = [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = args.input_data + b * args.in_stride_d * args.size_d;
        sub_args.output_data = args.output_data + b * args.size_d * args.out_size_h * args.out_size_w;
        AdaptiveAvgPool2dOutFrame<SCALAR_T>(ctx, sub_args, num);
      }
    };
    SWITCH_PARALLEL(shard_template, input_dim_sizes[0], num);
  }
  return KERNEL_STATUS_OK;
}

uint32_t AdaptiveAvgPool2d::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output number failed.",
                      kAdaptiveAvgPool2d);

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  auto data_type = static_cast<DataType>(input_0->GetDataType());
  // Compute by data_type
  switch (data_type) {
    case DT_FLOAT:
      return AdaptiveAvgPool2dOutTemplate<float>(ctx);
    case DT_DOUBLE:
      return AdaptiveAvgPool2dOutTemplate<double>(ctx);
    case DT_FLOAT16:
      return AdaptiveAvgPool2dOutTemplate<Eigen::half>(ctx);
    default:
      KERNEL_LOG_ERROR("AdaptiveAvgPool2D kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kAdaptiveAvgPool2d, AdaptiveAvgPool2d);
}  // namespace aicpu

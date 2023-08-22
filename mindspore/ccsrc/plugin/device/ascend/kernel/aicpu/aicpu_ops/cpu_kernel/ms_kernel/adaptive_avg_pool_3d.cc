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
#include "cpu_kernel/ms_kernel/adaptive_avg_pool_3d.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kAdaptiveAvgPool3d = "AdaptiveAvgPool3d";
constexpr uint32_t kInputNum = 1;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 4 * 1024;
constexpr int64_t kOneDim = 1;
constexpr int64_t kTwoDim = 2;
constexpr int64_t kThreeDim = 3;
constexpr int64_t kFourDim = 4;
constexpr int64_t kFiveDim = 5;
constexpr int32_t kPyValueNone = -1;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  SCALAR_T *input_data = nullptr;
  SCALAR_T *output_data = nullptr;

  int64_t size_d = 0;
  int64_t in_size_t = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;

  int64_t out_size_t = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;

  int64_t in_stride_d = 0;
  int64_t in_stride_t = 0;
  int64_t in_stride_h = 0;
  int64_t in_stride_w = 0;
};

inline int StartIndex(int offset, int out_size, int in_size) {
  assert(out_size != 0);
  return static_cast<int>(std::floor(static_cast<float>(offset * in_size) / out_size));
}

inline int EndIndex(int offset, int out_size, int in_size) {
  assert(out_size != 0);
  return static_cast<int>(std::ceil(static_cast<float>((offset + 1) * in_size) / out_size));
}
}  // namespace

namespace aicpu {
template <typename SCALAR_T>
uint32_t AdaptiveAvgPool3dOutFrame(const CpuKernelContext &ctx, AdaptiveCalcArgs<SCALAR_T> args, int64_t num) {
  auto shard_frame = [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      // calculate average
      for (int64_t out_t = 0; out_t < args.out_size_t; out_t++) {
        int in_start_t = StartIndex(out_t, args.out_size_t, args.in_size_t);
        int in_end_t = EndIndex(out_t, args.out_size_t, args.in_size_t);
        int span_t = in_end_t - in_start_t;

        for (int64_t out_h = 0; out_h < args.out_size_h; out_h++) {
          int in_start_h = StartIndex(out_h, args.out_size_h, args.in_size_h);
          int in_end_h = EndIndex(out_h, args.out_size_h, args.in_size_h);
          int span_h = in_end_h - in_start_h;

          for (int64_t out_w = 0; out_w < args.out_size_w; out_w++) {
            int in_start_w = StartIndex(out_w, args.out_size_w, args.in_size_w);
            int in_end_w = EndIndex(out_w, args.out_size_w, args.in_size_w);
            int span_w = in_end_w - in_start_w;

            // local pointers
            SCALAR_T *in_point = args.input_data + d * args.in_stride_d + in_start_t * args.in_stride_t +
                                 in_start_h * args.in_stride_h + in_start_w * args.in_stride_w;
            SCALAR_T *out_point = args.output_data + d * args.out_size_t * args.out_size_h * args.out_size_w +
                                  out_t * args.out_size_h * args.out_size_w + out_h * args.out_size_w + out_w;

            // compute local average
            double sum = 0;
            for (int in_t = 0; in_t < span_t; in_t++) {
              for (int in_h = 0; in_h < span_h; in_h++) {
                for (int in_w = 0; in_w < span_w; in_w++) {
                  SCALAR_T val =
                    *(in_point + in_t * args.in_stride_t + in_h * args.in_stride_h + in_w * args.in_stride_w);

                  sum += static_cast<double>(val);
                }
              }
            }
            // set output to local average
            *out_point = SCALAR_T(sum / span_t / span_h / span_w);
          }
        }
      }
    }
  };
  if (num <= kParallelDataNums) {
    for (size_t i = 0; i < size_t(args.size_d); i++) {
      shard_frame(i, i + 1);
    }
  } else {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, args.size_d, 1, shard_frame),
                        "AdaptiveAvgPool3d shard_frame Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename SCALAR_T>
uint32_t AdaptiveAvgPool3dOutTemplate(const CpuKernelContext &ctx) {
  Tensor &input = *(ctx.Input(kFirstInputIndex));

  auto input_shape_ptr = input.GetTensorShape();
  int32_t input_dims = input_shape_ptr->GetDims();

  KERNEL_CHECK_FALSE((input_dims == kFourDim || input_dims == kFiveDim), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] or [5D] (batch mode) tensor expected for input 0.");

  for (int32_t i = 0; i < input_dims; i++) {
    KERNEL_CHECK_FALSE((input_shape_ptr->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "Adaptive_avg_pool3d: expected input to have non-empty spatial "
                       "dimensions, "
                       "but input 0 has sizes [%d] with dimension [%d] being empty.",
                       input_dims, i);
  }

  AdaptiveCalcArgs<SCALAR_T> args;
  // sizes
  std::vector<int64_t> input_dim_sizes = input_shape_ptr->GetDimSizes();
  std::vector<int64_t> output_shape(input_dim_sizes);
  args.size_d = input_dim_sizes.end()[-kFourDim];
  args.in_size_t = input_dim_sizes.end()[-kThreeDim];
  args.in_size_h = input_dim_sizes.end()[-kTwoDim];
  args.in_size_w = input_dim_sizes.end()[-kOneDim];

  // strides
  args.in_stride_w = 1;
  args.in_stride_h = args.in_size_w;
  args.in_stride_t = args.in_stride_h * args.in_size_h;
  args.in_stride_d = args.in_stride_t * args.in_size_t;

  // output sizes
  AttrValue *attr = ctx.GetAttr("output_size");
  std::vector<int64_t> output_size_data = attr->GetListInt();
  for (int i = 0; i < kThreeDim; i++) {
    if (output_size_data[i] != kPyValueNone) {
      output_shape.end()[i - kThreeDim] = output_size_data[i];
    }
  }
  args.out_size_t = output_shape.end()[-kThreeDim];
  args.out_size_h = output_shape.end()[-kTwoDim];
  args.out_size_w = output_shape.end()[-kOneDim];

  args.input_data = static_cast<SCALAR_T *>(input.GetData());
  args.output_data = static_cast<SCALAR_T *>(ctx.Output(kFirstOutputIndex)->GetData());
  int64_t num = input.NumElements();

  // resize output
  if (input_dims == kFourDim) {
    AdaptiveAvgPool3dOutFrame<SCALAR_T>(ctx, args, num);
  } else {
    auto shard_template = [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = args.input_data + b * args.in_stride_d * args.size_d;
        sub_args.output_data = args.output_data + b * args.size_d * args.out_size_t * args.out_size_h * args.out_size_w;
        AdaptiveAvgPool3dOutFrame<SCALAR_T>(ctx, sub_args, num);
      }
    };
    if (num <= kParallelDataNums) {
      for (size_t i = 0; i < size_t(input_dim_sizes[0]); i++) {
        shard_template(i, i + 1);
      }
    } else {
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, input_dim_sizes[0], 1, shard_template),
                          "AdaptiveAvgPool3d shard_template Compute failed.");
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t AdaptiveAvgPool3d::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output number failed.",
                      kAdaptiveAvgPool3d);

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  auto data_type = static_cast<DataType>(input_0->GetDataType());
  // Compute by data_type
  switch (data_type) {
    case DT_INT8:
      return AdaptiveAvgPool3dOutTemplate<int8_t>(ctx);
    case DT_UINT8:
      return AdaptiveAvgPool3dOutTemplate<uint8_t>(ctx);
    case DT_INT16:
      return AdaptiveAvgPool3dOutTemplate<int16_t>(ctx);
    case DT_INT32:
      return AdaptiveAvgPool3dOutTemplate<int32_t>(ctx);
    case DT_INT64:
      return AdaptiveAvgPool3dOutTemplate<int64_t>(ctx);
    case DT_FLOAT:
      return AdaptiveAvgPool3dOutTemplate<float>(ctx);
    case DT_DOUBLE:
      return AdaptiveAvgPool3dOutTemplate<double>(ctx);
    case DT_FLOAT16:
      return AdaptiveAvgPool3dOutTemplate<Eigen::half>(ctx);
    default:
      KERNEL_LOG_ERROR("AdaptiveAvgPool3d kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kAdaptiveAvgPool3d, AdaptiveAvgPool3d);
}  // namespace aicpu

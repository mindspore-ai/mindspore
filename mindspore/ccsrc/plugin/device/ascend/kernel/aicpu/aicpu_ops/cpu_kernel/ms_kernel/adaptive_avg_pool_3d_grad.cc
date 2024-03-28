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

#include "cpu_kernel/ms_kernel/adaptive_avg_pool_3d_grad.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kAdaptiveAvgPool3dGrad = "AdaptiveAvgPool3dGrad";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 1024 * 2;
constexpr int64_t kOneDim = 1;
constexpr int64_t kTwoDim = 2;
constexpr int64_t kThreeDim = 3;
constexpr int64_t kFourDim = 4;
constexpr int64_t kFiveDim = 5;

template <typename SCALAR_T>
struct AdaptiveCalcArgs {
  double *input_data = nullptr;
  SCALAR_T *output_data = nullptr;

  int64_t size_d = 0;

  int64_t in_size_t = 0;
  int64_t in_size_h = 0;
  int64_t in_size_w = 0;

  int64_t out_size_t = 0;
  int64_t out_size_h = 0;
  int64_t out_size_w = 0;
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
uint32_t AdaptiveAvgPool3dGradOutFrame(CpuKernelContext &ctx, AdaptiveCalcArgs<SCALAR_T> args, int64_t num) {
  auto shard_frame = [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      double *grad_input_p_d = args.input_data + d * args.in_size_t * args.in_size_w * args.in_size_h;
      SCALAR_T *grad_output_p_d = args.output_data + d * args.out_size_t * args.out_size_w * args.out_size_h;

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
            auto local_grad =
              grad_output_p_d[out_t * args.out_size_h * args.out_size_w + out_h * args.out_size_w + out_w];
            double grad_delta = static_cast<double>(local_grad) / span_t / span_h / span_w;
            for (int in_t = in_start_t; in_t < in_end_t; in_t++) {
              for (int in_h = in_start_h; in_h < in_end_h; in_h++) {
                for (int in_w = in_start_w; in_w < in_end_w; in_w++) {
                  grad_input_p_d[in_t * args.in_size_h * args.in_size_w + in_h * args.in_size_w + in_w] += grad_delta;
                }
              }
            }
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
    CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, args.size_d, 1, shard_frame),
                             "AdaptiveAvgPool3dGrad shard_frame Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename SCALAR_T>
uint32_t AdaptiveAvgPool3dGradOutTemplate(CpuKernelContext &ctx) {
  Tensor &orig_input_shape = *(ctx.Input(kSecondInputIndex));

  auto orig_input_shape_shape = orig_input_shape.GetTensorShape();
  CUST_KERNEL_CHECK_FALSE(ctx, orig_input_shape_shape->GetDims() == 1, KERNEL_STATUS_PARAM_INVALID,
                          "Non-empty [1D] tensor expected for orig_input_shape.");

  int32_t orig_input_shape_dims = orig_input_shape_shape->GetDimSize(0);

  CUST_KERNEL_CHECK_FALSE(ctx, (orig_input_shape_dims == kFourDim || orig_input_shape_dims == kFiveDim),
                          KERNEL_STATUS_PARAM_INVALID, "Non-empty [4D] or [5D] tensor expected for orig_input.");

  auto orig_input_shape_data = static_cast<int32_t *>(orig_input_shape.GetData());

  for (int32_t i = 0; i < orig_input_shape_dims; i++) {
    CUST_KERNEL_CHECK_FALSE(ctx, (orig_input_shape_data[i] > 0), KERNEL_STATUS_PARAM_INVALID,
                            "AdaptiveAvgPool3dGrad: expected orig_input to have "
                            "non-empty spatial dimensions");
  }

  /**
   * grad_output is the grad of output of AdaptiveAvgPool3d,
   * which is also "input_grad" of AdaptiveAvgPool3dGrad
   */
  Tensor &grad_output = *(ctx.Input(kFirstInputIndex));
  auto grad_output_shape_ptr = grad_output.GetTensorShape();
  int32_t grad_output_dims = grad_output_shape_ptr->GetDims();

  CUST_KERNEL_CHECK_FALSE(ctx, (grad_output_dims == kFourDim || grad_output_dims == kFiveDim),
                          KERNEL_STATUS_PARAM_INVALID,
                          "Non-empty [4D] or [5D] (batch mode) tensor expected for input 0.");

  AdaptiveCalcArgs<SCALAR_T> args;
  args.size_d = orig_input_shape_data[orig_input_shape_dims - kFourDim];
  args.in_size_t = orig_input_shape_data[orig_input_shape_dims - kThreeDim];
  args.in_size_h = orig_input_shape_data[orig_input_shape_dims - kTwoDim];
  args.in_size_w = orig_input_shape_data[orig_input_shape_dims - kOneDim];

  // sizes
  std::vector<int64_t> grad_output_dim_sizes = grad_output_shape_ptr->GetDimSizes();
  args.out_size_t = grad_output_dim_sizes.end()[-kThreeDim];
  args.out_size_h = grad_output_dim_sizes.end()[-kTwoDim];
  args.out_size_w = grad_output_dim_sizes.end()[-kOneDim];

  // indices will contain i,j locations for each output point
  auto input_data_ptr_ret = static_cast<SCALAR_T *>(ctx.Output(kFirstOutputIndex)->GetData());
  auto output_num = ctx.Output(kFirstOutputIndex)->NumElements();
  auto input_data_ptr = new double[output_num];
  std::fill_n(input_data_ptr, output_num, 0.0);
  auto output_data_ptr = static_cast<SCALAR_T *>(grad_output.GetData());

  int64_t num = orig_input_shape_shape->NumElements();

  // resize output
  if (orig_input_shape_dims == kFourDim) {
    args.input_data = input_data_ptr;
    args.output_data = output_data_ptr;
    AdaptiveAvgPool3dGradOutFrame<SCALAR_T>(ctx, args, num);
  } else {
    auto shard_template = [&](int64_t start, int64_t end) {
      for (auto b = start; b < end; b++) {
        AdaptiveCalcArgs<SCALAR_T> sub_args = args;
        sub_args.input_data = input_data_ptr + b * args.size_d * args.in_size_t * args.in_size_h * args.in_size_w;
        sub_args.output_data = output_data_ptr + b * args.size_d * args.out_size_t * args.out_size_h * args.out_size_w;

        AdaptiveAvgPool3dGradOutFrame<SCALAR_T>(ctx, sub_args, num);
      }
    };
    if (num <= kParallelDataNums) {
      for (size_t i = 0; i < size_t(orig_input_shape_data[0]); i++) {
        shard_template(i, i + 1);
      }
    } else {
      CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, orig_input_shape_data[0], 1, shard_template),
                               "AdaptiveAvgPool3dGrad shard_template Compute failed.");
    }
  }
  for (int64_t i = 0; i < output_num; i++) {
    input_data_ptr_ret[i] = static_cast<SCALAR_T>(input_data_ptr[i]);
  }
  delete input_data_ptr;
  return KERNEL_STATUS_OK;
}

uint32_t AdaptiveAvgPool3dGrad::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output number failed.",
                           kAdaptiveAvgPool3dGrad);

  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  auto data_type = static_cast<DataType>(input_0->GetDataType());
  // Compute by data_type
  switch (data_type) {
    case DT_INT8:
      return AdaptiveAvgPool3dGradOutTemplate<int8_t>(ctx);
    case DT_UINT8:
      return AdaptiveAvgPool3dGradOutTemplate<uint8_t>(ctx);
    case DT_INT16:
      return AdaptiveAvgPool3dGradOutTemplate<int16_t>(ctx);
    case DT_INT32:
      return AdaptiveAvgPool3dGradOutTemplate<int32_t>(ctx);
    case DT_INT64:
      return AdaptiveAvgPool3dGradOutTemplate<int64_t>(ctx);
    case DT_FLOAT:
      return AdaptiveAvgPool3dGradOutTemplate<float>(ctx);
    case DT_DOUBLE:
      return AdaptiveAvgPool3dGradOutTemplate<double>(ctx);
    case DT_FLOAT16:
      return AdaptiveAvgPool3dGradOutTemplate<Eigen::half>(ctx);
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "AdaptiveAvgPool3dGrad kernel data type [%s] not support.",
                            DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kAdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGrad);
}  // namespace aicpu

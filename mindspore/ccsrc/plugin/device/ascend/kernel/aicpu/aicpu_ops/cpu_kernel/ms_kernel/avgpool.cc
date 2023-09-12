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

#include "cpu_kernel/ms_kernel/avgpool.h"
#include <vector>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kAvgPool = "AvgPool";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 2 * 1024;
const int64_t kParallelDataNumMid = 16 * 1024;
const char *defaultDataFormat = "NCHW";

#define AVGPOOL_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = AvgPoolCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("AvgPool kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }

}  // namespace

namespace aicpu {
uint32_t AvgPoolCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "AvgPool check input and output number failed.");
  KERNEL_HANDLE_ERROR(AvgPoolParamCheck(ctx), "AvgPool check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    AVGPOOL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    AVGPOOL_COMPUTE_CASE(DT_FLOAT, float, ctx)
    AVGPOOL_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("AvgPool kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t AvgPoolCpuKernel::AvgPoolParamCheck(const CpuKernelContext &ctx) {
  // the non null of input_0, input_1, output has been verified in NormalCheck
  Tensor *input_0 = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("ksize"), KERNEL_STATUS_PARAM_INVALID, "Attr ksize can't be null")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("strides"), KERNEL_STATUS_PARAM_INVALID, "Attr strides can't be null")
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("padding"), KERNEL_STATUS_PARAM_INVALID, "Attr padding can't be null")
  KERNEL_LOG_DEBUG(
    "AvgPoolCpuKernel[%s], input0: size[%llu];"
    "output: size[%llu].",
    ctx.GetOpType().c_str(), input_0->GetDataSize(), output->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AvgPoolCpuKernel::AvgPoolProcess(const CpuKernelContext &ctx, AvgPoolCalcArgs args) {
  // NCHW
  auto input0 = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output0 = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  int64_t data_num = ctx.Output(0)->NumElements();

  if (data_num >= kParallelDataNum && args.batch_size > 1) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    // the core_num is not more than the batchsize
    if (max_core_num > args.batch_size) {
      max_core_num = static_cast<uint32_t>(args.batch_size);
    }

    auto sharder_avgpool = [&](int64_t start, int64_t end) {
      if (args.data_format == "NCHW") {
        RealComputeNCHW<T>(start, end, args, input0, output0);
      } else {
        RealComputeNHWC<T>(start, end, args, input0, output0);
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, args.batch_size, args.batch_size / max_core_num, sharder_avgpool),
      "AvgPool Compute failed.");
  } else {
    if (args.data_format == "NCHW") {
      RealComputeNCHW<T>(0, args.batch_size, args, input0, output0);
    } else {
      RealComputeNHWC<T>(0, args.batch_size, args, input0, output0);
    }
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
void AvgPoolCpuKernel::RealComputeNCHW(int64_t start, int64_t end, AvgPoolCalcArgs args, T *input0, T *output0) {
  for (int64_t n = start; n < end; n++) {
    for (int64_t c = 0; c < args.out_size_c; c++) {
      for (int64_t offset_h = 0; offset_h < args.out_size_h; offset_h++) {
        int64_t in_start_h = offset_h * args.stride_h;
        int64_t in_end_h = offset_h * args.stride_h + args.window_h;

        for (int64_t offset_w = 0; offset_w < args.out_size_w; offset_w++) {
          int64_t in_start_w = offset_w * args.stride_w;
          int64_t in_end_w = offset_w * args.stride_w + args.window_w;
          // local pointers
          T *in_point = input0 + n * args.image_size + c * args.in_size_h * args.in_size_w;
          T *out_point = output0 + n * args.out_size_h * args.out_size_w * args.out_size_c +
                         c * args.out_size_h * args.out_size_w + offset_h * args.out_size_w + offset_w;

          // compute local avg:
          int64_t ih = 0;
          int64_t iw = 0;
          T avg_val = static_cast<T>(0);
          T window_element_num =
            static_cast<T>((std::min(in_end_h, args.in_size_h + args.pad_top) - std::max(in_start_h, args.pad_top)) *
                           (std::min(in_end_w, args.in_size_w + args.pad_left) - std::max(in_start_w, args.pad_left)));
          for (ih = in_start_h; ih < in_end_h; ih++) {
            for (iw = in_start_w; iw < in_end_w; iw++) {
              if (ih < args.pad_top || ih >= args.pad_top + args.in_size_h || iw < args.pad_left ||
                  iw >= args.pad_left + args.in_size_w) {
                continue;
              }
              avg_val += *(in_point + (ih - args.pad_top) * args.in_size_w + (iw - args.pad_left)) / window_element_num;
            }
          }
          // set output to local avg
          *out_point = avg_val;
        }
      }
    }
  }
}

template <typename T>
void AvgPoolCpuKernel::RealComputeNHWC(int64_t start, int64_t end, AvgPoolCalcArgs args, T *input0, T *output0) {
  for (int64_t n = start; n < end; n++) {
    for (int64_t offset_h = 0; offset_h < args.out_size_h; offset_h++) {
      int64_t in_start_h = offset_h * args.stride_h;
      int64_t in_end_h = offset_h * args.stride_h + args.window_h;

      for (int64_t offset_w = 0; offset_w < args.out_size_w; offset_w++) {
        int64_t in_start_w = offset_w * args.stride_w;
        int64_t in_end_w = offset_w * args.stride_w + args.window_w;

        T *in_point = input0 + n * args.image_size;
        for (int64_t c = 0; c < args.out_size_c; c++) {
          // local pointers
          T *out_point = output0 + n * args.out_size_h * args.out_size_w * args.out_size_c +
                         offset_h * args.out_size_w * args.out_size_c + offset_w * args.out_size_c + c;
          // compute local avg:
          int64_t ih = 0;
          int64_t iw = 0;
          T avg_val = static_cast<T>(0);
          T window_element_num =
            static_cast<T>((std::min(in_end_h, args.in_size_h + args.pad_top) - std::max(in_start_h, args.pad_top)) *
                           (std::min(in_end_w, args.in_size_w + args.pad_left) - std::max(in_start_w, args.pad_left)));
          for (ih = in_start_h; ih < in_end_h; ih++) {
            for (iw = in_start_w; iw < in_end_w; iw++) {
              if (ih < args.pad_top || ih >= args.pad_top + args.in_size_h || iw < args.pad_left ||
                  iw >= args.pad_left + args.in_size_w) {
                continue;
              }
              avg_val += *(in_point + (ih - args.pad_top) * args.in_size_w * args.in_size_c +
                           (iw - args.pad_left) * args.in_size_c + c) /
                         window_element_num;
            }
          }
          // set output to local avg
          *out_point = avg_val;
        }
      }
    }
  }
}

template <typename T>
uint32_t AvgPoolCpuKernel::AvgPoolCompute(const CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();

  Tensor *output0_tensor = ctx.Output(0);
  auto output0_shape = output0_tensor->GetTensorShape()->GetDimSizes();

  std::vector<int64_t> strides = ctx.GetAttr("strides")->GetListInt();
  std::vector<int64_t> ksize = ctx.GetAttr("ksize")->GetListInt();
  std::string padding = ctx.GetAttr("padding")->GetString();
  std::string data_format =
    ctx.GetAttr("data_format") == nullptr ? defaultDataFormat : ctx.GetAttr("data_format")->GetString();

  auto n_position = data_format.find("N");
  auto c_position = data_format.find("C");
  auto h_position = data_format.find("H");
  auto w_position = data_format.find("W");
  AvgPoolCalcArgs args;
  args.batch_size = input0_shape[n_position];

  args.in_size_c = input0_shape[c_position];
  args.in_size_h = input0_shape[h_position];
  args.in_size_w = input0_shape[w_position];

  args.out_size_c = output0_shape[c_position];
  args.out_size_h = output0_shape[h_position];
  args.out_size_w = output0_shape[w_position];

  args.stride_h = strides[h_position];
  args.stride_w = strides[w_position];

  args.window_h = ksize[h_position];
  args.window_w = ksize[w_position];

  args.image_size = args.in_size_c * args.in_size_h * args.in_size_w;
  args.data_format = data_format;

  if (padding == "SAME") {
    args.pad_h = std::max((args.out_size_h - 1) * args.stride_h + args.window_h - args.in_size_h, 0L);
    args.pad_top = floor(args.pad_h / 2);
    args.pad_bottom = args.pad_h - args.pad_top;
    args.pad_w = std::max((args.out_size_w - 1) * args.stride_w + args.window_w - args.in_size_w, 0L);
    args.pad_left = floor(args.pad_w / 2);
    args.pad_right = args.pad_w - args.pad_left;
  }
  return AvgPoolProcess<T>(ctx, args);
}

REGISTER_CPU_KERNEL(kAvgPool, AvgPoolCpuKernel);
}  // namespace aicpu

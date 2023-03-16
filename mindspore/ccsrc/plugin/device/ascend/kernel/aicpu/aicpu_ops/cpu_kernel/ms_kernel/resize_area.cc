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

#include "resize_area.h"
#include <securec.h>
#include <vector>
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
constexpr uint32_t kIndex4 = 4;
const int64_t kParallelDataNum = 1024 * 1024;
const char *kResizeArea = "ResizeArea";

#define RESIZEAREA_COMPUTE_CASE(DTYPE, CHANNEL, TYPE, CTX)          \
  case (DTYPE): {                                                   \
    uint32_t result = DoCompute<TYPE>(st, x_interps, CHANNEL, CTX); \
    if (result != KERNEL_STATUS_OK) {                               \
      KERNEL_LOG_ERROR("ResizeArea kernel compute failed.");        \
      return result;                                                \
    }                                                               \
    break;                                                          \
  }

inline int64_t Bound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }

float Scaling_(size_t in_size, size_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}
}  // namespace

namespace aicpu {
void ResizeAreaSt::CalSt(CpuKernelContext &ctx, std::vector<int64_t> &in_shape1, bool align_corners) {
  Tensor *input_tensor2 = ctx.Input(1);
  auto outsize = reinterpret_cast<int32_t *>(input_tensor2->GetData());
  batch_size = in_shape1[0];
  channels = in_shape1[kIndex3];
  in_height = in_shape1[1];
  in_width = in_shape1[kIndex2];
  out_height = outsize[0];
  out_width = outsize[1];
  height_scale = Scaling_(in_height, out_height, align_corners);
  width_scale = Scaling_(in_width, out_width, align_corners);
}

uint32_t ResizeAreaCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE(res == KERNEL_STATUS_OK, res, "GetInputAndCheck failed.");
  ResizeAreaSt st;
  st.CalSt(ctx, in_shape1, align_corners);
  // compute the weight of pixels in rows
  std::vector<ResizeAreaCachedInterpolation> x_interps(st.out_width);
  for (size_t x = 0; x < st.out_width; x++) {
    auto &x_interp = x_interps[x];
    const float transit_x0 = x * st.width_scale;
    const float transit_x1 = (x + 1) * st.width_scale;
    size_t v = std::floor(transit_x0);
    x_interp.start = v;
    x_interp.start_scale = (v + 1 > transit_x1 ? st.width_scale : v + 1 - transit_x0);
    v = std::ceil(transit_x1);
    x_interp.end = v;
    v = x_interp.end - 1;
    x_interp.end_minus_one_scale = (v + 1 > transit_x1 ? transit_x1 - v : 1.0);
  }
  auto channels_num = -1;
  if (st.channels == kIndex3) {
    channels_num = kIndex3;
  }

  switch (dtype_) {
    RESIZEAREA_COMPUTE_CASE(DT_INT8, channels_num, int8_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_INT16, channels_num, int16_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_INT32, channels_num, int32_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_INT64, channels_num, int64_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_UINT8, channels_num, uint8_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_UINT16, channels_num, uint16_t, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_FLOAT, channels_num, float, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_FLOAT16, channels_num, Eigen::half, ctx)
    RESIZEAREA_COMPUTE_CASE(DT_DOUBLE, channels_num, double, ctx)
    default:
      KERNEL_LOG_ERROR("ResizeArea doesn't support input tensor types: [%s]", DTypeStr(dtype_).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ResizeAreaCpuKernel::DoCompute(const ResizeAreaSt &st, std::vector<ResizeAreaCachedInterpolation> &x_interps,
                                        int64_t kKnownNumChannels, CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_ptr = reinterpret_cast<float *>(ctx.Output(0)->GetData());
  int64_t data_num = ctx.Input(0)->NumElements();
  float scale = 1.0 / (st.height_scale * st.width_scale);

  if (data_num >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > st.out_height) {
      max_core_num = st.out_height;
    }
    for (size_t b = 0; b < st.batch_size; ++b) {
      auto shared_resize_area = [&](size_t start, size_t end) {
        // compute the weight of pixels in columns
        for (size_t y = start; y < end; ++y) {
          const float transit_y0 = y * st.height_scale;
          const float transit_y1 = (y + 1) * st.height_scale;
          // The start and end height indices of all the cells that could
          // contribute to the target cell.
          const int64_t y_start = std::floor(transit_y0);
          const int64_t y_end = std::ceil(transit_y1);
          std::vector<float> y_scales;
          std::vector<const T *> y_ptrs;
          y_scales.clear();
          y_ptrs.clear();
          for (int64_t i = y_start; i < y_end; ++i) {
            float scale_y;
            if (i < transit_y0) {
              scale_y = (i + 1 > transit_y1 ? st.height_scale : i + 1 - transit_y0);
            } else {
              scale_y = (i + 1 > transit_y1 ? transit_y1 - i : 1.0);
            }
            y_scales.push_back(scale_y);
            y_ptrs.push_back(input_ptr + (b * st.in_height * st.in_width * st.channels +
                                          Bound(i, st.in_height) * st.in_width * st.channels));
          }
          float *output_patch_ptr =
            output_ptr + (b * st.out_height * st.out_width * st.channels + y * st.out_width * st.channels);
          if (kKnownNumChannels == kIndex3) {
            for (size_t x = 0; x < st.out_width; ++x) {
              const ResizeAreaCachedInterpolation &x_interp = x_interps[x];
              if (x_interp.needs_bounding) {
                ComputePatchSumOf3Channels<true>(scale, st, y_ptrs, y_scales, x_interp, output_patch_ptr);
              } else {
                ComputePatchSumOf3Channels<false>(scale, st, y_ptrs, y_scales, x_interp, output_patch_ptr);
              }
              output_patch_ptr += kIndex3;
            }
          } else {
            for (size_t x = 0; x < st.out_width; ++x) {
              const ResizeAreaCachedInterpolation &x_interp = x_interps[x];
              if (x_interp.needs_bounding) {
                ComputePatchSum<true>(scale, st, y_ptrs, y_scales, x_interp, output_patch_ptr);
              } else {
                ComputePatchSum<false>(scale, st, y_ptrs, y_scales, x_interp, output_patch_ptr);
              }
              output_patch_ptr += st.channels;
            }
          }
        }
      };
      CpuKernelUtils::ParallelFor(ctx, st.out_height, st.out_height / max_core_num, shared_resize_area);
    }
  } else {
    std::vector<float> y_scales;
    std::vector<const T *> y_ptrs;
    for (size_t b = 0; b < st.batch_size; ++b) {
      for (size_t y = 0; y < st.out_height; ++y) {
        y_scales.clear();
        y_ptrs.clear();
        const float transit_y0 = y * st.height_scale;
        const float transit_y1 = (y + 1) * st.height_scale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        const size_t y_start = std::floor(transit_y0);
        const size_t y_end = std::ceil(transit_y1);
        for (size_t i = y_start; i < y_end; ++i) {
          float scale_y;
          if (i < transit_y0) {
            scale_y = (i + 1 > transit_y1 ? st.height_scale : i + 1 - transit_y0);
          } else {
            scale_y = (i + 1 > transit_y1 ? transit_y1 - i : 1.0);
          }
          y_scales.push_back(scale_y);
          y_ptrs.push_back(input_ptr + (b * st.in_height * st.in_width * st.channels +
                                        Bound(i, st.in_height) * st.in_width * st.channels));
        }
        if (kKnownNumChannels == kIndex3) {
          for (size_t x = 0; x < st.out_width; ++x) {
            const ResizeAreaCachedInterpolation &x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSumOf3Channels<true>(scale, st, y_ptrs, y_scales, x_interp, output_ptr);
            } else {
              ComputePatchSumOf3Channels<false>(scale, st, y_ptrs, y_scales, x_interp, output_ptr);
            }
            output_ptr += kIndex3;
          }
        } else {
          for (size_t x = 0; x < st.out_width; ++x) {
            const ResizeAreaCachedInterpolation &x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSum<true>(scale, st, y_ptrs, y_scales, x_interp, output_ptr);
            } else {
              ComputePatchSum<false>(scale, st, y_ptrs, y_scales, x_interp, output_ptr);
            }
            output_ptr += st.channels;
          }
        }
      }
    }
  }
  ctx.Output(0)->GetTensorShape()->SetDimSizes(out_shape);
  return KERNEL_STATUS_OK;
}
// compute the value of the specific pxiel when the num of channels is 3
template <bool NeedsXBounding, typename T>
void ResizeAreaCpuKernel::ComputePatchSumOf3Channels(float scale, const ResizeAreaSt &st,
                                                     const std::vector<const T *> &y_ptrs,
                                                     const std::vector<float> &y_scales,
                                                     const ResizeAreaCachedInterpolation &x_interp,
                                                     float *&output_patch_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))

  float sum_0 = 0;
  float sum_1 = 0;
  float sum_2 = 0;
  for (size_t i = 0; i < y_ptrs.size(); ++i) {
    const T *ptr = y_ptrs[i];
    float scale_x = x_interp.start_scale;
    int64_t offset = 3 * BOUND_IF_NEEDED(x_interp.start, st.in_width);
    float sum_y_0 = static_cast<float>(ptr[offset + 0]) * scale_x;
    float sum_y_1 = static_cast<float>(ptr[offset + 1]) * scale_x;
    float sum_y_2 = static_cast<float>(ptr[offset + kIndex2]) * scale_x;

    if (x_interp.start + 1 != x_interp.end) {
      for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
        int64_t offset = 3 * BOUND_IF_NEEDED(x, st.in_width);
        sum_y_0 += static_cast<float>(ptr[offset + 0]);
        sum_y_1 += static_cast<float>(ptr[offset + 1]);
        sum_y_2 += static_cast<float>(ptr[offset + kIndex2]);
      }
      scale_x = x_interp.end_minus_one_scale;
      offset = 3 * BOUND_IF_NEEDED(x_interp.end - 1, st.in_width);
      sum_y_0 += static_cast<float>(ptr[offset + 0]) * scale_x;
      sum_y_1 += static_cast<float>(ptr[offset + 1]) * scale_x;
      sum_y_2 += static_cast<float>(ptr[offset + kIndex2]) * scale_x;
    }
    sum_0 += sum_y_0 * y_scales[i];
    sum_1 += sum_y_1 * y_scales[i];
    sum_2 += sum_y_2 * y_scales[i];
  }

  output_patch_ptr[0] = sum_0 * scale;
  output_patch_ptr[1] = sum_1 * scale;
  output_patch_ptr[kIndex2] = sum_2 * scale;

#undef BOUND_IF_NEEDED
}

// compute the value of the specific pxiel when the num of channels is not 3
template <bool NeedsXBounding, typename T>
void ResizeAreaCpuKernel::ComputePatchSum(float scale, const ResizeAreaSt &st, const std::vector<const T *> &y_ptrs,
                                          const std::vector<float> &y_scales,
                                          const ResizeAreaCachedInterpolation &x_interp, float *&output_patch_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))

  const auto num_channels = st.channels;
  for (size_t c = 0; c < num_channels; ++c) {
    float sum = 0;
    for (size_t i = 0; i < y_ptrs.size(); ++i) {
      const T *ptr = y_ptrs[i];
      float scale_x = x_interp.start_scale;
      float sum_y = static_cast<float>(ptr[num_channels * BOUND_IF_NEEDED(x_interp.start, st.in_width) + c]) * scale_x;
      if (x_interp.start + 1 != x_interp.end) {
        for (size_t x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          sum_y += static_cast<float>(ptr[num_channels * BOUND_IF_NEEDED(x, st.in_width) + c]);
        }
        scale_x = x_interp.end_minus_one_scale;
        sum_y += static_cast<float>(ptr[num_channels * BOUND_IF_NEEDED(x_interp.end - 1, st.in_width) + c]) * scale_x;
      }
      sum += sum_y * y_scales[i];
    }
    output_patch_ptr[c] = sum * scale;
  }
#undef BOUND_IF_NEEDED
}

// check params
uint32_t ResizeAreaCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ResizeArea check input and output number failed.");

  Tensor *input_tensor1 = ctx.Input(0);
  Tensor *input_tensor2 = ctx.Input(1);
  auto outsize = reinterpret_cast<int32_t *>(input_tensor2->GetData());
  int32_t out_height = static_cast<int32_t>(outsize[0]);
  int32_t out_width = static_cast<int32_t>(outsize[1]);

  in_shape1 = input_tensor1->GetTensorShape()->GetDimSizes();
  in_shape2 = input_tensor2->GetTensorShape()->GetDimSizes();
  out_shape = std::vector<int64_t>{in_shape1[kIndex0], outsize[0], outsize[1], in_shape1[kIndex3]};

  KERNEL_CHECK_FALSE(in_shape1.size() == kIndex4, KERNEL_STATUS_PARAM_INVALID,
                     "Dim of input[0] must be 4,but got[%zu].", in_shape1.size());
  KERNEL_CHECK_FALSE(in_shape2.size() == 1, KERNEL_STATUS_PARAM_INVALID, "Dim of input[1] must be 1,but got[%zu].",
                     in_shape2.size());
  KERNEL_CHECK_FALSE(out_shape.size() == kIndex4, KERNEL_STATUS_PARAM_INVALID,
                     "Dim of output[0] must be 4,but got[%zu].", out_shape.size());
  KERNEL_CHECK_FALSE(out_height > 0 && out_width > 0, KERNEL_STATUS_PARAM_INVALID, "outsize must be positive.");

  AttrValue *attr_align_corners = ctx.GetAttr("align_corners");
  align_corners = (attr_align_corners == nullptr) ? false : (attr_align_corners->GetBool());
  dtype_ = input_tensor1->GetDataType();
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kResizeArea, ResizeAreaCpuKernel);
}  // namespace aicpu
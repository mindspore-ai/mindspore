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

#include "resize_bicubic.h"

#include <securec.h>

#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
const int64_t kTableSize = (1 << 10);
const int kCalNum4 = 4;
const char *kResizeBicubic = "ResizeBicubic";
std::vector<int64_t> out_shape_;
std::vector<int64_t> in_shape_;
bool align_corners_ = false;
bool half_pixel_centers_ = false;
bool format_nchw_ = true;
int n_idx_{0};
int c_idx_{1};
int h_idx_{2};
int w_idx_{3};
}  // namespace

namespace aicpu {
float Scaling_(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

struct ResizerState {
  void CalculateSize() {
    batch_size = in_shape_[n_idx_];
    channels = in_shape_[c_idx_];
    in_height = in_shape_[h_idx_];
    in_width = in_shape_[w_idx_];

    out_height = out_shape_[h_idx_];
    out_width = out_shape_[w_idx_];

    out_hw_size = out_height * out_width;
    in_hw_size = in_height * in_width;
    bchw_size = in_hw_size * channels * batch_size;
    height_scale = Scaling_(in_height, out_height, align_corners_);
    width_scale = Scaling_(in_width, out_width, align_corners_);
  }
  int64_t batch_size;
  int64_t out_height;
  int64_t out_width;
  int64_t in_height;
  int64_t in_width;
  int64_t channels;
  float height_scale;
  float width_scale;
  int64_t out_hw_size;
  int64_t in_hw_size;
  int64_t bchw_size;
};

struct WeightsAndIndices {
  float weight_0;
  float weight_1;
  float weight_2;
  float weight_3;
  int64_t index_0;
  int64_t index_1;
  int64_t index_2;
  int64_t index_3;

  int advance;  // advance value.
};

uint32_t ResizeBicubicCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  Tensor *input_size_tensor = ctx.Input(1);
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ResizeBicubic check params failed.");

  auto format = input_tensor->GetTensorShape()->GetFormat();
  format_nchw_ = (format == Format::FORMAT_NCHW);
  if (format_nchw_) {
    c_idx_ = 1;
    h_idx_ = 2;
    w_idx_ = 3;
  } else {
    h_idx_ = 1;
    w_idx_ = 2;
    c_idx_ = 3;
  }

  in_shape_ = input_tensor->GetTensorShape()->GetDimSizes();
  auto size_tensor = input_size_tensor->GetTensorShape()->GetDimSizes();

  KERNEL_CHECK_FALSE((in_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID, "Dim of input[0] must be 4, but got[%zu].",
                     in_shape_.size());
  KERNEL_CHECK_FALSE((size_tensor.size() == 1), KERNEL_STATUS_PARAM_INVALID, "Dim of input[1] must be 1, but got[%zu].",
                     size_tensor.size());

  auto size_type = input_size_tensor->GetDataType();
  std::vector<int64_t> size_value{};
  if (size_type == DT_INT32) {
    auto input_size = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
    size_value.push_back(static_cast<int64_t>(input_size[0]));
    size_value.push_back(static_cast<int64_t>(input_size[1]));
  } else if (size_type == DT_INT64) {
    auto input_size = reinterpret_cast<int64_t *>(ctx.Input(1)->GetData());
    size_value.push_back(input_size[0]);
    size_value.push_back(input_size[1]);
  } else {
    KERNEL_CHECK_FALSE(false, KERNEL_STATUS_PARAM_INVALID,
                       "For primitive[ ResizeBicubic ], the input argument[size] must be a Tensor[Int64] or "
                       "Tensor[Int32] type, but got[%zu].",
                       size_type);
  }

  std::vector<int64_t> out_dims{in_shape_};
  out_dims[h_idx_] = size_value[0];
  out_dims[w_idx_] = size_value[1];
  ctx.Output(0)->GetTensorShape()->SetDimSizes(out_dims);

  out_shape_ = output_tensor->GetTensorShape()->GetDimSizes();
  int64_t out_height = out_shape_[h_idx_];
  int64_t out_width = out_shape_[w_idx_];

  KERNEL_CHECK_FALSE((out_shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID, "Dim of output[0] must be 4, but got[%zu].",
                     out_shape_.size());
  KERNEL_CHECK_FALSE((out_height > 0 && out_width > 0), KERNEL_STATUS_PARAM_INVALID,
                     "output dimensions must be positive.");

  AttrValue *pattr_align_corners = ctx.GetAttr("align_corners");
  if (pattr_align_corners == nullptr) {
    align_corners_ = false;
  } else {
    align_corners_ = pattr_align_corners->GetBool();
  }

  AttrValue *pattr_half_pixel_centers = ctx.GetAttr("half_pixel_centers");
  if (pattr_half_pixel_centers == nullptr) {
    half_pixel_centers_ = false;
  } else {
    half_pixel_centers_ = pattr_half_pixel_centers->GetBool();
  }

  dtype_ = input_tensor->GetDataType();

  return KERNEL_STATUS_OK;
}

struct HalfPixelScaler {
  HalfPixelScaler(){};
  inline float operator()(const int64_t x, const float scale) const {
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};

struct LegacyScaler {
  LegacyScaler(){};
  inline float operator()(const int64_t x, const float scale) const { return static_cast<float>(x) * scale; }
};

class CachedInterpolationCalculator {
 public:
  CachedInterpolationCalculator() : indexes_{-1, -1, -1, -1} {}
  inline int Advance(const int64_t x_0, const int64_t x_1, const int64_t x_2, const int64_t x_3) {
    const std::array<int64_t, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    int cached_values_hand = 0;
    int new_indices_hand = 0;
    while (cached_values_hand < 4) {
      if (indexes_[cached_values_hand] == new_x_indices[new_indices_hand]) {
        if (new_indices_hand < cached_values_hand) {
          indexes_[new_indices_hand] = indexes_[cached_values_hand];
        }
        cached_values_hand++;
        new_indices_hand++;
      } else {
        cached_values_hand++;
      }
    }
    for (int i = new_indices_hand; i < 4; ++i) {
      indexes_[i] = new_x_indices[i];
    }
    return new_indices_hand;
  }

 private:
  int64_t indexes_[4];
};

inline int64_t Bound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }

const float *InitCoeffsTable(const double a) {
  float *coeffs_table = new float[(kTableSize + 1) * 2];
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  }

  return coeffs_table;
}

const float *GetCoeffsTable(const bool use_keys_cubic) {
  if (use_keys_cubic) {
    static const float *coeffs_table = InitCoeffsTable(-0.5f);
    return coeffs_table;
  } else {
    static const float *coeffs_table = InitCoeffsTable(-0.75f);
    return coeffs_table;
  }
}

template <typename Scaler, bool use_keys_cubic>
inline void GetWeightsAndIndices(const float scale, const int64_t out_loc, const int64_t limit,
                                 WeightsAndIndices *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float *coeffs_table = GetCoeffsTable(use_keys_cubic);
  const float calnum = 1000.0f;
  if (use_keys_cubic) {
    out->index_0 = Bound(in_loc - 1, limit);
    out->weight_0 = (out->index_0 == in_loc - 1 ? coeffs_table[offset * 2 + 1] : 0.0f);
    out->index_1 = Bound(in_loc, limit);
    out->weight_1 = (out->index_1 == in_loc ? coeffs_table[offset * 2] : 0.0f);
    out->index_2 = Bound(in_loc + 1, limit);
    out->weight_2 = (out->index_2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * 2] : 0.0f);
    out->index_3 = Bound(in_loc + 2, limit);
    out->weight_3 = (out->index_3 == in_loc + 2 ? coeffs_table[(kTableSize - offset) * 2 + 1] : 0.0f);

    const float weight_sum = out->weight_0 + out->weight_1 + out->weight_2 + out->weight_3;
    if (std::abs(weight_sum) >= calnum * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum = 1.0f / weight_sum;
      out->weight_0 *= one_over_weight_sum;
      out->weight_1 *= one_over_weight_sum;
      out->weight_2 *= one_over_weight_sum;
      out->weight_3 *= one_over_weight_sum;
    }
  } else {
    out->weight_0 = coeffs_table[offset * 2 + 1];
    out->weight_1 = coeffs_table[offset * 2];
    out->weight_2 = coeffs_table[(kTableSize - offset) * 2];
    out->weight_3 = coeffs_table[(kTableSize - offset) * 2 + 1];
    out->index_0 = Bound(in_loc - 1, limit);
    out->index_1 = Bound(in_loc, limit);
    out->index_2 = Bound(in_loc + 1, limit);
    out->index_3 = Bound(in_loc + 2, limit);
  }
}

static void ComputeXWeightsAndIndices(const ResizerState &resizer_state, const bool half_pixel_centers,
                                      std::vector<WeightsAndIndices> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers) {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<HalfPixelScaler, true>(resizer_state.width_scale, x, resizer_state.in_width, &(*x_wais)[x]);
      auto &x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  } else {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<LegacyScaler, false>(resizer_state.width_scale, x, resizer_state.in_width, &(*x_wais)[x]);
      auto &x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  }
}

template <typename T>
inline float Interpolate1D(const float weight_0, const float weight_1, const float weight_2, const float weight_3,
                           const T value_0, const T value_1, const T value_2, const T value_3) {
  return static_cast<float>(value_0) * weight_0 + static_cast<float>(value_1) * weight_1 +
         static_cast<float>(value_2) * weight_2 + static_cast<float>(value_3) * weight_3;
}

template <typename T>
static float ComputeYInterpolation(int which, const WeightsAndIndices &y_wai, const T *y_ptr_0, const T *y_ptr_1,
                                   const T *y_ptr_2, const T *y_ptr_3, const WeightsAndIndices &x_wai) {
  int x_index;
  switch (which) {
    case 0:
      x_index = x_wai.index_0;
      break;
    case 1:
      x_index = x_wai.index_1;
      break;
    case 2:
      x_index = x_wai.index_2;
      break;
    default:
      x_index = x_wai.index_3;
      break;
  }
  return Interpolate1D<T>(y_wai.weight_0, y_wai.weight_1, y_wai.weight_2, y_wai.weight_3, y_ptr_0[x_index],
                          y_ptr_1[x_index], y_ptr_2[x_index], y_ptr_3[x_index]);
}

template <typename T>
static float ComputeYInterpolation(int which, const int64_t &channel_num, const int64_t &num_channels,
                                   const WeightsAndIndices &y_wai, const T *y_ptr_0, const T *y_ptr_1, const T *y_ptr_2,
                                   const T *y_ptr_3, const WeightsAndIndices &x_wai) {
  int x_index;
  switch (which) {
    case 0:
      x_index = x_wai.index_0;
      break;
    case 1:
      x_index = x_wai.index_1;
      break;
    case 2:
      x_index = x_wai.index_2;
      break;
    default:
      x_index = x_wai.index_3;
      break;
  }
  const int64_t pt_index = x_index * num_channels + channel_num;
  return Interpolate1D<T>(y_wai.weight_0, y_wai.weight_1, y_wai.weight_2, y_wai.weight_3, y_ptr_0[pt_index],
                          y_ptr_1[pt_index], y_ptr_2[pt_index], y_ptr_3[pt_index]);
}

static float ComputeOneD(float values_[4], const float xw_0, const float xw_1, const float xw_2, const float xw_3) {
  return Interpolate1D<float>(xw_0, xw_1, xw_2, xw_3, values_[0], values_[1], values_[2], values_[3]);
}

template <typename T1, typename T2>
inline void interpolate_with_caching(const T1 *input_data, const ResizerState &resizer_state,
                                     const bool half_pixel_centers, T2 *output_data, const bool reload) {
  std::vector<WeightsAndIndices> x_wais(resizer_state.out_width);
  ComputeXWeightsAndIndices(resizer_state, half_pixel_centers, &x_wais);
  const T1 *input_b_ptr = input_data;
  T2 *output_y_ptr = output_data;
  std::vector<float> cached_value(4, 0);
  const int64_t in_row_width = resizer_state.in_width * resizer_state.in_height;  // hw
  for (int64_t b = 0; b < resizer_state.batch_size; ++b) {
    for (int64_t c = 0; c < resizer_state.channels; ++c, input_b_ptr += in_row_width) {
      for (int64_t y = 0; y < resizer_state.out_height; ++y, output_y_ptr += resizer_state.out_width) {
        WeightsAndIndices y_wai;
        if (half_pixel_centers) {
          GetWeightsAndIndices<HalfPixelScaler, true>(resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
        } else {
          GetWeightsAndIndices<LegacyScaler, false>(resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
        }
        // Make pointers represent offsets of data in input_b_ptr.
        const T1 *y_ptr_0 = input_b_ptr + y_wai.index_0 * resizer_state.in_width;
        const T1 *y_ptr_1 = input_b_ptr + y_wai.index_1 * resizer_state.in_width;
        const T1 *y_ptr_2 = input_b_ptr + y_wai.index_2 * resizer_state.in_width;
        const T1 *y_ptr_3 = input_b_ptr + y_wai.index_3 * resizer_state.in_width;
        for (int64_t x = 0; x < resizer_state.out_width; ++x) {
          const WeightsAndIndices &x_wai = x_wais[x];
          // Shift values in cached_value to fill first 'advance' values.
          switch (x_wai.advance) {
            case 3:
              cached_value[0] = cached_value[1];
              cached_value[1] = cached_value[2];
              cached_value[2] = cached_value[3];
              break;
            case 2:
              cached_value[0] = cached_value[2];
              cached_value[1] = cached_value[3];
              break;
            case 1:
              cached_value[0] = cached_value[3];
              break;
            default:
              break;
          }
          // Set the remaining '4-advance' values by computing.
          for (int i = x_wai.advance; i < kCalNum4; ++i) {
            cached_value[i] = ComputeYInterpolation(i, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
          }
          output_y_ptr[x] = static_cast<T2>(
            ComputeOneD(cached_value.data(), x_wai.weight_0, x_wai.weight_1, x_wai.weight_2, x_wai.weight_3));
        }
      }
    }
  }
}

template <typename T1, typename T2>
inline void interpolate_with_caching(const T1 *input_data, const ResizerState &resizer_state,
                                     const bool half_pixel_centers, T2 *output_data) {
  std::vector<WeightsAndIndices> x_wais(resizer_state.out_width);
  ComputeXWeightsAndIndices(resizer_state, half_pixel_centers, &x_wais);
  const T1 *input_b_ptr = input_data;
  T2 *output_y_ptr = output_data;
  const auto num_channels = resizer_state.channels;
  const int64_t in_row_width = resizer_state.in_width * num_channels;
  const int64_t in_batch_width = resizer_state.in_height * in_row_width;
  std::vector<float> cached_value(4 * num_channels, 0);
  for (int64_t b = 0; b < resizer_state.batch_size; ++b, input_b_ptr += in_batch_width) {
    for (int64_t y = 0; y < resizer_state.out_height; ++y, output_y_ptr += resizer_state.out_width * num_channels) {
      WeightsAndIndices y_wai;
      if (half_pixel_centers) {
        GetWeightsAndIndices<HalfPixelScaler, true>(resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
      } else {
        GetWeightsAndIndices<LegacyScaler, false>(resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
      }
      // Make pointers represent offsets of data in input_b_ptr.
      const T1 *y_ptr_0 = input_b_ptr + y_wai.index_0 * in_row_width;
      const T1 *y_ptr_1 = input_b_ptr + y_wai.index_1 * in_row_width;
      const T1 *y_ptr_2 = input_b_ptr + y_wai.index_2 * in_row_width;
      const T1 *y_ptr_3 = input_b_ptr + y_wai.index_3 * in_row_width;
      for (int64_t x = 0; x < resizer_state.out_width; ++x) {
        const WeightsAndIndices &x_wai = x_wais[x];
        // Shift values in cached_value to fill first 'advance' values.
        switch (x_wai.advance) {
          case 3:
            for (int64_t c = 0; c < num_channels; ++c) {
              cached_value[4 * c + 0] = cached_value[4 * c + 1];
              cached_value[4 * c + 1] = cached_value[4 * c + 2];
              cached_value[4 * c + 2] = cached_value[4 * c + 3];
            }
            break;
          case 2:
            for (int64_t c = 0; c < num_channels; ++c) {
              cached_value[4 * c + 0] = cached_value[4 * c + 2];
              cached_value[4 * c + 1] = cached_value[4 * c + 3];
            }
            break;
          case 1:
            for (int64_t c = 0; c < num_channels; ++c) {
              cached_value[4 * c + 0] = cached_value[4 * c + 3];
            }
            break;
          default:
            break;
        }
        // Set the remaining '4-advance' values by computing.
        for (int64_t c = 0; c < num_channels; ++c) {
          for (int i = x_wai.advance; i < 4; ++i) {
            cached_value[4 * c + i] =
              ComputeYInterpolation(i, c, num_channels, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
          }
          output_y_ptr[x * num_channels + c] = static_cast<T2>(
            ComputeOneD(&cached_value[4 * c], x_wai.weight_0, x_wai.weight_1, x_wai.weight_2, x_wai.weight_3));
        }
      }
    }
  }
}

template <typename T1, typename T2>
uint32_t DoCompute(CpuKernelContext &ctx) {
  auto input_addr = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto output_addr = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  ResizerState sta;
  sta.CalculateSize();

  if (sta.out_height == sta.in_height && sta.out_width == sta.in_width) {
    for (int64_t i = 0; i < sta.bchw_size; ++i) {
      output_addr[i] = static_cast<T2>(input_addr[i]);
    }
    return KERNEL_STATUS_OK;
  }

  if (format_nchw_) {
    interpolate_with_caching<T1, T2>(input_addr, sta, half_pixel_centers_, output_addr, format_nchw_);
  } else {
    interpolate_with_caching<T1, T2>(input_addr, sta, half_pixel_centers_, output_addr);
  }
  return KERNEL_STATUS_OK;
}

uint32_t ResizeBicubicCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");

  if (dtype_ == DT_FLOAT16) {
    res = DoCompute<Eigen::half, Eigen::half>(ctx);
  } else if (dtype_ == DT_FLOAT) {
    res = DoCompute<float, float>(ctx);
  } else if (dtype_ == DT_INT8) {
    res = DoCompute<int8_t, float>(ctx);
  } else if (dtype_ == DT_UINT8) {
    res = DoCompute<uint8_t, float>(ctx);
  } else if (dtype_ == DT_INT16) {
    res = DoCompute<int16_t, float>(ctx);
  } else if (dtype_ == DT_UINT16) {
    res = DoCompute<uint16_t, float>(ctx);
  } else if (dtype_ == DT_INT32) {
    res = DoCompute<int32_t, float>(ctx);
  } else if (dtype_ == DT_INT64) {
    res = DoCompute<int64_t, float>(ctx);
  } else if (dtype_ == DT_DOUBLE) {
    res = DoCompute<double, double>(ctx);
  } else {
    KERNEL_LOG_ERROR("ResizeBicubic doesn't support input tensor types: [%s]", DTypeStr(dtype_).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "ResizeBicubic Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kResizeBicubic, ResizeBicubicCpuKernel);
}  // namespace aicpu

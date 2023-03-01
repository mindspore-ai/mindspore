#include "resize_bicubic_grad.h"

#include <securec.h>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "utils/sparse_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
static const int64_t kTableSize = (1 << 10);
const int64_t kParallelDataNum = 1024 * 256;
const char *kResizeBicubicGrad = "ResizeBicubicGrad";
std::vector<int64_t> size_;
std::vector<int64_t> shape_;
float height_scale_ = 0;
float width_scale_ = 0;
bool align_corners_ = false;
bool half_pixel_centers_ = false;

}  // namespace

namespace aicpu {

DataType dtype0_ = DT_FLOAT;
DataType dtype1_ = DT_FLOAT;
DataType dtype2_ = DT_FLOAT;

float Scaling64_(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}
struct ResizerGradState {
  void CalculateSize(CpuKernelContext &ctx) {
    Tensor *input0_tensor = ctx.Input(0);
    Tensor *input1_tensor = ctx.Input(1);
    shape_ = input0_tensor->GetTensorShape()->GetDimSizes();
    size_ = input1_tensor->GetTensorShape()->GetDimSizes();

    batch_size = shape_[0];
    channels = shape_[3];
    resized_height = shape_[1];
    resized_width = shape_[2];

    original_height = size_[1];
    original_width = size_[2];

    height_scale = Scaling64_(original_height, resized_height, align_corners_);
    width_scale = Scaling64_(original_width, resized_width, align_corners_);
  }
  int64_t Calindex(const int64_t x1, const int64_t x2, const int64_t x3, const int64_t x4, const bool flag_) {
    if (!flag_) {
      return static_cast<int64_t>(x1 * original_height * original_width * channels) +
             static_cast<int64_t>(x2 * original_width * channels) + static_cast<int64_t>(x3 * channels) +
             static_cast<int64_t>(x4);
    } else {
      return static_cast<int64_t>(x1 * resized_height * resized_width * channels) +
             static_cast<int64_t>(x2 * resized_width * channels) + static_cast<int64_t>(x3 * channels) +
             static_cast<int64_t>(x4);
    }
  }
  int64_t batch_size;
  int64_t channels;

  int64_t original_height;
  int64_t original_width;

  int64_t resized_height;
  int64_t resized_width;

  float height_scale;
  float width_scale;
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

  int advance;
};

struct HalfPixelScalerGrad {
  HalfPixelScalerGrad(){};
  inline float operator()(const size_t x, const float scale) const {
    return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
  }
};
struct LegacyScalerGrad {
  LegacyScalerGrad(){};
  inline float operator()(const size_t x, const float scale) const { return static_cast<float>(x) * scale; }
};

class CachedInterpolationCalculator {
 public:
  CachedInterpolationCalculator() : indexes_{-1, -1, -1, -1} {}
  inline int Advance(const int64_t x_0, const int64_t x_1, const int64_t x_2, const int64_t x_3) {
    const std::array<int64_t, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    int64_t cached_values_hand = 0;
    int64_t new_indices_hand = 0;
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
    switch (new_indices_hand) {
      case 0:
        indexes_[0] = x_0;
      case 1:
        indexes_[1] = x_1;
      case 2:
        indexes_[2] = x_2;
      case 3:
        indexes_[3] = x_3;
        break;
    }
    return new_indices_hand;
  }

 private:
  int64_t indexes_[4];
};

const float *InitCoeffsTable_(const double a) {
  float *coeffs_table = new float[(kTableSize + 1) * 2];
  for (int64_t i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  }

  return coeffs_table;
}

const float *GetCoeffsTable_(const bool use_keys_cubic) {
  if (use_keys_cubic) {
    static const float *coeffs_table = InitCoeffsTable_(-0.5f);
    return coeffs_table;
  } else {
    static const float *coeffs_table = InitCoeffsTable_(-0.75f);
    return coeffs_table;
  }
}

inline int64_t Bound(int64_t val, int64_t limit) { return std::min(limit - 1, std::max(int64_t{0}, val)); }

template <typename Scaler, bool use_keys_cubic>
inline void GetWeightsAndIndicesGrad(const float scale, const size_t out_loc, const size_t limit,
                                     WeightsAndIndices *out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float *coeffs_table = GetCoeffsTable_(use_keys_cubic);
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
    if (std::abs(weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
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

uint32_t ResizeBicubicGradCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(0);
  Tensor *input1_tensor = ctx.Input(1);
  Tensor *output_tensor = ctx.Output(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ResizeBicubicGrad check params failed.");

  shape_ = input0_tensor->GetTensorShape()->GetDimSizes();
  size_ = input1_tensor->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((shape_.size() == 4), KERNEL_STATUS_PARAM_INVALID, "Dim of input[0] must be 4, but got[%zu].",
                     shape_.size());
  KERNEL_CHECK_FALSE((size_.size() == 4), KERNEL_STATUS_PARAM_INVALID, "Dim of input[1] must be 4, but got[%zu].",
                     size_.size());
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
  dtype0_ = input0_tensor->GetDataType();
  dtype1_ = input1_tensor->GetDataType();
  dtype2_ = output_tensor->GetDataType();

  KERNEL_CHECK_FALSE((dtype0_ == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "ResizeBicubicGrad op doesn't support input[0] tensor types: [%s]", DTypeStr(dtype0_).c_str());

  KERNEL_CHECK_FALSE((dtype1_ == DT_FLOAT || dtype1_ == DT_DOUBLE), KERNEL_STATUS_PARAM_INVALID,
                     "ResizeBicubicGrad op doesn't support input[1] tensor types: [%s]", DTypeStr(dtype1_).c_str());

  KERNEL_CHECK_FALSE((dtype1_ == dtype2_), KERNEL_STATUS_PARAM_INVALID,
                     "The type of input[1] and output must be the same");

  int64_t in_height = shape_[1];
  int64_t in_width = shape_[2];
  int64_t out_height = size_[1];
  int64_t out_width = size_[2];
  height_scale_ = Scaling64_(out_height, in_height, align_corners_);
  width_scale_ = Scaling64_(out_width, in_width, align_corners_);
  return KERNEL_STATUS_OK;
}

static void ComputeGradientXWeightsAndIndices(const ResizerGradState &resizer_state, const bool half_pixel_centers,
                                              std::vector<WeightsAndIndices> *x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers) {
    for (int64_t x = 0; x < resizer_state.resized_width; ++x) {
      GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(resizer_state.width_scale, x, resizer_state.original_width,
                                                          &(*x_wais)[x]);
      auto &x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }

  } else {
    for (int64_t x = 0; x < resizer_state.resized_width; ++x) {
      GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(resizer_state.width_scale, x, resizer_state.original_width,
                                                        &(*x_wais)[x]);
      auto &x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2, x_wai.index_3);
    }
  }
}

template <typename T>
inline void ResizeBicubicGrad(const float *input_grad, ResizerGradState &resizer_state, const bool half_pixel_centers,
                              T *output_grad, CpuKernelContext &ctx) {
  const float height_scale = resizer_state.height_scale;
  const int64_t original_height = resizer_state.original_height;
  const int64_t channels = resizer_state.channels;
  const int64_t resized_width = resizer_state.resized_width;
  const int64_t resized_height = resizer_state.resized_height;

  std::vector<WeightsAndIndices> x_wais(resizer_state.resized_width);
  ComputeGradientXWeightsAndIndices(resizer_state, half_pixel_centers, &x_wais);
  const bool flag = true;
  bool utils_flag = false;
  if (resizer_state.original_width * original_height * channels * resizer_state.batch_size >= kParallelDataNum) {
    utils_flag = true;
  }
  if (utils_flag) {
    for (int64_t b = 0; b < resizer_state.batch_size; ++b) {
      uint32_t min_core_num = 1;
      int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
      if (max_core_num > resized_height) {
        max_core_num = resized_height;
      }
      auto shard_resize_bicubic_grad = [&](int64_t start, int64_t end) {
        for (int64_t y = start; y < end; ++y) {
          WeightsAndIndices y_wai;
          if (half_pixel_centers) {
            GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(height_scale, y, original_height, &y_wai);
          } else {
            GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(height_scale, y, original_height, &y_wai);
          }
          for (int64_t x = 0; x < resized_width; ++x) {
            const WeightsAndIndices &x_wai = x_wais[x];
            for (int64_t c = 0; c < channels; ++c) {
              T curr_input_grad = input_grad[resizer_state.Calindex(b, y, x, c, flag)];
              // row 0 of 0, 1, 2, 3
              output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_0, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_0 * x_wai.weight_0);
              output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_1, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_0 * x_wai.weight_1);
              output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_2, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_0 * x_wai.weight_2);
              output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_3, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_0 * x_wai.weight_3);

              // row 1 of 0, 1, 2, 3
              output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_0, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_1 * x_wai.weight_0);
              output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_1, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_1 * x_wai.weight_1);
              output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_2, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_1 * x_wai.weight_2);
              output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_3, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_1 * x_wai.weight_3);
              // row 2 of 0, 1, 2, 3
              output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_0, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_2 * x_wai.weight_0);
              output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_1, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_2 * x_wai.weight_1);
              output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_2, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_2 * x_wai.weight_2);
              output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_3, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_2 * x_wai.weight_3);
              // row 3 of 0, 1, 2, 3
              output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_0, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_3 * x_wai.weight_0);
              output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_1, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_3 * x_wai.weight_1);
              output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_2, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_3 * x_wai.weight_2);
              output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_3, c, !flag)] +=
                T(curr_input_grad * y_wai.weight_3 * x_wai.weight_3);
            }
          }
        }
      };
      CpuKernelUtils::ParallelFor(ctx, resized_height, resized_height / max_core_num, shard_resize_bicubic_grad);
    }
  } else {
    for (int64_t b = 0; b < resizer_state.batch_size; ++b) {
      for (int64_t y = 0; y < resized_height; ++y) {
        WeightsAndIndices y_wai;
        if (half_pixel_centers) {
          GetWeightsAndIndicesGrad<HalfPixelScalerGrad, true>(height_scale, y, original_height, &y_wai);
        } else {
          GetWeightsAndIndicesGrad<LegacyScalerGrad, false>(height_scale, y, original_height, &y_wai);
        }
        for (int64_t x = 0; x < resized_width; ++x) {
          const WeightsAndIndices &x_wai = x_wais[x];
          for (int64_t c = 0; c < channels; ++c) {
            T curr_input_grad = input_grad[resizer_state.Calindex(b, y, x, c, flag)];
            // row 0 of 0, 1, 2, 3
            output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_0, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_0);
            output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_1, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_1);
            output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_2, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_2);
            output_grad[resizer_state.Calindex(b, y_wai.index_0, x_wai.index_3, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_3);

            // row 1 of 0, 1, 2, 3
            output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_0, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_0);
            output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_1, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_1);
            output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_2, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_2);
            output_grad[resizer_state.Calindex(b, y_wai.index_1, x_wai.index_3, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_3);
            // row 2 of 0, 1, 2, 3
            output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_0, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_0);
            output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_1, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_1);
            output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_2, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_2);
            output_grad[resizer_state.Calindex(b, y_wai.index_2, x_wai.index_3, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_3);
            // row 3 of 0, 1, 2, 3
            output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_0, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_0);
            output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_1, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_1);
            output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_2, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_2);
            output_grad[resizer_state.Calindex(b, y_wai.index_3, x_wai.index_3, c, !flag)] +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_3);
          }
        }
      }
    }
  }
}

template <typename T>
uint32_t ResizeBicubicGradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_addr = reinterpret_cast<float *>(ctx.Input(0)->GetData());
  auto output_addr = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  AttrValue *pattr_half_pixel_centers = ctx.GetAttr("half_pixel_centers");
  if (pattr_half_pixel_centers == nullptr) {
    half_pixel_centers_ = false;
  } else {
    half_pixel_centers_ = pattr_half_pixel_centers->GetBool();
  }
  ResizerGradState sta;
  sta.CalculateSize(ctx);

  auto ret = memset_s(output_addr, ctx.Output(0)->GetDataSize(), 0, ctx.Output(0)->GetDataSize());
  KERNEL_CHECK_FALSE((ret == EOK), ret, "Output buffer memset failed, ret: [%d].", ret);

  ResizeBicubicGrad(input0_addr, sta, half_pixel_centers_, output_addr, ctx);
  return KERNEL_STATUS_OK;
}

uint32_t ResizeBicubicGradCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "GetInputAndCheck failed.");

  if (dtype1_ == DT_DOUBLE) {
    res = DoCompute<double>(ctx);
  } else if (dtype1_ == DT_FLOAT) {
    res = DoCompute<float>(ctx);
  } else {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "ResizeBicubicGrad Compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kResizeBicubicGrad, ResizeBicubicGradCpuKernel);
}  // namespace aicpu

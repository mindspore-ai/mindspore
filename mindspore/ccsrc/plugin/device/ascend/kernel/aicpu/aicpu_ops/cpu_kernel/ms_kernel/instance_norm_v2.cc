/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "instance_norm_v2.h"

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "securec.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"
#include <vector>
#include <map>
#include <algorithm>

namespace {
const char *const kInstanceNormV2 = "InstanceNormV2";
constexpr float float_init_zero = 0.0;
constexpr float float_init_one = 1.0;
constexpr float momentum_min = 0.0;
constexpr float momentum_max = 1.0;
constexpr double double_init_zero = 0.0;
constexpr double double_init_one = 1.0;
constexpr int32_t int32_init_one = 1;
constexpr int64_t int64_init_one = 1;
constexpr auto kInstanceNormV2InputsNum = 5;
constexpr auto kInstanceNormV2OutputNum = 3;
// GRAIN_SIZE for Parallel
constexpr auto kGrainSize = 4 * 1024;
constexpr auto kDim3 = 3;
constexpr auto kDim4 = 4;
constexpr auto kDim5 = 5;
constexpr auto InstanceNormV2InXIndex = 0;
constexpr auto InstanceNormV2InGammaIndex = 1;
constexpr auto InstanceNormV2InBetaIndex = 2;
constexpr auto InstanceNormV2InMeanIndex = 3;
constexpr auto InstanceNormV2InVarianceIndex = 4;
constexpr auto InstanceNormV2OutYIndex = 0;
constexpr auto InstanceNormV2OutBatchMeanIndex = 1;
constexpr auto InstanceNormV2OutBatchVarianceIndex = 2;

template <typename T>
struct InvStd {
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(int32_init_one) / std::sqrt(var + epsilon);
    }
    return invstd;
  }
};
}  // namespace

namespace aicpu {
uint32_t InstanceNormV2CpuKernel::InstanceNormV2TypeCheck(const CpuKernelContext &ctx) {
  auto x_type = ctx.Input(InstanceNormV2InXIndex)->GetDataType();
  auto gamma_type = ctx.Input(InstanceNormV2InGammaIndex)->GetDataType();
  auto beta_type = ctx.Input(InstanceNormV2InBetaIndex)->GetDataType();
  auto mean_type = ctx.Input(InstanceNormV2InMeanIndex)->GetDataType();
  auto variance_type = ctx.Input(InstanceNormV2InVarianceIndex)->GetDataType();
  auto y_type = ctx.Output(InstanceNormV2OutYIndex)->GetDataType();
  auto batch_mean_type = ctx.Output(InstanceNormV2OutBatchMeanIndex)->GetDataType();
  auto batch_variance_type = ctx.Output(InstanceNormV2OutBatchVarianceIndex)->GetDataType();

  if (x_type != y_type) {
    KERNEL_LOG_ERROR(
      "For primitive[%s]'s input arguments x should have the same "
      "data type with output arguments y, but y type is [%s], x type is "
      "[%s].",
      kInstanceNormV2, DTypeStr(y_type).c_str(), DTypeStr(x_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const std::map<std::string, DataType> types = {{"gamma", gamma_type},
                                                 {"beta", beta_type},
                                                 {"mean", mean_type},
                                                 {"variance", variance_type},
                                                 {"batch_mean", batch_mean_type},
                                                 {"batch_variance", batch_variance_type}};
  return CheckTensorTypeSame(types, DT_FLOAT, kInstanceNormV2);
}

uint32_t InstanceNormV2CpuKernel::InstanceNormV2ShapeCheck(const CpuKernelContext &ctx) {
  auto x_shape_ptr = ctx.Input(InstanceNormV2InXIndex)->GetTensorShape();
  auto gamma_shape_ptr = ctx.Input(InstanceNormV2InGammaIndex)->GetTensorShape();
  auto beta_shape_ptr = ctx.Input(InstanceNormV2InBetaIndex)->GetTensorShape();
  auto mean_shape_ptr = ctx.Input(InstanceNormV2InMeanIndex)->GetTensorShape();
  auto variance_shape_ptr = ctx.Input(InstanceNormV2InVarianceIndex)->GetTensorShape();
  auto y_shape_ptr = ctx.Output(InstanceNormV2OutYIndex)->GetTensorShape();
  auto batch_mean_shape_ptr = ctx.Output(InstanceNormV2OutBatchMeanIndex)->GetTensorShape();
  auto batch_variance_shape_ptr = ctx.Output(InstanceNormV2OutBatchVarianceIndex)->GetTensorShape();

  auto y_shape = y_shape_ptr->GetDimSizes();
  auto x_shape = x_shape_ptr->GetDimSizes();
  auto res = CheckTensorShapeSame({{"x", x_shape_ptr}}, y_shape, kInstanceNormV2);
  if (res != KERNEL_STATUS_OK) {
    return res;
  };
  auto x_format = x_shape_ptr->GetFormat();
  std::vector<int64_t> check_shape;
  check_shape = y_shape;
  int64_t image_size = 0;
  if (x_shape.size() == kDim4) {
    if (x_format == FORMAT_NCHW || x_format == FORMAT_ND) {
      // consider (N, C, H, W) as (N*C, H, W, 1), similar to (N, H, W, C)
      check_shape[kFormatNCHWIndexH] = int64_init_one;
      check_shape[kFormatNCHWIndexW] = int64_init_one;
      image_size = x_shape[kFormatNCHWIndexH] * x_shape[kFormatNCHWIndexW];
      instance_num = x_shape[kFormatNCHWIndexN] * x_shape[kFormatNCHWIndexC];
      constexpr int64_t kNumberOne = 1;
      x_shape_4d_ = {x_shape[kFormatNCHWIndexN] * x_shape[kFormatNCHWIndexC], x_shape[kFormatNCHWIndexH],
                     x_shape[kFormatNCHWIndexW], kNumberOne};
      batch_channels_2d_ = {x_shape[kFormatNCHWIndexN] * x_shape[kFormatNCHWIndexC], kNumberOne};
    } else if (x_format == FORMAT_NHWC) {
      check_shape[kFormatNHWCIndexH] = int64_init_one;
      check_shape[kFormatNHWCIndexW] = int64_init_one;
      image_size = x_shape[kFormatNHWCIndexH] * x_shape[kFormatNHWCIndexW];
      instance_num = x_shape[kFormatNHWCIndexN] * x_shape[kFormatNHWCIndexC];
      x_shape_4d_ = x_shape;
      batch_channels_2d_ = {x_shape[kFormatNHWCIndexN], x_shape[kFormatNHWCIndexC]};
    }
  } else if (x_format == FORMAT_NC1HWC0 || x_shape.size() == kDim5) {
    // consider (N, C1, H, W, C0) as (N*C1, H, W, C0), similar to (N, H, W, C)
    check_shape[kFormatNC1HWC0IndexH] = int64_init_one;
    check_shape[kFormatNC1HWC0IndexW] = int64_init_one;
    image_size = x_shape[kFormatNC1HWC0IndexH] * x_shape[kFormatNC1HWC0IndexW];
    instance_num = x_shape[kFormatNC1HWC0IndexN] * x_shape[kFormatNC1HWC0IndexC1] * x_shape[kFormatNC1HWC0IndexC0];
    x_shape_4d_ = {x_shape[kFormatNC1HWC0IndexN] * x_shape[kFormatNC1HWC0IndexC1], x_shape[kFormatNC1HWC0IndexH],
                   x_shape[kFormatNC1HWC0IndexW], x_shape[kFormatNC1HWC0IndexC0]};
    batch_channels_2d_ = {x_shape[kFormatNC1HWC0IndexN] * x_shape[kFormatNC1HWC0IndexC1],
                          x_shape[kFormatNC1HWC0IndexC0]};
  } else {
    KERNEL_LOG_ERROR(
      "For primitive[%s]'s input arguments x only "
      "support NHWC, NCHW and NC1HWC0, but get data format [%s]",
      kInstanceNormV2, FormatToSerialString(x_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  constexpr int64_t image_min = 1;
  if (image_size <= image_min) {
    KERNEL_LOG_ERROR(
      "For primitive[%s], expected more than 1 value per instance, but get "
      "[%ld] value per instance.",
      kInstanceNormV2, image_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const std::map<std::string, TensorShapePtr> shapes = {{"gamma", gamma_shape_ptr},
                                                        {"beta", beta_shape_ptr},
                                                        {"mean", mean_shape_ptr},
                                                        {"variance", variance_shape_ptr},
                                                        {"batch_mean", batch_mean_shape_ptr},
                                                        {"batch_variance", batch_variance_shape_ptr}};
  return CheckTensorShapeSame(shapes, check_shape, kInstanceNormV2);
}

uint32_t InstanceNormV2CpuKernel::InstanceNormV2AttrCheck(const CpuKernelContext &ctx) {
  constexpr float epsilon_min = 0.0;
  constexpr float epsilon_max = 1.0;
  auto epsilon_ptr = ctx.GetAttr("epsilon");
  if (epsilon_ptr) {
    epsilon_ = epsilon_ptr->GetFloat();
  }
  if (epsilon_ < epsilon_min || epsilon_ >= epsilon_max) {
    KERNEL_LOG_ERROR(
      "For primitive[%s], attr epsilon value should be in [0, 1), but get "
      "[%f].",
      kInstanceNormV2, epsilon_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto momentum_ptr = ctx.GetAttr("momentum");
  if (momentum_ptr) {
    momentum_ = momentum_ptr->GetFloat();
  }
  if (momentum_ < epsilon_min || momentum_ > epsilon_max) {
    KERNEL_LOG_ERROR(
      "For primitive[%s], attr momentum value should be in [0, 1], but get "
      "[%f].",
      kInstanceNormV2, momentum_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto is_training_ptr = ctx.GetAttr("is_training");
  if (is_training_ptr) {
    is_training_ = is_training_ptr->GetBool();
  }
  return KERNEL_STATUS_OK;
}

uint32_t InstanceNormV2CpuKernel::InstanceNormV2ParamCheck(const CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(InstanceNormV2TypeCheck(ctx), "InstanceNormV2 check type failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2ShapeCheck(ctx), "InstanceNormV2 check shape failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2AttrCheck(ctx), "InstanceNormV2 check attr failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InstanceNormV2CpuKernel::CollectStatsKernel(const CpuKernelContext &ctx, float *_mean_, float *_var_sum) {
  const int64_t batch = x_shape_4d_[kFormatNHWCIndexN];
  const int64_t channel = x_shape_4d_[kFormatNHWCIndexC];
  const int64_t image_size = x_shape_4d_[kFormatNHWCIndexH] * x_shape_4d_[kFormatNHWCIndexW];
  KERNEL_CHECK_FALSE((channel != 0), KERNEL_STATUS_PARAM_INVALID, "Channel can not be zero!");
  KERNEL_CHECK_FALSE((image_size != 0), KERNEL_STATUS_PARAM_INVALID, "image_size can not be zero!");
  std::vector<int64_t> shape_3d = {batch, image_size, channel};
  auto x_3d = EigenTensor(shape_3d, ctx.Input(InstanceNormV2InXIndex)->GetData()).tensor<T, kDim3>();
  auto loop_batch = [&](int64_t begin, int64_t end) {
    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
      for (int64_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
        // compute mean per input
        double sum = double_init_zero;
        for (int64_t idx = 0; idx < image_size; ++idx) {
          sum += static_cast<double>(x_3d(batch_idx, idx, channel_idx));
        }
        double cur_mean = sum / static_cast<double>(image_size);
        _mean_[batch_idx * channel + channel_idx] = static_cast<float>(cur_mean);
        // compute variance per input
        double cur_var_sum = double_init_zero;
        for (int64_t idx = 0; idx < image_size; ++idx) {
          double cur_piexl = static_cast<double>(x_3d(batch_idx, idx, channel_idx));
          cur_var_sum += (cur_piexl - cur_mean) * (cur_piexl - cur_mean);
        }
        _var_sum[batch_idx * channel + channel_idx] = static_cast<float>(cur_var_sum);
      }
    }
  };
  int64_t block_size = std::max(int64_init_one, (kGrainSize / (channel * image_size)));

  return CpuKernelUtils::ParallelFor(ctx, batch, block_size, loop_batch);
}

template <typename T, template <typename S> class VarTransform>
uint32_t InstanceNormV2CpuKernel::UpdateStatsTemplate(const CpuKernelContext &ctx) {
  std::vector<float> _var_sum(instance_num, float_init_zero);
  std::vector<float> _mean_(instance_num, float_init_zero);
  (void)CollectStatsKernel<T>(ctx, _mean_.data(), _var_sum.data());
  const int64_t image_size = x_shape_4d_[kFormatNHWCIndexH] * x_shape_4d_[kFormatNHWCIndexW];
  KERNEL_CHECK_FALSE((image_size != 0), KERNEL_STATUS_PARAM_INVALID, "image_size can not be zero!");
  std::vector<int64_t> batch_channels_1d_ = {batch_channels_2d_.front() * batch_channels_2d_.back()};
  auto running_mean_vec = EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InMeanIndex)->GetData()).vec<float>();
  auto running_var_vec =
    EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InVarianceIndex)->GetData()).vec<float>();
  auto save_mean_vec =
    EigenTensor(batch_channels_1d_, ctx.Output(InstanceNormV2OutBatchMeanIndex)->GetData()).vec<float>();
  auto save_var_vec =
    EigenTensor(batch_channels_1d_, ctx.Output(InstanceNormV2OutBatchVarianceIndex)->GetData()).vec<float>();
  auto loop_momentum = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      save_mean_vec(idx) = _mean_[idx];
      save_var_vec(idx) =
        VarTransform<double>{}(static_cast<double>(_var_sum[idx]) / static_cast<double>(image_size), epsilon_);
      running_mean_vec(idx) =
        static_cast<float>(momentum_ * static_cast<double>(_mean_[idx]) +
                           (double_init_one - momentum_) * static_cast<double>(running_mean_vec(idx)));
      double unbiased_var = double_init_zero;
      if (image_size - int64_init_one == 0) {
        unbiased_var = static_cast<double>(_var_sum[idx]);
      } else {
        unbiased_var = static_cast<double>(_var_sum[idx]) / static_cast<double>(image_size - int64_init_one);
      }
      running_var_vec(idx) = static_cast<float>(momentum_ * unbiased_var + (double_init_one - momentum_) *
                                                                             static_cast<double>(running_var_vec(idx)));
    }
  };
  return CpuKernelUtils::ParallelFor(ctx, instance_num, kGrainSize, loop_momentum);
}

uint32_t InstanceNormV2CpuKernel::CollectLinearAndConstant(
  const CpuKernelContext &ctx, const typename TTypes<float>::Vec &gamma, const typename TTypes<float>::Vec &beta,
  const typename TTypes<float>::Vec &running_mean, const typename TTypes<float>::Vec &running_var,
  const typename TTypes<float>::Vec &save_mean, const typename TTypes<float>::Vec &save_invstd, float *_alpha_,
  float *_beta_) {
  auto loop_instance = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      float mean = float_init_zero, invstd = float_init_zero;
      if (is_training_) {
        mean = save_mean(idx);
        invstd = save_invstd(idx);
      } else {
        mean = running_mean(idx);
        float _std_ = std::sqrt(running_var(idx) + static_cast<float>(epsilon_));
        KERNEL_CHECK_FALSE((_std_ != 0), KERNEL_STATUS_PARAM_INVALID, "_std_ can not be zero!");
        invstd = float_init_one / _std_;
      }
      _alpha_[idx] = invstd * gamma(idx);
      _beta_[idx] = beta(idx) - mean * _alpha_[idx];
    }
    return KERNEL_STATUS_OK;
  };
  return CpuKernelUtils::ParallelFor(ctx, instance_num, kGrainSize, loop_instance);
}

template <typename T>
uint32_t InstanceNormV2CpuKernel::TransformInput(const CpuKernelContext &ctx) {
  const int64_t batch = x_shape_4d_[kFormatNHWCIndexN];
  const int64_t channel = x_shape_4d_[kFormatNHWCIndexC];
  const int64_t image_size = x_shape_4d_[kFormatNHWCIndexH] * x_shape_4d_[kFormatNHWCIndexW];
  std::vector<float> _alpha_(instance_num, float_init_zero);
  std::vector<float> _beta_(instance_num, float_init_zero);
  std::vector<int64_t> batch_channels_1d_ = {batch_channels_2d_.front() * batch_channels_2d_.back()};
  auto gamma = EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InGammaIndex)->GetData()).vec<float>();
  auto beta = EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InBetaIndex)->GetData()).vec<float>();
  auto running_mean = EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InMeanIndex)->GetData()).vec<float>();
  auto running_var = EigenTensor(batch_channels_1d_, ctx.Input(InstanceNormV2InVarianceIndex)->GetData()).vec<float>();
  auto save_mean = EigenTensor(batch_channels_1d_, ctx.Output(InstanceNormV2OutBatchMeanIndex)->GetData()).vec<float>();
  auto save_invstd =
    EigenTensor(batch_channels_1d_, ctx.Output(InstanceNormV2OutBatchVarianceIndex)->GetData()).vec<float>();
  CollectLinearAndConstant(ctx, gamma, beta, running_mean, running_var, save_mean, save_invstd, _alpha_.data(),
                           _beta_.data());
  // cast (B, H, W, C) to (B, H*W, C)
  std::vector<int64_t> shape_3d = {batch, image_size, channel};
  auto x_3d = EigenTensor(shape_3d, ctx.Input(InstanceNormV2InXIndex)->GetData()).tensor<T, kDim3>();
  auto y_3d = EigenTensor(shape_3d, ctx.Output(InstanceNormV2OutYIndex)->GetData()).tensor<T, kDim3>();
  // Apply the linear terms to the input,
  auto loop_transform = [&](int64_t begin, int64_t end) {
    for (int64_t batch_idx = begin; batch_idx < end; ++batch_idx) {
      for (int64_t idx = 0; idx < image_size; ++idx) {
        for (int64_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
          float alpha = _alpha_[batch_idx * channel + channel_idx];
          float beta = _beta_[batch_idx * channel + channel_idx];
          y_3d(batch_idx, idx, channel_idx) =
            static_cast<T>(alpha * static_cast<float>(x_3d(batch_idx, idx, channel_idx)) + beta);
        }
      }
    }
  };
  int64_t block_size = std::max(int64_init_one, (kGrainSize / (channel * image_size)));
  return CpuKernelUtils::ParallelFor(ctx, batch, block_size, loop_transform);
}

template <typename T>
uint32_t InstanceNormV2CpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto batch_mean_ptr = static_cast<float *>(ctx.Output(InstanceNormV2OutBatchMeanIndex)->GetData());
  auto batch_var_ptr = static_cast<float *>(ctx.Output(InstanceNormV2OutBatchVarianceIndex)->GetData());
  (void)std::fill_n(batch_mean_ptr, instance_num, float_init_zero);
  (void)std::fill_n(batch_var_ptr, instance_num, float_init_zero);

  if (is_training_) {
    // UpdateStatsTemplate to init save_mean and save_var
    (void)UpdateStatsTemplate<T, InvStd>(ctx);
  }
  return TransformInput<T>(ctx);
}

uint32_t InstanceNormV2CpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInstanceNormV2InputsNum, kInstanceNormV2OutputNum),
                      "InstanceNormV2 check input and output number failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2ParamCheck(ctx), "InstanceNormV2 check params failed.");
  auto data_type = ctx.Input(InstanceNormV2InXIndex)->GetDataType();
  uint32_t result;
  switch (data_type) {
    case (DT_FLOAT16):
      result = DoCompute<Eigen::half>(ctx);
      break;
    case (DT_FLOAT):
      result = DoCompute<float>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("InstanceNormV2 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("InstanceNormV2 kernel compute failed.");
  }
  return result;
}

REGISTER_CPU_KERNEL(kInstanceNormV2, InstanceNormV2CpuKernel);
}  // namespace aicpu

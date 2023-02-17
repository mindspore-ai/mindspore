/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "instance_norm_v2_grad.h"

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include <vector>
#include <map>
#include <iostream>

namespace {
const char *const kInstanceNormV2Grad = "InstanceNormV2Grad";
constexpr uint32_t kOutputNum = 3;
constexpr uint32_t kInputNum = 7;
constexpr int64_t int64_init_one = 1;
constexpr int64_t kGrainSize = 4 * 1024;
constexpr float float_init_zero = 0.0;
constexpr float float_init_one = 1.0;
constexpr double double_init_zero = 0.0;
constexpr auto kDim3 = 3;
constexpr auto kDim4 = 4;
constexpr auto kDim5 = 5;
constexpr auto InstanceNormV2GradInDyIndex = 0;
constexpr auto InstanceNormV2GradInXIndex = 1;
constexpr auto InstanceNormV2GradInGammaIndex = 2;
constexpr auto InstanceNormV2GradInMeanIndex = 3;
constexpr auto InstanceNormV2GradInVarianceIndex = 4;
constexpr auto InstanceNormV2GradInSaveMeanIndex = 5;
constexpr auto InstanceNormV2GradInSaveVarianceIndex = 6;
constexpr auto InstanceNormV2GradOutDxIndex = 0;
constexpr auto InstanceNormV2GradOutPdGammaIndex = 1;
constexpr auto InstanceNormV2GradOutPdBetaIndex = 2;

inline double FloatToDouble(float v) { return static_cast<double>(v); }
inline double LongToDouble(int64_t v) { return static_cast<double>(v); }
}  // namespace

namespace aicpu {
uint32_t InstanceNormV2GradCpuKernel::InstanceNormV2GradTypeCheck(const CpuKernelContext &ctx) {
  auto dy_type = ctx.Input(InstanceNormV2GradInDyIndex)->GetDataType();
  auto x_type = ctx.Input(InstanceNormV2GradInXIndex)->GetDataType();
  auto gamma_type = ctx.Input(InstanceNormV2GradInGammaIndex)->GetDataType();
  auto mean_type = ctx.Input(InstanceNormV2GradInMeanIndex)->GetDataType();
  auto variance_type = ctx.Input(InstanceNormV2GradInVarianceIndex)->GetDataType();
  auto save_mean_type = ctx.Input(InstanceNormV2GradInSaveMeanIndex)->GetDataType();
  auto save_variance_type = ctx.Input(InstanceNormV2GradInSaveVarianceIndex)->GetDataType();

  if (dy_type != x_type) {
    KERNEL_LOG_ERROR(
      "For primitive[%s]'s input arguments dy and x should have the same "
      "data type, but dy type is [%s], x type is [%s].",
      kInstanceNormV2Grad, DTypeStr(dy_type).c_str(), DTypeStr(x_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const std::map<std::string, DataType> types = {{"gamma", gamma_type},
                                                 {"mean", mean_type},
                                                 {"variance", variance_type},
                                                 {"save_mean", save_mean_type},
                                                 {"save_variance", save_variance_type}};
  return CheckTensorTypeSame(types, DT_FLOAT, kInstanceNormV2Grad);
}

uint32_t InstanceNormV2GradCpuKernel::InstanceNormV2GradShapeCheck(const CpuKernelContext &ctx) {
  auto dy_shape_ptr = ctx.Input(InstanceNormV2GradInDyIndex)->GetTensorShape();
  auto x_shape_ptr = ctx.Input(InstanceNormV2GradInXIndex)->GetTensorShape();
  auto gamma_shape_ptr = ctx.Input(InstanceNormV2GradInGammaIndex)->GetTensorShape();
  auto mean_shape_ptr = ctx.Input(InstanceNormV2GradInMeanIndex)->GetTensorShape();
  auto variance_shape_ptr = ctx.Input(InstanceNormV2GradInVarianceIndex)->GetTensorShape();
  auto save_mean_shape_ptr = ctx.Input(InstanceNormV2GradInSaveMeanIndex)->GetTensorShape();
  auto save_variance_shape_ptr = ctx.Input(InstanceNormV2GradInSaveVarianceIndex)->GetTensorShape();
  auto pd_gamma_shape_ptr = ctx.Output(InstanceNormV2GradOutPdGammaIndex)->GetTensorShape();
  auto pd_beta_shape_ptr = ctx.Output(InstanceNormV2GradOutPdBetaIndex)->GetTensorShape();

  auto dy_shape = dy_shape_ptr->GetDimSizes();
  auto res = CheckTensorShapeSame({{"input x", x_shape_ptr}}, dy_shape, kInstanceNormV2Grad);
  if (res != KERNEL_STATUS_OK) {
    return res;
  };
  auto x_format = x_shape_ptr->GetFormat();
  std::vector<int64_t> check_shape;
  check_shape = dy_shape;
  int64_t image_size = 0;
  if (dy_shape.size() == kDim4) {
    if (x_format == FORMAT_NCHW || x_format == FORMAT_ND) {
      // consider (N, C, H, W) as (N*C, H, W, 1), similar to (N, H, W, C)
      check_shape[kFormatNCHWIndexH] = int64_init_one;
      check_shape[kFormatNCHWIndexW] = int64_init_one;
      image_size = dy_shape[kFormatNCHWIndexH] * dy_shape[kFormatNCHWIndexW];
      instance_num = dy_shape[kFormatNCHWIndexN] * dy_shape[kFormatNCHWIndexC];
      constexpr int64_t kNumberOne = 1;
      dy_shape_4d_ = {dy_shape[kFormatNCHWIndexN] * dy_shape[kFormatNCHWIndexC], dy_shape[kFormatNCHWIndexH],
                      dy_shape[kFormatNCHWIndexW], kNumberOne};
      batch_channels_2d_ = {dy_shape[kFormatNCHWIndexN] * dy_shape[kFormatNCHWIndexC], kNumberOne};
    } else if (x_format == FORMAT_NHWC) {
      check_shape[kFormatNHWCIndexH] = int64_init_one;
      check_shape[kFormatNHWCIndexW] = int64_init_one;
      image_size = dy_shape[kFormatNHWCIndexH] * dy_shape[kFormatNHWCIndexW];
      instance_num = dy_shape[kFormatNHWCIndexN] * dy_shape[kFormatNHWCIndexC];
      dy_shape_4d_ = dy_shape;
      batch_channels_2d_ = {dy_shape[kFormatNHWCIndexN], dy_shape[kFormatNHWCIndexC]};
    }
  } else if (x_format == FORMAT_NC1HWC0 || dy_shape.size() == kDim5) {
    // consider (N, C1, H, W, C0) as (N*C1, H, W, C0), similar to (N, H, W, C)
    check_shape[kFormatNC1HWC0IndexH] = int64_init_one;
    check_shape[kFormatNC1HWC0IndexW] = int64_init_one;
    image_size = dy_shape[kFormatNC1HWC0IndexH] * dy_shape[kFormatNC1HWC0IndexW];
    instance_num = dy_shape[kFormatNC1HWC0IndexN] * dy_shape[kFormatNC1HWC0IndexC1] * dy_shape[kFormatNC1HWC0IndexC0];
    dy_shape_4d_ = {dy_shape[kFormatNC1HWC0IndexN] * dy_shape[kFormatNC1HWC0IndexC1], dy_shape[kFormatNC1HWC0IndexH],
                    dy_shape[kFormatNC1HWC0IndexW], dy_shape[kFormatNC1HWC0IndexC0]};
    batch_channels_2d_ = {dy_shape[kFormatNC1HWC0IndexN] * dy_shape[kFormatNC1HWC0IndexC1],
                          dy_shape[kFormatNC1HWC0IndexC0]};
  } else {
    KERNEL_LOG_ERROR(
      "For primitive[%s]'s input arguments dy and x only "
      "support NHWC, NCHW and NC1HWC0, but get data format [%s]",
      kInstanceNormV2Grad, FormatToSerialString(x_format).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  constexpr int64_t image_min = 1;
  if (image_size <= image_min) {
    KERNEL_LOG_ERROR(
      "For primitive[%s], expected more than 1 value per instance, but get "
      "[%ld] value per instance.",
      kInstanceNormV2Grad, image_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  const std::map<std::string, TensorShapePtr> shapes = {{"gamma", gamma_shape_ptr},
                                                        {"mean", mean_shape_ptr},
                                                        {"variance", variance_shape_ptr},
                                                        {"save_mean", save_mean_shape_ptr},
                                                        {"save_variance", save_variance_shape_ptr},
                                                        {"pd_gamma", pd_gamma_shape_ptr},
                                                        {"pd_beta", pd_beta_shape_ptr}};
  return CheckTensorShapeSame(shapes, check_shape, kInstanceNormV2Grad);
}

uint32_t InstanceNormV2GradCpuKernel::InstanceNormV2GradAttrCheck(const CpuKernelContext &ctx) {
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
      kInstanceNormV2Grad, epsilon_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto is_training_ptr = ctx.GetAttr("is_training");
  if (is_training_ptr) {
    is_training_ = is_training_ptr->GetBool();
  }
  return KERNEL_STATUS_OK;
}

uint32_t InstanceNormV2GradCpuKernel::InstanceNormV2GradParamCheck(const CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(InstanceNormV2GradTypeCheck(ctx), "InstanceNormV2Grad check type failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2GradShapeCheck(ctx), "InstanceNormV2Grad check shape failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2GradAttrCheck(ctx), "InstanceNormV2Grad check attr failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t InstanceNormV2GradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  const int64_t batch = dy_shape_4d_[kFormatNHWCIndexN];
  const int64_t channel = dy_shape_4d_[kFormatNHWCIndexC];
  const int64_t image_size = dy_shape_4d_[kFormatNHWCIndexH] * dy_shape_4d_[kFormatNHWCIndexW];
  std::vector<int64_t> dy_shape_3d_ = {batch, image_size, channel};
  auto dy_3d = EigenTensor(dy_shape_3d_, ctx.Input(InstanceNormV2GradInDyIndex)->GetData()).tensor<T, kDim3>();
  auto in_x_3d = EigenTensor(dy_shape_3d_, ctx.Input(InstanceNormV2GradInXIndex)->GetData()).tensor<T, kDim3>();
  auto weight_matrix =
    EigenTensor(batch_channels_2d_, ctx.Input(InstanceNormV2GradInGammaIndex)->GetData()).matrix<float>();
  auto running_mean_matrix =
    EigenTensor(batch_channels_2d_, ctx.Input(InstanceNormV2GradInMeanIndex)->GetData()).matrix<float>();
  auto running_var_matrix =
    EigenTensor(batch_channels_2d_, ctx.Input(InstanceNormV2GradInVarianceIndex)->GetData()).matrix<float>();
  auto save_mean_matrix =
    EigenTensor(batch_channels_2d_, ctx.Input(InstanceNormV2GradInSaveMeanIndex)->GetData()).matrix<float>();
  auto save_invstd_matrix =
    EigenTensor(batch_channels_2d_, ctx.Input(InstanceNormV2GradInSaveVarianceIndex)->GetData()).matrix<float>();

  auto dx_3d = EigenTensor(dy_shape_3d_, ctx.Output(InstanceNormV2GradOutDxIndex)->GetData()).tensor<T, kDim3>();
  auto grad_weight_matrix =
    EigenTensor(batch_channels_2d_, ctx.Output(InstanceNormV2GradOutPdGammaIndex)->GetData()).matrix<float>();
  auto grad_bias_matrix =
    EigenTensor(batch_channels_2d_, ctx.Output(InstanceNormV2GradOutPdBetaIndex)->GetData()).matrix<float>();
  auto loop_batch = [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      for (int64_t c_idx = 0; c_idx < channel; ++c_idx) {
        float w = weight_matrix(idx, c_idx);
        float mean = float_init_zero, invstd = float_init_zero;
        mean = is_training_ ? save_mean_matrix(idx, c_idx) : running_mean_matrix(idx, c_idx);
        float _invstd_ = std::sqrt(running_var_matrix(idx, c_idx) + epsilon_);
        invstd = is_training_ ? save_invstd_matrix(idx, c_idx) : float_init_one / _invstd_;

        double sum = double_init_zero, dotp = double_init_zero;
        for (int64_t img_idx = 0; img_idx < image_size; ++img_idx) {
          sum += static_cast<double>(dy_3d(idx, img_idx, c_idx));
          dotp += (static_cast<double>(in_x_3d(idx, img_idx, c_idx)) - FloatToDouble(mean)) *
                  static_cast<double>(dy_3d(idx, img_idx, c_idx));
        }
        float k = static_cast<float>(dotp * FloatToDouble(invstd) * FloatToDouble(invstd) / LongToDouble(image_size));
        float grad_mean = static_cast<float>(sum / LongToDouble(image_size));
        for (int64_t img_idx = 0; img_idx < image_size; ++img_idx) {
          float _dx_ = (static_cast<float>(in_x_3d(idx, img_idx, c_idx)) - mean) * k;
          dx_3d(idx, img_idx, c_idx) =
            is_training_
              ? static_cast<T>((static_cast<float>(dy_3d(idx, img_idx, c_idx)) - grad_mean - _dx_) * invstd * w)
              : static_cast<T>(static_cast<float>(dy_3d(idx, img_idx, c_idx)) * invstd * w);
        }
        grad_weight_matrix(idx, c_idx) = static_cast<float>(dotp * FloatToDouble(invstd));
        grad_bias_matrix(idx, c_idx) = static_cast<float>(sum);
      }
    }
  };
  int64_t block_size = std::max(int64_init_one, (kGrainSize / (channel * image_size)));
  return CpuKernelUtils::ParallelFor(ctx, batch, block_size, loop_batch);
}

uint32_t InstanceNormV2GradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "InstanceNormV2Grad check input and output number failed.");
  KERNEL_HANDLE_ERROR(InstanceNormV2GradParamCheck(ctx), "InstanceNormV2Grad check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t result;
  switch (data_type) {
    case (DT_FLOAT16):
      result = DoCompute<Eigen::half>(ctx);
      break;
    case (DT_FLOAT):
      result = DoCompute<float>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("InstanceNormV2Grad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  if (result != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("InstanceNormV2Grad kernel compute failed.");
  }
  return result;
}

REGISTER_CPU_KERNEL(kInstanceNormV2Grad, InstanceNormV2GradCpuKernel);
}  // namespace aicpu

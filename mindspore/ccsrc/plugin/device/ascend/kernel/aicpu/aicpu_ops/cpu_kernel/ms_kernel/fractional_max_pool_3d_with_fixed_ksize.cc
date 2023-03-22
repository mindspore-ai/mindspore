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

#include "fractional_max_pool_3d_with_fixed_ksize.h"

#include <iostream>
#include <limits>
#include <vector>
#include <cmath>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 2;
const uint32_t Num2 = 2;
const uint32_t Num3 = 3;
const uint32_t Num4 = 4;
const uint32_t Num5 = 5;
const char *kFractionalMaxPool3DWithFixedKsize = "FractionalMaxPool3DWithFixedKsize";
}  // namespace

namespace aicpu {
uint32_t FractionalMaxPool3DWithFixedKsizeCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *random_samples = ctx.Input(1);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "FractionalMaxPool3DWithFixedKsize check params failed.");
  AttrValue *output_shape = ctx.GetAttr("output_shape");
  KERNEL_CHECK_NULLPTR(output_shape, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr:output_shape failed.",
                       kFractionalMaxPool3DWithFixedKsize);
  output_size = output_shape->GetListInt();
  KERNEL_CHECK_FALSE(output_size.size() == 1 || output_size.size() == Num3, KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize: output_size must be equal to 1 or 3.");
  if (output_size.size() == 1) {
    for (int64_t i = 0; i < Num3; i++) {
      output_size[i] = output_size[0];
    }
  }
  AttrValue *ksize = ctx.GetAttr("ksize");
  KERNEL_CHECK_NULLPTR(ksize, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr:ksize failed.",
                       kFractionalMaxPool3DWithFixedKsize);
  kernel_size = ksize->GetListFloat();
  KERNEL_CHECK_FALSE(kernel_size.size() == 1 || kernel_size.size() == Num3, KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize: kernel_size must be equal to 1 or 3.");
  if (kernel_size.size() == 1) {
    for (int64_t i = 0; i < Num3; i++) {
      kernel_size[i] = kernel_size[0];
    }
  }
  random_samples_shape = random_samples->GetTensorShape()->GetDimSizes();
  input_shape = input->GetTensorShape()->GetDimSizes();
  int64_t input_dims = input_shape.size();
  for (int64_t i = 0; i < input_dims; i++) {
    KERNEL_CHECK_FALSE((input->GetTensorShape()->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "FractionalMaxPool3DWithFixedKsize: expected input to have non-empty spatial dimensions, "
                       "but input has sizes [%d] with dimension [%d] being empty.",
                       input_dims, i);
  }
  KERNEL_CHECK_FALSE((input_dims == Num4 || input_dims == Num5), KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] or [5D] (batch mode) tensor expected for input.");

  return KERNEL_STATUS_OK;
}

template <typename random_sample_t>
static std::vector<int> generate_intervals(random_sample_t random_sample, int input_size, int output_size,
                                           float kernel_size) {
  std::vector<int> sequence(output_size);
  if (output_size > 1) {
    random_sample_t alpha =
      static_cast<random_sample_t>(input_size - kernel_size) / static_cast<random_sample_t>(output_size - 1);

    for (int i = 0; i < output_size - 1; ++i) {
      sequence[i] = static_cast<int>((static_cast<random_sample_t>(i) + random_sample) * alpha) -
                    static_cast<int>(random_sample * alpha);
    }
  }
  sequence[output_size - 1] = input_size - kernel_size;

  return sequence;
}

template <typename scalar_t, typename random_sample_t, typename argmax_t>
uint32_t FractionalMaxPool3DWithFixedKsizeCpuKernel::FractionalMaxPool3DWithFixedKsizeOutCpuTemplate(
  CpuKernelContext &ctx) {
  auto input_data = reinterpret_cast<scalar_t *>(ctx.Input(0)->GetData());
  auto random_samples_data = reinterpret_cast<random_sample_t *>(ctx.Input(1)->GetData());
  auto output_data = reinterpret_cast<scalar_t *>(ctx.Output(0)->GetData());
  auto argmax_data = reinterpret_cast<argmax_t *>(ctx.Output(1)->GetData());
  int64_t input_dims = input_shape.size();
  int64_t random_samples_dims = random_samples_shape.size();
  std::string format = "NCDHW";
  AttrValue *data_format = ctx.GetAttr("data_format");
  if (data_format != nullptr) {
    format = data_format->GetString();
  }
  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];
  float kernelsizeT = kernel_size[0];
  float kernelsizeH = kernel_size[1];
  float kernelsizeW = kernel_size[2];
  int64_t inputN = 0;
  int64_t inputC = 0;
  int64_t inputT = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  if (format == "NCDHW") {
    if (input_dims == Num5) {
      inputN = input_shape[0];
      inputC = input_shape[1];
      inputT = input_shape[Num2];
      inputH = input_shape[Num3];
      inputW = input_shape[Num4];
    } else {
      inputC = input_shape[0];
      inputT = input_shape[1];
      inputH = input_shape[Num2];
      inputW = input_shape[Num3];
    }
  } else {
    if (input_dims == Num5) {
      inputN = input_shape[0];
      inputC = input_shape[Num4];
      inputT = input_shape[1];
      inputH = input_shape[Num2];
      inputW = input_shape[Num3];
    } else {
      inputC = input_shape[Num3];
      inputT = input_shape[0];
      inputH = input_shape[1];
      inputW = input_shape[Num2];
    }
  }
  KERNEL_CHECK_FALSE((outputT + kernelsizeT - 1 < inputT), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize out(): pool time [%d],"
                     "too large relative to input time [%d].",
                     kernelsizeT, inputT);
  KERNEL_CHECK_FALSE((outputH + kernelsizeH - 1 < inputH), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize out(): pool height [%d],"
                     "too large relative to input height [%d].",
                     kernelsizeH, inputH);
  KERNEL_CHECK_FALSE((outputW + kernelsizeW - 1 < inputW), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize out(): pool width [%d],"
                     "too large relative to input width [%d].",
                     kernelsizeW, inputW);
  KERNEL_CHECK_FALSE((random_samples_dims == Num3), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize: random_samples' size must be equal to 3.");
  KERNEL_CHECK_FALSE((random_samples_shape[Num2] == Num3), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalMaxPool3DWithFixedKsize: The third dim of random_samples must be 3, but got [%d].",
                     random_samples_shape[Num2]);
  if (input_dims == Num4) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > inputC) {
      max_core_num = inputC;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num should not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, inputC, inputC / max_core_num, [&](int64_t start, int64_t end) {
      for (auto plane = start; plane < end; ++plane) {
        /* each plane contains 3 random random_samples,
           one for T, one for W, and one for H */
        random_sample_t *random_samplesForPlane = random_samples_data + plane * 3;

        /* Generate interval sequence */
        auto sequenceT = generate_intervals<random_sample_t>(random_samplesForPlane[0], inputT, outputT, kernelsizeT);
        auto sequenceH = generate_intervals<random_sample_t>(random_samplesForPlane[1], inputH, outputH, kernelsizeH);
        auto sequenceW = generate_intervals<random_sample_t>(random_samplesForPlane[2], inputW, outputW, kernelsizeW);

        /* loop over output */
        int64_t t, h, w;

        scalar_t *inputForPlane = input_data + plane * inputT * inputH * inputW;
        scalar_t *outputForPlane = output_data + plane * outputT * outputH * outputW;
        argmax_t *argmaxForPlane = argmax_data + plane * outputT * outputH * outputW;

        for (t = 0; t < outputT; ++t) {
          int64_t inputTStart = sequenceT[t];

          for (h = 0; h < outputH; ++h) {
            int64_t inputHStart = sequenceH[h];

            for (w = 0; w < outputW; ++w) {
              int64_t inputWStart = sequenceW[w];

              int64_t t2 = inputTStart, h2 = inputHStart, w2 = inputWStart;
              scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
              argmax_t maxIndex = t2 * inputH * inputW + h2 * inputW + w2;

              for (t2 = inputTStart; t2 < inputTStart + kernelsizeT; ++t2) {
                for (h2 = inputHStart; h2 < inputHStart + kernelsizeH; ++h2) {
                  for (w2 = inputWStart; w2 < inputWStart + kernelsizeW; ++w2) {
                    KERNEL_CHECK_FALSE(t2 >= 0 && t2 < inputT, KERNEL_STATUS_PARAM_INVALID,
                                       "FractionalMaxPool3DWithFixedKsize index T value is illegal.");
                    KERNEL_CHECK_FALSE(h2 >= 0 && h2 < inputH, KERNEL_STATUS_PARAM_INVALID,
                                       "FractionalMaxPool3DWithFixedKsize index H value is illegal.");
                    KERNEL_CHECK_FALSE(w2 >= 0 && w2 < inputW, KERNEL_STATUS_PARAM_INVALID,
                                       "FractionalMaxPool3DWithFixedKsize index W value is illegal.");
                    argmax_t planeIndex = t2 * inputH * inputW + h2 * inputW + w2;
                    scalar_t val = inputForPlane[planeIndex];
                    if (val > maxVal || std::isnan(static_cast<double>(val))) {
                      maxVal = val;
                      maxIndex = planeIndex;
                    }
                  }
                }
              }

              outputForPlane[t * outputH * outputW + h * outputW + w] = maxVal;
              argmaxForPlane[t * outputH * outputW + h * outputW + w] = maxIndex;
            }
          }
        }
      }
      return KERNEL_STATUS_OK;
    });
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > inputN) {
      max_core_num = inputN;
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num should not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, inputN, inputN / max_core_num, [&](int64_t start, int64_t end) {
      for (auto batch = start; batch < end; ++batch) {
        for (auto plane = 0; plane < inputC; ++plane) {
          /* each plane contains 3 random random_samples,
            one for T, one for W, and one for H */
          auto intput_data_n = input_data + batch * inputC * inputW * inputH * inputT;
          auto output_data_n = output_data + batch * inputC * outputW * outputH * outputT;
          auto argmax_data_n = argmax_data + batch * inputC * outputW * outputH * outputT;
          auto random_samples_data_n = random_samples_data + batch * inputC * 3;
          random_sample_t *random_samplesForPlane = random_samples_data_n + plane * 3;

          /* Generate interval sequence */
          auto sequenceT = generate_intervals<random_sample_t>(random_samplesForPlane[0], inputT, outputT, kernelsizeT);
          auto sequenceH = generate_intervals<random_sample_t>(random_samplesForPlane[1], inputH, outputH, kernelsizeH);
          auto sequenceW = generate_intervals<random_sample_t>(random_samplesForPlane[2], inputW, outputW, kernelsizeW);

          /* loop over output */
          int64_t t, h, w;

          scalar_t *inputForPlane = intput_data_n + plane * inputT * inputH * inputW;
          scalar_t *outputForPlane = output_data_n + plane * outputT * outputH * outputW;
          argmax_t *argmaxForPlane = argmax_data_n + plane * outputT * outputH * outputW;

          for (t = 0; t < outputT; ++t) {
            int64_t inputTStart = sequenceT[t];

            for (h = 0; h < outputH; ++h) {
              int64_t inputHStart = sequenceH[h];

              for (w = 0; w < outputW; ++w) {
                int64_t inputWStart = sequenceW[w];

                int64_t t2 = inputTStart, h2 = inputHStart, w2 = inputWStart;
                scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
                argmax_t maxIndex = t2 * inputH * inputW + h2 * inputW + w2;

                for (t2 = inputTStart; t2 < inputTStart + kernelsizeT; ++t2) {
                  for (h2 = inputHStart; h2 < inputHStart + kernelsizeH; ++h2) {
                    for (w2 = inputWStart; w2 < inputWStart + kernelsizeW; ++w2) {
                      KERNEL_CHECK_FALSE(t2 >= 0 && t2 < inputT, KERNEL_STATUS_PARAM_INVALID,
                                         "FractionalMaxPool3DWithFixedKsize index T value is illegal.");
                      KERNEL_CHECK_FALSE(h2 >= 0 && h2 < inputH, KERNEL_STATUS_PARAM_INVALID,
                                         "FractionalMaxPool3DWithFixedKsize index H value is illegal.");
                      KERNEL_CHECK_FALSE(w2 >= 0 && w2 < inputW, KERNEL_STATUS_PARAM_INVALID,
                                         "FractionalMaxPool3DWithFixedKsize index W value is illegal.");
                      argmax_t planeIndex = t2 * inputH * inputW + h2 * inputW + w2;
                      scalar_t val = inputForPlane[planeIndex];
                      if (val > maxVal || std::isnan(static_cast<double>(val))) {
                        maxVal = val;
                        maxIndex = planeIndex;
                      }
                    }
                  }
                }

                outputForPlane[t * outputH * outputW + h * outputW + w] = maxVal;
                argmaxForPlane[t * outputH * outputW + h * outputW + w] = maxIndex;
              }
            }
          }
        }
      }
      return KERNEL_STATUS_OK;
    });
  }
  return KERNEL_STATUS_OK;
}
template <typename scalar_t, typename random_sample_t>
uint32_t FractionalMaxPool3DWithFixedKsizeCpuKernel::DoComputeWithArgmaxType(CpuKernelContext &ctx,
                                                                             DataType argmax_type) {
  switch (argmax_type) {
    case DT_INT32:
      return FractionalMaxPool3DWithFixedKsizeOutCpuTemplate<scalar_t, random_sample_t, int32_t>(ctx);
    case DT_INT64:
      return FractionalMaxPool3DWithFixedKsizeOutCpuTemplate<scalar_t, random_sample_t, int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("argmax_type [%s] must be in [{DT_INT32, DT_INT64}].", DTypeStr(argmax_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename scalar_t>
uint32_t FractionalMaxPool3DWithFixedKsizeCpuKernel::DoComputeWithRandomSamplesType(CpuKernelContext &ctx,
                                                                                    DataType random_samples_type) {
  auto argmax_type = ctx.Output(1)->GetDataType();
  switch (random_samples_type) {
    case DT_FLOAT16:
      return DoComputeWithArgmaxType<scalar_t, Eigen::half>(ctx, argmax_type);
    case DT_FLOAT:
      return DoComputeWithArgmaxType<scalar_t, float>(ctx, argmax_type);
    case DT_DOUBLE:
      return DoComputeWithArgmaxType<scalar_t, double>(ctx, argmax_type);
    default:
      KERNEL_LOG_ERROR("random_samples_type [%s] must be in [{DT_FLOAT16, DT_FLOAT, DT_DOUBLE}].",
                       DTypeStr(random_samples_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

uint32_t FractionalMaxPool3DWithFixedKsizeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(GetInputAndCheck(ctx), "FractionalMaxPool3DWithFixedKsize check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  auto random_samples_type = ctx.Input(1)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return DoComputeWithRandomSamplesType<Eigen::half>(ctx, random_samples_type);
    case DT_FLOAT:
      return DoComputeWithRandomSamplesType<float>(ctx, random_samples_type);
    case DT_DOUBLE:
      return DoComputeWithRandomSamplesType<double>(ctx, random_samples_type);
    case DT_INT32:
      return DoComputeWithRandomSamplesType<int32_t>(ctx, random_samples_type);
    case DT_INT64:
      return DoComputeWithRandomSamplesType<int64_t>(ctx, random_samples_type);
    default:
      KERNEL_LOG_ERROR("FractionalMaxPool3DWithFixedKsize kernel data type [%s] not support.",
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kFractionalMaxPool3DWithFixedKsize, FractionalMaxPool3DWithFixedKsizeCpuKernel);
}  // namespace aicpu

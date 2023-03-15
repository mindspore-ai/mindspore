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

#include "scale_and_translate.h"

#include <iostream>
#include <type_traits>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "../utils/sampling_kernels.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
constexpr int64_t kParallelDataNums = 1024;
const char *kScaleAndTranslate = "ScaleAndTranslate";
const char *kScaleAndTranslateGrad = "ScaleAndTranslateGrad";
#define SCALEANDTRANSLATE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                   \
    uint32_t result = ScaleAndTranslateCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                               \
      KERNEL_LOG_ERROR("ScaleAndTranslate kernel compute failed."); \
      return result;                                                \
    }                                                               \
    break;                                                          \
  }

#define SCALEANDTRANSLATEGRAD_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                                       \
    uint32_t result = ScaleAndTranslateGradCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                                   \
      KERNEL_LOG_ERROR("ScaleAndTranslateGrad kernel compute failed."); \
      return result;                                                    \
    }                                                                   \
    break;                                                              \
  }

#define SWITCH_PARALLEL(SHARD, end_num, ctx)                                 \
  if ((end_num) <= kParallelDataNums) {                                      \
    for (size_t i = 0; i < size_t(end_num); i++) {                           \
      SHARD(i, i + 1);                                                       \
    }                                                                        \
  } else {                                                                   \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, 1, SHARD), \
                        "ScaleAndTranslate #SHARD Compute failed.");         \
  }

}  // namespace

namespace aicpu {
namespace {
template <typename T>
inline const T &Clamp(const T &low, const T &high, const T &value) {
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

template <typename Kernel>
uint32_t ComputeSpansCore(CpuKernelContext &context, const Kernel &kernel, const int64_t output_size,
                          const int64_t input_size, const float scale, const float translate, const bool antialias,
                          Spans *spans) {
  const float inv_scale = 1.0 / scale;
  const float inv_translate = -inv_scale * translate;
  const float kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
  spans->span_size =
    std::min(2 * static_cast<int>(std::ceil(kernel.Radius() * kernel_scale)) + 1, static_cast<int>(input_size));

  spans->starts = new Eigen::Tensor<int32_t, 1>(output_size);

  spans->weights = new Eigen::Tensor<float, 1>(spans->span_size * output_size);
  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
  weights_vec.setZero();

  const float one_over_kernel_scale = 1.0f / kernel_scale;
  int max_span_size = 0;
  std::vector<float> temp_weights;
  auto shard_x = [&](int start, int end) {
    for (auto x = start; x < end; ++x) {
      const float col_f = x + 0.5f;
      const float sample_f = col_f * inv_scale + inv_translate;
      // Don't sample when the sampling location is outside the source image.
      if (sample_f < 0 || sample_f > input_size) {
        // Add an empty span. 添加一个空跨度
        starts_vec(x) = 0;
        continue;
      }
      int64_t span_start = std::ceil(sample_f - kernel.Radius() * kernel_scale - 0.5f);
      int64_t span_end = std::floor(sample_f + kernel.Radius() * kernel_scale - 0.5f);
      span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
      span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
      const int this_span_size = span_end - span_start;
      if (this_span_size > spans->span_size) {
        KERNEL_LOG_ERROR("Span is too large: [%d] vs [%d].", this_span_size, spans->span_size);
      }
      float total_weight_sum = 0.0f;
      temp_weights.clear();
      for (int source = span_start; source < span_end; ++source) {
        float kernel_pos = static_cast<float>(source) + 0.5f - sample_f;
        float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
        total_weight_sum += weight;
        temp_weights.push_back(weight);
      }
      max_span_size = std::max(max_span_size, this_span_size);
      if (std::abs(total_weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
        float one_over_total_weight_sum = 1.0f / total_weight_sum;
        int out_index = spans->span_size * x;
        for (float weight : temp_weights) {
          weights_vec(out_index) = weight * one_over_total_weight_sum;
          ++out_index;
        }
      }
      starts_vec(x) = span_start;
    }
  };
  SWITCH_PARALLEL(shard_x, output_size, context);
  return KERNEL_STATUS_OK;
}

uint32_t ComputeGradSpansCore(CpuKernelContext &context, const Spans *spans, const int64_t forward_output_size,
                              const int64_t forward_input_size, Spans *grad_spans) {
  struct GradComponent {
    int index;
    float weight;
  };
  std::vector<std::vector<GradComponent>> grad_components(forward_input_size);

  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
  auto shard_grad_output = [&](int start, int end) {
    for (auto output_index = start; output_index < end; ++output_index) {
      int input_index = starts_vec(output_index);
      for (int j = 0; j < spans->span_size; ++j, ++input_index) {
        const float weight = weights_vec(output_index * spans->span_size + j);
        if (weight != 0.0f && input_index < forward_input_size) {
          grad_components[input_index].push_back(GradComponent{output_index, weight});
        }
      }
    }
  };
  SWITCH_PARALLEL(shard_grad_output, forward_output_size, context);
  int max_size = 0;
  for (std::vector<GradComponent> &gc : grad_components) {
    if (!gc.empty()) {
      std::sort(gc.begin(), gc.end(),
                [](const GradComponent &x1, const GradComponent &x2) { return x1.index < x2.index; });
      max_size = std::max(gc.back().index - gc.front().index + 1, max_size);
    }
  }
  grad_spans->span_size = max_size;

  grad_spans->starts = new Eigen::Tensor<int32_t, 1>(forward_input_size);
  grad_spans->weights = new Eigen::Tensor<float, 1>(grad_spans->span_size * forward_input_size);
  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> grad_starts_vec(grad_spans->starts->data(),
                                                              grad_spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> grad_weights_vec(grad_spans->weights->data(),
                                                             grad_spans->weights->dimensions());
  grad_weights_vec.setZero();
  auto shard_grad_input = [&](int start, int end) {
    for (int input_index = start; input_index < end; ++input_index) {
      if (!grad_components[input_index].empty()) {
        const int start_span = grad_components[input_index].front().index;
        grad_starts_vec(input_index) = start_span;
        for (const GradComponent &gc : grad_components[input_index]) {
          grad_weights_vec(input_index * grad_spans->span_size + gc.index - start_span) += gc.weight;
        }
      } else {
        grad_starts_vec(input_index) = 0;
      }
    }
  };
  SWITCH_PARALLEL(shard_grad_input, forward_input_size, context);
  return KERNEL_STATUS_OK;
}

uint32_t ComputeSpans(CpuKernelContext &context, const aicpu::SamplingKernelType kernel_type, const int64_t output_size,
                      const int64_t input_size, const float scale, const float translate, const bool antialias,
                      Spans *spans) {
  switch (kernel_type) {
    case Lanczos1Kernel: {
      return ComputeSpansCore(context, CreateLanczos1Kernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case Lanczos3Kernel: {
      return ComputeSpansCore(context, CreateLanczos3Kernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case Lanczos5Kernel: {
      return ComputeSpansCore(context, CreateLanczos5Kernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case GaussianKernel: {
      return ComputeSpansCore(context, CreateGaussianKernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case BoxKernel: {
      return ComputeSpansCore(context, CreateBoxKernel(), output_size, input_size, scale, translate, antialias, spans);
    }
    case TriangleKernel: {
      return ComputeSpansCore(context, CreateTriangleKernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case KeysCubicKernel: {
      return ComputeSpansCore(context, CreateKeysCubicKernel(), output_size, input_size, scale, translate, antialias,
                              spans);
    }
    case MitchellCubicKernel: {
      return ComputeSpansCore(context, CreateMitchellCubicKernel(), output_size, input_size, scale, translate,
                              antialias, spans);
    }
    default:
      KERNEL_LOG_ERROR("kernel_type kernel data type [%u] not support.", kernel_type);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t ComputeGradSpans(CpuKernelContext &context, const SamplingKernelType kernel_type,
                          const int64_t forward_output_size, const int64_t forward_input_size, const float scale,
                          const float translate, const bool antialias, Spans *grad_spans) {
  Spans spans;
  ComputeSpans(context, kernel_type, forward_output_size, forward_input_size, scale, translate, antialias, &spans);
  uint32_t Status = ComputeGradSpansCore(context, &spans, forward_output_size, forward_input_size, grad_spans);
  delete spans.starts;
  delete spans.weights;
  return Status;
}
}  // namespace

uint32_t ScaleAndTranslateCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ScaleAndTranslate check input and output number failed.");
  KERNEL_HANDLE_ERROR(ScaleAndTranslateCheck(ctx), "ScaleAndTranslate check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SCALEANDTRANSLATE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("ScaleAndTranslate kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

// namespace aicpu {
uint32_t ScaleAndTranslateGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "ScaleAndTranslateGrad check input and output number failed.");
  KERNEL_HANDLE_ERROR(ScaleAndTranslateGradCheck(ctx), "ScaleAndTranslateGrad check params failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    SCALEANDTRANSLATEGRAD_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      KERNEL_LOG_ERROR("ScaleAndTranslateGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCheck(CpuKernelContext &ctx) {
  auto input0_shape = ctx.Input(0)->GetTensorShape();
  auto input1_shape = ctx.Input(1)->GetTensorShape();
  auto input2_shape = ctx.Input(2)->GetTensorShape();
  auto input3_shape = ctx.Input(3)->GetTensorShape();
  // dims check
  KERNEL_CHECK_FALSE((input0_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "The input0's dims=[%d] must be 4-dimensional", input0_shape->GetDims())
  KERNEL_CHECK_FALSE((input1_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "The input1's dims=[%d] must be 1-dimensional", input1_shape->GetDims())
  KERNEL_CHECK_FALSE((input1_shape->NumElements() == 2), KERNEL_STATUS_PARAM_INVALID,
                     "The input1's numelements=[%d] must have two elements", input1_shape->NumElements())

  DataType input1_type = ctx.Input(1)->GetDataType();
  DataType input2_type = ctx.Input(2)->GetDataType();
  DataType input3_type = ctx.Input(3)->GetDataType();

  // dtypes check
  KERNEL_CHECK_FALSE((input1_type == DT_INT32), KERNEL_STATUS_PARAM_INVALID, "The input1's dtype=[%d] must be DT_INT32",
                     DTypeStr(input1_type).c_str())
  KERNEL_CHECK_FALSE((input2_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID, "The input2's dtype=[%d] must be DT_FLOAT",
                     DTypeStr(input2_type).c_str())
  KERNEL_CHECK_FALSE((input3_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID, "The input3's dtype=[%d] must be DT_FLOAT",
                     DTypeStr(input3_type).c_str())

  KERNEL_LOG_INFO(
    "ScaleAndTranslateCpuKernel[%s], input0: size[%llu], input1: size[%llu];"
    "input2: size[%llu], input3: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(), ctx.Input(2)->GetDataSize(),
    ctx.Input(3)->GetDataSize(), ctx.Output(0)->GetDataSize());
  return KERNEL_STATUS_OK;
}

uint32_t ScaleAndTranslateGradCpuKernel::ScaleAndTranslateGradCheck(CpuKernelContext &ctx) {
  auto input0_shape = ctx.Input(0)->GetTensorShape();  // batch_size
  auto input1_shape = ctx.Input(1)->GetTensorShape();  // forward_input_height
  auto input2_shape = ctx.Input(2)->GetTensorShape();  // channels
  auto input3_shape = ctx.Input(3)->GetTensorShape();  // forward_input_width
  // dims check
  KERNEL_CHECK_FALSE((input0_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "The input_grad dims=[%d] must be 4-dimensional", input0_shape->GetDims())
  KERNEL_CHECK_FALSE((input1_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
                     "The original_image dims=[%d] must be 4-dimensional", input1_shape->GetDims())

  DataType input1_type = ctx.Input(1)->GetDataType();
  DataType input2_type = ctx.Input(2)->GetDataType();
  DataType input3_type = ctx.Input(3)->GetDataType();
  // dtypes check
  KERNEL_CHECK_FALSE((input1_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID, "The input1's dtype=[%d] must be DT_FLOAT",
                     DTypeStr(input1_type).c_str())
  KERNEL_CHECK_FALSE((input2_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID, "The input2's type=[%d] must be DT_FLOAT",
                     DTypeStr(input2_type).c_str())
  KERNEL_CHECK_FALSE((input3_type == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID, "The input3's dtype=[%d] must be DT_FLOAT",
                     DTypeStr(input3_type).c_str())

  KERNEL_LOG_INFO(
    "ScaleAndTranslateGradCpuKernel[%s], input0: size[%llu], input1: "
    "size[%llu];"
    "input2: size[%llu], input3: size[%llu], output: size[%llu].",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize(), ctx.Input(2)->GetDataSize(),
    ctx.Input(3)->GetDataSize(), ctx.Output(0)->GetDataSize());

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ScaleAndTranslateCpuKernel::ScaleAndTranslateCompute(CpuKernelContext &ctx) {
  auto input_size = reinterpret_cast<int32_t *>(ctx.Input(1)->GetData());
  auto input_scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());
  auto input_translation = reinterpret_cast<float *>(ctx.Input(3)->GetData());
  std::string kernel_type_str = ctx.GetAttr("kernel_type")->GetString();
  bool antialias_;
  antialias_ = ctx.GetAttr("antialias")->GetBool();
  SamplingKernelType kernel_type_ = SamplingKernelTypeFromString(kernel_type_str);

  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);

  auto input0_shape = ctx.Input(0)->GetTensorShape();
  auto input1_shape = ctx.Input(1)->GetTensorShape();

  const int64_t output_height = input_size[0];
  const int64_t output_width = input_size[1];

  const int64_t batch_size = input0_shape->GetDimSize(0);
  const int64_t input_height = input0_shape->GetDimSize(1);
  const int64_t input_width = input0_shape->GetDimSize(2);
  const int64_t channels = input0_shape->GetDimSize(3);

  KERNEL_CHECK_FALSE((output_height > 0 && output_width > 0), KERNEL_STATUS_PARAM_INVALID,
                     "output_height = [%d] and output_width = [%d] must be positive", output_height, output_width)
  KERNEL_CHECK_FALSE((channels > 0), KERNEL_STATUS_PARAM_INVALID, "image_channel = [%d] must have at least one",
                     channels)
  KERNEL_CHECK_FALSE((input_height > 0 && input_width > 0), KERNEL_STATUS_PARAM_INVALID,
                     "input_height = [%d] and input_width = [%d] must be of non-zero size", input_height, input_width)

  float row_scale, col_scale;
  row_scale = input_scale[0];
  col_scale = input_scale[1];

  KERNEL_CHECK_FALSE((row_scale > 0 && col_scale > 0), KERNEL_STATUS_PARAM_INVALID,
                     "row_scale = [%d] and col_scale = [%d] must be greater than zero.", row_scale, col_scale)

  float row_translation, col_translation;
  row_translation = input_translation[0];
  col_translation = input_translation[1];

  EigenTensor inputTensor(input, input->GetData());
  EigenTensor outputTensor(output, output->GetData());

  typename TTypes<T, 4>::Tensor image_data(inputTensor.tensor<T, 4>());

  typename TTypes<float, 4>::Tensor output_data(outputTensor.tensor<float, 4>());

  Spans col_spans;
  ComputeSpans(ctx, kernel_type_, output_width, input_width, col_scale, col_translation, antialias_, &col_spans);

  Spans row_spans;
  ComputeSpans(ctx, kernel_type_, output_height, input_height, row_scale, row_translation, antialias_, &row_spans);

  Eigen::Tensor<float, 4> intermediate_tensor_middle(batch_size, output_height, input_width, channels);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_data(intermediate_tensor_middle.data(),
                                                              intermediate_tensor_middle.dimensions());
  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());

  GatherSpans<T>()(ctx, row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts, col_weights,
                   image_data, intermediate_data, output_data);

  delete col_spans.starts;
  delete col_spans.weights;
  delete row_spans.starts;
  delete row_spans.weights;

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ScaleAndTranslateGradCpuKernel::ScaleAndTranslateGradCompute(CpuKernelContext &ctx) {
  auto input_scale = reinterpret_cast<float *>(ctx.Input(2)->GetData());
  auto input_translation = reinterpret_cast<float *>(ctx.Input(3)->GetData());
  std::string kernel_type_str = ctx.GetAttr("kernel_type")->GetString();
  bool antialias_;
  antialias_ = ctx.GetAttr("antialias")->GetBool();
  SamplingKernelType kernel_type_ = SamplingKernelTypeFromString(kernel_type_str);

  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  Tensor *original_image = ctx.Input(1);
  auto input0_shape = input->GetTensorShape();
  auto input1_shape = original_image->GetTensorShape();

  const int64_t batch_size = input0_shape->GetDimSize(0);
  const int64_t forward_input_height = input1_shape->GetDimSize(1);
  const int64_t forward_input_width = input1_shape->GetDimSize(2);
  const int64_t channels = input0_shape->GetDimSize(3);

  float row_scale, col_scale;
  row_scale = input_scale[0];
  col_scale = input_scale[1];

  KERNEL_CHECK_FALSE((row_scale > 0 && col_scale > 0), KERNEL_STATUS_PARAM_INVALID,
                     "row_scale = [%d] and col_scale = [%d] must be of non-zero size", row_scale, col_scale)

  float row_translation, col_translation;
  row_translation = input_translation[0];
  col_translation = input_translation[1];

  EigenTensor inputTensor(input, input->GetData());

  // output shape should be {batch_size, forward_input_height,forward_input_width, channels};
  EigenTensor outputTensor(output, output->GetData());
  TTypes<float, 4>::Tensor input_grad(inputTensor.tensor<float, 4>());
  typename TTypes<T, 4>::Tensor output_grad(outputTensor.tensor<T, 4>());

  const int64_t forward_output_height = input_grad.dimension(1);
  const int64_t forward_output_width = input_grad.dimension(2);

  Spans col_spans;
  ComputeGradSpans(ctx, kernel_type_, forward_output_width, forward_input_width, col_scale, col_translation, antialias_,
                   &col_spans);

  Spans row_spans;
  ComputeGradSpans(ctx, kernel_type_, forward_output_height, forward_input_height, row_scale, row_translation,
                   antialias_, &row_spans);

  Eigen::Tensor<float, 4> intermediate_tensor_middle(batch_size, forward_input_height, forward_output_width, channels);
  Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_data(intermediate_tensor_middle.data(),
                                                              intermediate_tensor_middle.dimensions());

  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());

  GatherSpans<T>()(ctx, row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts, col_weights,
                   input_grad, intermediate_data, output_grad);

  delete col_spans.starts;
  delete col_spans.weights;
  delete row_spans.starts;
  delete row_spans.weights;

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GatherColumns(CpuKernelContext &context, int span_size, const int32_t *starts, const float *weights,
                       const T *image, const int64_t input_height, const int64_t input_width,
                       const int64_t output_height, const int64_t output_width, const int channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto shard_column = [&](int start, int end) {
    for (int y = start; y < end; ++y) {
      const T *input_row_start = image + in_row_size * y;
      float *out_pix = output + out_row_size * y;
      for (int x = 0; x < output_width; ++x, out_pix += channels) {
        const T *in_pix = input_row_start + starts[x] * channels;
        const float *weights_start = weights + x * span_size;
        const int real_span_size = std::min(starts[x] + span_size, static_cast<int>(input_width)) - starts[x];
        const float *weights_end = weights_start + real_span_size;
        for (int c = 0; c < channels; ++c) {
          out_pix[c] = 0.0f;
        }
        for (const float *weight_ptr = weights_start; weight_ptr != weights_end; ++weight_ptr) {
          float w = *weight_ptr;
          for (int c = 0; c < channels; ++c) {
            out_pix[c] += w * static_cast<float>(in_pix[c]);
          }
          in_pix += channels;
        }
      }
    }
  };
  SWITCH_PARALLEL(shard_column, output_height, context);
  return KERNEL_STATUS_OK;
}

template <typename T>
inline void AddScaledVector(const T *in_vec, int vec_len, float weight, float *out_vec) {
  float *out_vec_end = out_vec + vec_len;
  for (; out_vec != out_vec_end; ++out_vec, ++in_vec) {
    *out_vec += weight * static_cast<float>(*in_vec);
  }
}

template <typename T>
uint32_t GatherRows(CpuKernelContext &context, int span_size, const int32_t *starts, const float *weights,
                    const T *image, const int64_t input_height, const int64_t input_width, const int64_t output_height,
                    const int64_t output_width, const int channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto shard_rows = [&](int start, int end) {
    for (int y = start; y < end; ++y) {
      float *out_row_data = output + out_row_size * y;
      std::fill(out_row_data, out_row_data + out_row_size, 0.0f);
      int in_row = starts[y];
      const T *in_row_data = image + in_row_size * in_row;
      const float *weights_start = weights + y * span_size;
      const int real_span_size = std::min(starts[y] + span_size, static_cast<int>(input_height)) - starts[y];
      const float *const weights_end = weights_start + real_span_size;

      for (const float *weight_it = weights_start; weight_it != weights_end; ++weight_it) {
        AddScaledVector(in_row_data, in_row_size, *weight_it, out_row_data);
        in_row_data += in_row_size;
      }
    }
  };
  SWITCH_PARALLEL(shard_rows, output_height, context);
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GatherSpans<T>::operator()(aicpu::CpuKernelContext &context, int row_span_size,
                                    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> row_starts,
                                    Eigen::TensorMap<Eigen::Tensor<float, 1>> row_weights, int col_span_size,
                                    Eigen::TensorMap<Eigen::Tensor<int32_t, 1>> col_starts,
                                    Eigen::TensorMap<Eigen::Tensor<float, 1>> col_weights,
                                    typename TTypes<T, 4>::Tensor images,
                                    Eigen::TensorMap<Eigen::Tensor<float, 4>> intermediate_buffer,
                                    typename TTypes<float, 4>::Tensor resized_images) {
  const int batch_size = images.dimension(0);
  const int64_t input_height = images.dimension(1);
  const int64_t input_width = images.dimension(2);
  const int channels = images.dimension(3);

  const int64_t output_height = resized_images.dimension(1);
  const int64_t output_width = resized_images.dimension(2);

  const int64_t input_pix_per_batch = input_width * input_height * channels;
  const int64_t intermediate_pix_per_batch = input_width * output_height * channels;
  const int64_t output_pix_per_batch = output_width * output_height * channels;
  float *intermediate_ptr = intermediate_buffer.data();

  const T *image_ptr = images.data();
  float *out_ptr = resized_images.data();

  auto row_start_data = row_starts.data();
  auto row_weights_data = row_weights.data();
  for (int b = 0; b < batch_size; ++b, image_ptr += input_pix_per_batch, intermediate_ptr += intermediate_pix_per_batch,
           out_ptr += output_pix_per_batch) {
    GatherRows(context, row_span_size, row_start_data, row_weights_data, image_ptr, input_height, input_width,
               output_height, input_width, channels, intermediate_ptr);
    GatherColumns(context, col_span_size, col_starts.data(), col_weights.data(), intermediate_ptr, output_height,
                  input_width, output_height, output_width, channels, out_ptr);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScaleAndTranslate, ScaleAndTranslateCpuKernel);
REGISTER_CPU_KERNEL(kScaleAndTranslateGrad, ScaleAndTranslateGradCpuKernel);
}  // namespace aicpu

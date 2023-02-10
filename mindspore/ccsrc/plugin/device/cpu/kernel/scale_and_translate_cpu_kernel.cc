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

#include "plugin/device/cpu/kernel/scale_and_translate_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include "mindspore/core/ops/scale_and_translate.h"
#include "mindspore/core/ops/grad/scale_and_translate_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kScaleAndTranslateInputsNum = 4;
constexpr size_t kScaleAndTranslateOutputsNum = 1;
constexpr size_t kScaleAndTranslateGradInputsNum = 4;
constexpr size_t kScaleAndTranslateGradOutputsNum = 1;
}  // namespace

bool ScaleAndTranslateCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ScaleAndTranslate>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ScaleAndTranslate ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  input1_shape_ = inputs[kIndex1]->GetShapeVector();
  input2_shape_ = inputs[kIndex2]->GetShapeVector();
  input3_shape_ = inputs[kIndex3]->GetShapeVector();
  input0_dtype_ = inputs[kIndex0]->GetDtype();
  kernel_type_ = kernel_ptr->get_kernel_type();
  antialias_ = kernel_ptr->get_antialias();
  size_t input0_dim = 4;
  std::vector<int64_t> valid_shape = {2};
  // dims check
  if (input0_shape_.size() != input0_dim) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[images]'s rank must be 4, but got "
                      << input0_shape_.size() << ".";
  }
  if (input1_shape_ != valid_shape) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[size]'s shape must be (2,), but got " << input1_shape_
                      << ".";
  }
  if (input2_shape_ != valid_shape) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[scale]'s shape must be (2,), but got " << input1_shape_
                      << ".";
  }
  if (input3_shape_ != valid_shape) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[translation]'s shape must be (2,), but got "
                      << input1_shape_ << ".";
  }
  switch (input0_dtype_) {
    case kNumberTypeInt8:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int8_t>;
      break;
    case kNumberTypeInt16:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int16_t>;
      break;
    case kNumberTypeInt32:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int32_t>;
      break;
    case kNumberTypeInt64:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<int64_t>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<float16>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat64:
      kernel_func_ = &ScaleAndTranslateCpuKernelMod::LaunchKernel<double>;
      break;
    default:
      MS_LOG(ERROR) << "ScaleAndTranslate kernel does not support " << TypeIdToString(input0_dtype_);
      return false;
  }
  return true;
}

bool ScaleAndTranslateGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ScaleAndTranslateGrad>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast ScaleAndTranslateGrad ops failed!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  input1_shape_ = inputs[kIndex1]->GetShapeVector();
  input2_shape_ = inputs[kIndex2]->GetShapeVector();
  input3_shape_ = inputs[kIndex3]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  kernel_type_ = kernel_ptr->get_kernel_type();
  antialias_ = kernel_ptr->get_antialias();
  size_t dim = 4;
  std::vector<int64_t> valid_shape = {2};
  // dims check
  if (input1_shape_.size() != dim) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[original_image]'s rank must be 4, but got "
                      << input1_shape_.size() << ".";
  }
  if (input2_shape_ != valid_shape) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[scale]'s shape must be (2,), but got " << input1_shape_
                      << ".";
  }
  if (input3_shape_ != valid_shape) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", the input[translation]'s shape must be (2,), but got "
                      << input1_shape_ << ".";
  }
  kernel_func_ = &ScaleAndTranslateGradCpuKernelMod::LaunchKernel<float>;
  return true;
}

template <typename T>
inline void AddScaledVector(const T *in_vec, int64_t vec_len, float weight, float *out_vec) {
  float *out_vec_end = out_vec + vec_len;
  for (; out_vec != out_vec_end; ++out_vec, ++in_vec) {
    *out_vec += weight * static_cast<float>(*in_vec);
  }
}

template <typename T>
void ScaleAndTranslateCpuKernelMod::GatherRows(int64_t span_size, const int64_t *starts, const float *weights,
                                               const T *image, const int64_t input_height, const int64_t input_width,
                                               const int64_t output_height, const int64_t output_width,
                                               const int64_t channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto task = [&span_size, &starts, &weights, &image, &input_height, &output, &in_row_size, &out_row_size](
                int64_t start, int64_t end) {
    for (int64_t y = start; y < end; ++y) {
      float *out_row_data = output + out_row_size * y;
      std::fill(out_row_data, out_row_data + out_row_size, 0.0f);
      int64_t in_row = starts[y];
      const T *in_row_data = image + in_row_size * in_row;
      const float *weights_start = weights + y * span_size;
      const int64_t real_span_size = std::min(starts[y] + span_size, input_height) - starts[y];
      const float *const weights_end = weights_start + real_span_size;
      for (const float *weight_it = weights_start; weight_it != weights_end; ++weight_it) {
        AddScaledVector(in_row_data, in_row_size, *weight_it, out_row_data);
        in_row_data += in_row_size;
      }
    }
  };
  ParallelLaunchAutoSearch(task, output_height, this, &parallel_search_info_);
}

template <typename T>
void ScaleAndTranslateCpuKernelMod::GatherColumns(int64_t span_size, const int64_t *starts, const float *weights,
                                                  const T *image, const int64_t input_height, const int64_t input_width,
                                                  const int64_t output_height, const int64_t output_width,
                                                  const int64_t channels, float *output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;
  auto task = [&span_size, &starts, &weights, &image, &input_height, &input_width, &output_width, &channels, &output,
               &in_row_size, &out_row_size](int64_t start, int64_t end) {
    for (int64_t y = start; y < end; ++y) {
      const T *input_row_start = image + in_row_size * y;
      float *out_pix = output + out_row_size * y;
      for (int64_t x = 0; x < output_width; ++x, out_pix += channels) {
        const T *in_pix = input_row_start + starts[x] * channels;
        const float *weights_start = weights + x * span_size;
        const int64_t real_span_size = std::min(starts[x] + span_size, input_width) - starts[x];
        const float *weights_end = weights_start + real_span_size;
        for (int64_t c = 0; c < channels; ++c) {
          out_pix[c] = 0.0f;
        }
        for (const float *weight_ptr = weights_start; weight_ptr != weights_end; ++weight_ptr) {
          float w = *weight_ptr;
          for (int64_t c = 0; c < channels; ++c) {
            out_pix[c] += w * static_cast<float>(in_pix[c]);
          }
          in_pix += channels;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, output_height, this, &parallel_search_info_);
}

template <typename T>
uint32_t ScaleAndTranslateCpuKernelMod::GatherSpans(
  int64_t row_span_size, Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts,
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights, int64_t col_span_size,
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts, Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights,
  typename TTypes<T, dim4>::Tensor images, Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_buffer,
  typename TTypes<float, dim4>::Tensor resized_images) {
  const int64_t batch_size = images.dimension(0);
  const int64_t input_height = images.dimension(1);
  const int64_t input_width = images.dimension(2);
  const int64_t channels = images.dimension(3);
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
  for (int64_t b = 0; b < batch_size; ++b, image_ptr += input_pix_per_batch,
               intermediate_ptr += intermediate_pix_per_batch, out_ptr += output_pix_per_batch) {
    GatherRows(row_span_size, row_start_data, row_weights_data, image_ptr, input_height, input_width, output_height,
               input_width, channels, intermediate_ptr);
    GatherColumns(col_span_size, col_starts.data(), col_weights.data(), intermediate_ptr, output_height, input_width,
                  output_height, output_width, channels, out_ptr);
  }
  return true;
}

template <typename T>
inline const T &Clamp(const T &low, const T &high, const T &value) {
  if (high < value) {
    return high;
  }
  if (value < low) {
    return low;
  }
  return value;
}

template <typename Kernel>
void ScaleAndTranslateCpuKernelMod::ComputeSpansCore(const Kernel &kernel, const int64_t output_size,
                                                     const int64_t input_size, const float scale, const float translate,
                                                     bool antialias, Spans *spans) {
  const float EPSINON = 0.00001;
  if ((scale >= -EPSINON) && (scale <= EPSINON)) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", divisor scale cannot be 0.";
  }
  const float inv_scale = 1.0 / scale;
  const float inv_translate = -inv_scale * translate;
  const float kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
  int64_t num = 2;
  spans->span_size = std::min(num * FloatToInt(std::ceil(kernel.Radius() * kernel_scale)) + 1, input_size);
  spans->starts = std::make_shared<Eigen::Tensor<int64_t, dim1>>(output_size);
  spans->weights = std::make_shared<Eigen::Tensor<float, dim1>>(spans->span_size * output_size);
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
  (void)weights_vec.setZero();
  const float one_over_kernel_scale = 1.0f / kernel_scale;
  int64_t max_span_size = 0;
  std::vector<float> temp_weights;
  for (auto x = 0; x < output_size; ++x) {
    const float col_f = x + 0.5f;
    const float sample_f = col_f * inv_scale + inv_translate;
    // Don't sample when the sampling location is outside the source image.
    if (sample_f < 0 || sample_f > input_size) {
      // Add an empty span.
      starts_vec(x) = 0;
      continue;
    }
    int64_t span_start = std::ceil(sample_f - kernel.Radius() * kernel_scale - 0.5f);
    int64_t span_end = std::floor(sample_f + kernel.Radius() * kernel_scale - 0.5f);
    span_start = Clamp(IntToLong(0), input_size - 1, span_start);
    span_end = Clamp(IntToLong(0), input_size - 1, span_end) + 1;
    const int64_t this_span_size = span_end - span_start;
    if (this_span_size > spans->span_size) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", span size cannot be larger than " << spans->span_size
                        << ", but got " << this_span_size << ".";
    }
    float total_weight_sum = 0.0f;
    temp_weights.clear();
    for (int64_t source = span_start; source < span_end; ++source) {
      float kernel_pos = LongToFloat(source) + 0.5f - sample_f;
      float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
      total_weight_sum += weight;
      temp_weights.push_back(weight);
    }
    max_span_size = std::max(max_span_size, this_span_size);
    if (std::abs(total_weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
      float one_over_total_weight_sum = 1.0f / total_weight_sum;
      int64_t out_index = spans->span_size * x;
      for (float weight : temp_weights) {
        weights_vec(out_index) = weight * one_over_total_weight_sum;
        ++out_index;
      }
    }
    starts_vec(x) = span_start;
  }
}

void ScaleAndTranslateGradCpuKernelMod::ComputeGradSpansCore(const Spans *spans, const int64_t forward_output_size,
                                                             const int64_t forward_input_size, Spans *grad_spans) {
  struct GradComponent {
    int64_t index;
    float weight;
  };
  std::vector<std::vector<GradComponent>> grad_components(forward_input_size);

  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> starts_vec(spans->starts->data(), spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> weights_vec(spans->weights->data(), spans->weights->dimensions());
  auto shard_grad_output = [&spans, &starts_vec, &weights_vec, &grad_components, &forward_input_size](int64_t start,
                                                                                                      int64_t end) {
    for (auto output_index = start; output_index < end; ++output_index) {
      int64_t input_index = starts_vec(output_index);
      for (int64_t j = 0; j < spans->span_size; ++j, ++input_index) {
        const float weight = weights_vec(output_index * spans->span_size + j);
        if (weight != static_cast<float>(0.0f) && input_index < forward_input_size) {
          grad_components[input_index].push_back(GradComponent{output_index, weight});
        }
      }
    }
  };
  ParallelLaunchAutoSearch(shard_grad_output, forward_output_size, this, &parallel_search_info_);
  int64_t max_size = 0;
  for (std::vector<GradComponent> &gc : grad_components) {
    if (!gc.empty()) {
      std::sort(gc.begin(), gc.end(),
                [](const GradComponent &x1, const GradComponent &x2) { return x1.index < x2.index; });
      max_size = std::max(gc.back().index - gc.front().index + 1, max_size);
    }
  }
  grad_spans->span_size = max_size;
  grad_spans->starts = std::make_shared<Eigen::Tensor<int64_t, dim1>>(forward_input_size);
  grad_spans->weights = std::make_shared<Eigen::Tensor<float, dim1>>(grad_spans->span_size * forward_input_size);

  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> grad_starts_vec(grad_spans->starts->data(),
                                                                 grad_spans->starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> grad_weights_vec(grad_spans->weights->data(),
                                                                grad_spans->weights->dimensions());
  (void)grad_weights_vec.setZero();
  auto shard_grad_input = [&grad_components, &grad_starts_vec, &grad_weights_vec, &grad_spans](int64_t start,
                                                                                               int64_t end) {
    for (int64_t input_index = start; input_index < end; ++input_index) {
      if (!grad_components[input_index].empty()) {
        const int64_t start_span = grad_components[input_index].front().index;
        grad_starts_vec(input_index) = start_span;
        for (const GradComponent &gc : grad_components[input_index]) {
          grad_weights_vec(input_index * grad_spans->span_size + gc.index - start_span) += gc.weight;
        }
      } else {
        grad_starts_vec(input_index) = 0;
      }
    }
  };
  ParallelLaunchAutoSearch(shard_grad_input, forward_input_size, this, &parallel_search_info_);
}

bool ScaleAndTranslateCpuKernelMod::ComputeSpans(const KernelType kernel_type, const int64_t output_size,
                                                 const int64_t input_size, const float scale, const float translate,
                                                 const bool antialias, Spans *spans, const std::string kernel_name) {
  switch (kernel_type) {
    case Lanczos1: {
      ComputeSpansCore(CreateLanczos1Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Lanczos3: {
      ComputeSpansCore(CreateLanczos3Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Lanczos5: {
      ComputeSpansCore(CreateLanczos5Kernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Gaussian: {
      ComputeSpansCore(CreateGaussianKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Box: {
      ComputeSpansCore(CreateBoxKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case Triangle: {
      ComputeSpansCore(CreateTriangleKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case KeysCubic: {
      ComputeSpansCore(CreateKeysCubicKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    case MitchellCubic: {
      ComputeSpansCore(CreateMitchellCubicKernel(), output_size, input_size, scale, translate, antialias, spans);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "For " << kernel_name << ", kernel_type kernel data type [" << kernel_type
                        << "] not support.";
      return false;
  }
  return true;
}

bool ScaleAndTranslateGradCpuKernelMod::ComputeGradSpans(const KernelType kernel_type,
                                                         const int64_t forward_output_size,
                                                         const int64_t forward_input_size, const float scale,
                                                         const float translate, const bool antialias, Spans *grad_spans,
                                                         const std::string kernel_name) {
  Spans spans;
  ScaleAndTranslateCpuKernelMod scale_and_translate_mod;
  (void)scale_and_translate_mod.ComputeSpans(kernel_type, forward_output_size, forward_input_size, scale, translate,
                                             antialias, &spans, kernel_name);
  ComputeGradSpansCore(&spans, forward_output_size, forward_input_size, grad_spans);
  return true;
}

template <typename T>
bool ScaleAndTranslateCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScaleAndTranslateInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScaleAndTranslateOutputsNum, kernel_name_);
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto input_size = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto input_scale = reinterpret_cast<float *>(inputs[2]->addr);
  auto input_translation = reinterpret_cast<float *>(inputs[3]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  KernelType sampling_kernel_type = KernelTypeFromString(kernel_type_);
  const int64_t output_height = IntToLong(input_size[0]);
  const int64_t output_width = IntToLong(input_size[1]);
  const int64_t batch_size = input0_shape_[0];
  const int64_t input_height = input0_shape_[1];
  const int64_t input_width = input0_shape_[2];
  const int64_t channels = input0_shape_[3];
  if (output_height <= 0 || output_width <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", output_height and output_width must be positive, but got "
                      << "output_height: " << output_height << " and output_width: " << output_width << ".";
  }
  if (channels <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", image_channel must have at least one, but got image_channel: " << channels << ".";
  }
  if (input_height <= 0 || input_width <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", input_height and input_width must be of non-zero size, but got "
                      << "input_height: " << input_height << " and input_width: " << input_width << ".";
  }
  float row_scale = input_scale[0];
  float col_scale = input_scale[1];
  if (row_scale <= 0 || col_scale <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", row_scale and col_scale must be greater than zero, but got "
                      << "row_scale: " << row_scale << " and col_scale: " << col_scale << ".";
  }
  float row_translation = input_translation[0];
  float col_translation = input_translation[1];
  EigenTensor inputTensor(input0_shape_, input);
  EigenTensor outputTensor(output_shape_, output);
  typename TTypes<T, dim4>::Tensor image_data(inputTensor.tensor<T, dim4>());

  typename TTypes<float, dim4>::Tensor output_data(outputTensor.tensor<float, dim4>());
  Spans col_spans;
  (void)ComputeSpans(sampling_kernel_type, output_width, input_width, col_scale, col_translation, antialias_,
                     &col_spans, kernel_name_);

  Spans row_spans;
  (void)ComputeSpans(sampling_kernel_type, output_height, input_height, row_scale, row_translation, antialias_,
                     &row_spans, kernel_name_);

  Eigen::Tensor<float, dim4> intermediate_tensor_middle(batch_size, output_height, input_width, channels);
  Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_data(intermediate_tensor_middle.data(),
                                                                 intermediate_tensor_middle.dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());
  GatherSpans<T>(row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts, col_weights, image_data,
                 intermediate_data, output_data);
  return true;
}

template <typename T>
bool ScaleAndTranslateGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kScaleAndTranslateGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kScaleAndTranslateGradOutputsNum, kernel_name_);
  auto input = reinterpret_cast<float *>(inputs[0]->addr);
  auto input_scale = reinterpret_cast<float *>(inputs[2]->addr);
  auto input_translation = reinterpret_cast<float *>(inputs[3]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  KernelType sampling_kernel_type = KernelTypeFromString(kernel_type_);

  const int64_t batch_size = input0_shape_[0];
  const int64_t forward_input_height = input1_shape_[1];
  const int64_t forward_input_width = input1_shape_[2];
  const int64_t channels = input0_shape_[3];
  float row_scale = input_scale[0];
  float col_scale = input_scale[1];
  if (row_scale <= 0 || col_scale <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", row_scale and col_scale must be greater than zero, but got "
                      << "row_scale: " << row_scale << " and col_scale: " << col_scale << ".";
    return false;
  }
  float row_translation = input_translation[0];
  float col_translation = input_translation[1];
  EigenTensor inputTensor(input0_shape_, input);
  // output shape should be {batch_size, forward_input_height,forward_input_width, channels};
  EigenTensor outputTensor(output_shape_, output);
  TTypes<float, dim4>::Tensor input_grad(inputTensor.tensor<float, dim4>());
  typename TTypes<T, dim4>::Tensor output_grad(outputTensor.tensor<T, dim4>());
  const int64_t forward_output_height = input_grad.dimension(1);
  const int64_t forward_output_width = input_grad.dimension(2);

  Spans col_spans;
  (void)ComputeGradSpans(sampling_kernel_type, forward_output_width, forward_input_width, col_scale, col_translation,
                         antialias_, &col_spans, kernel_name_);
  Spans row_spans;
  (void)ComputeGradSpans(sampling_kernel_type, forward_output_height, forward_input_height, row_scale, row_translation,
                         antialias_, &row_spans, kernel_name_);

  Eigen::Tensor<float, dim4> intermediate_tensor_middle(batch_size, forward_input_height, forward_output_width,
                                                        channels);
  Eigen::TensorMap<Eigen::Tensor<float, dim4>> intermediate_data(intermediate_tensor_middle.data(),
                                                                 intermediate_tensor_middle.dimensions());

  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> row_starts(row_spans.starts->data(), row_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> row_weights(row_spans.weights->data(), row_spans.weights->dimensions());
  Eigen::TensorMap<Eigen::Tensor<int64_t, dim1>> col_starts(col_spans.starts->data(), col_spans.starts->dimensions());
  Eigen::TensorMap<Eigen::Tensor<float, dim1>> col_weights(col_spans.weights->data(), col_spans.weights->dimensions());

  ScaleAndTranslateCpuKernelMod scale_and_translate_mod;
  scale_and_translate_mod.GatherSpans<T>(row_spans.span_size, row_starts, row_weights, col_spans.span_size, col_starts,
                                         col_weights, input_grad, intermediate_data, output_grad);
  return true;
}

int ScaleAndTranslateCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  return 0;
}

int ScaleAndTranslateGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    return ret;
  }
  input0_shape_ = inputs[kIndex0]->GetShapeVector();
  return 0;
}

std::vector<KernelAttr> ScaleAndTranslateCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeInt8)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt16)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeInt64)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat16)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32),
                                          KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat64)
                                            .AddInputAttr(kNumberTypeInt32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

std::vector<KernelAttr> ScaleAndTranslateGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {KernelAttr()
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddInputAttr(kNumberTypeFloat32)
                                            .AddOutputAttr(kNumberTypeFloat32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScaleAndTranslate, ScaleAndTranslateCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ScaleAndTranslateGrad, ScaleAndTranslateGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

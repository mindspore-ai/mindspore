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

#include "minddata/dataset/audio/kernels/audio_utils.h"

#include <complex>

#include "mindspore/core/base/float16.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
/// \brief Generate linearly spaced vector.
/// \param[in] start - Value of the startpoint.
/// \param[in] end - Value of the endpoint.
/// \param[in] n - N points in the output tensor.
/// \param[out] output - Tensor has n points with linearly space. The spacing between the points is (end-start)/(n-1).
/// \return Status return code.
template <typename T>
Status Linspace(std::shared_ptr<Tensor> *output, T start, T end, int n) {
  if (start > end) {
    std::string err = "Linspace: input param end must be greater than start.";
    RETURN_STATUS_UNEXPECTED(err);
  }
  n = std::isnan(n) ? 100 : n;
  TensorShape out_shape({n});
  std::vector<T> linear_vect(n);
  T interval = (n == 1) ? 0 : ((end - start) / (n - 1));
  for (int i = 0; i < linear_vect.size(); ++i) {
    linear_vect[i] = start + i * interval;
  }
  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(linear_vect, out_shape, &out_t));
  linear_vect.clear();
  linear_vect.shrink_to_fit();
  *output = out_t;
  return Status::OK();
}

/// \brief Calculate complex tensor angle.
/// \param[in] input - Input tensor, must be complex, <channel, freq, time, complex=2>.
/// \param[out] output - Complex tensor angle.
/// \return Status return code.
template <typename T>
Status ComplexAngle(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check complex
  if (!input->IsComplex()) {
    std::string err_msg = "ComplexAngle: input tensor is not in shape of <..., 2>.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  TensorShape input_shape = input->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2]});
  std::vector<T> phase(input_shape[0] * input_shape[1] * input_shape[2]);
  int ind = 0;

  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
    auto x = (*itr);
    itr++;
    auto y = (*itr);
    phase[ind] = atan2(y, x);
  }

  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(phase, out_shape, &out_t));
  phase.clear();
  phase.shrink_to_fit();
  *output = out_t;
  return Status::OK();
}

/// \brief Calculate complex tensor abs.
/// \param[in] input - Input tensor, must be complex, <channel, freq, time, complex=2>.
/// \param[out] output - Complex tensor abs.
/// \return Status return code.
template <typename T>
Status ComplexAbs(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check complex
  if (!input->IsComplex()) {
    std::string err_msg = "ComplexAngle: input tensor is not in shape of <..., 2>.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  TensorShape input_shape = input->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2]});
  std::vector<T> abs(input_shape[0] * input_shape[1] * input_shape[2]);
  int ind = 0;
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
    T x = (*itr);
    itr++;
    T y = (*itr);
    abs[ind] = sqrt(pow(y, 2) + pow(x, 2));
  }

  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(abs, out_shape, &out_t));
  *output = out_t;
  return Status::OK();
}

/// \brief Reconstruct complex tensor from norm and angle.
/// \param[in] abs - The absolute value of the complex tensor.
/// \param[in] angle - The angle of the complex tensor.
/// \param[out] output - Complex tensor, <channel, freq, time, complex=2>.
/// \return Status return code.
template <typename T>
Status Polar(const std::shared_ptr<Tensor> &abs, const std::shared_ptr<Tensor> &angle,
             std::shared_ptr<Tensor> *output) {
  // check shape
  if (abs->shape() != angle->shape()) {
    std::string err_msg = "Polar: input tensor shape of abs and angle must be the same.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  TensorShape input_shape = abs->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2], 2});
  std::vector<T> complex_vec(input_shape[0] * input_shape[1] * input_shape[2] * 2);
  int ind = 0;
  auto itr_abs = abs->begin<T>();
  auto itr_angle = angle->begin<T>();

  for (; itr_abs != abs->end<T>(); itr_abs++, itr_angle++) {
    complex_vec[ind++] = cos(*itr_angle) * (*itr_abs);
    complex_vec[ind++] = sin(*itr_angle) * (*itr_abs);
  }

  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(complex_vec, out_shape, &out_t));
  *output = out_t;
  return Status::OK();
}

/// \brief Pad complex tensor.
/// \param[in] input - The complex tensor.
/// \param[in] length - The length of padding.
/// \param[in] dim - The dim index for padding.
/// \param[out] output - Complex tensor, <channel, freq, time, complex=2>.
/// \return Status return code.
template <typename T>
Status PadComplexTensor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int length, int dim) {
  TensorShape input_shape = input->shape();
  std::vector<int64_t> pad_shape_vec = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
  pad_shape_vec[dim] += length;
  TensorShape input_shape_with_pad(pad_shape_vec);
  std::vector<T> in_vect(input_shape_with_pad[0] * input_shape_with_pad[1] * input_shape_with_pad[2] *
                         input_shape_with_pad[3]);
  auto itr_input = input->begin<T>();
  int input_cnt = 0;
  for (int ind = 0; ind < in_vect.size(); ind++) {
    in_vect[ind] = (*itr_input);
    input_cnt = (input_cnt + 1) % (input_shape[2] * input_shape[3]);
    itr_input++;
    // complex tensor last dim equals 2, fill zero count equals 2*width
    if (input_cnt == 0 && ind != 0) {
      for (int c = 0; c < length * 2; c++) {
        in_vect[++ind] = 0.0f;
      }
    }
  }
  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(in_vect, input_shape_with_pad, &out_t));
  *output = out_t;
  return Status::OK();
}

/// \brief Calculate phase.
/// \param[in] angle_0 - The angle.
/// \param[in] angle_1 - The angle.
/// \param[in] phase_advance - The phase advance.
/// \param[in] phase_time0 - The phase at time 0.
/// \param[out] output - Phase tensor.
/// \return Status return code.
template <typename T>
Status Phase(const std::shared_ptr<Tensor> &angle_0, const std::shared_ptr<Tensor> &angle_1,
             const std::shared_ptr<Tensor> &phase_advance, const std::shared_ptr<Tensor> &phase_time0,
             std::shared_ptr<Tensor> *output) {
  TensorShape phase_shape = angle_0->shape();
  std::vector<T> phase(phase_shape[0] * phase_shape[1] * phase_shape[2]);
  auto itr_angle_0 = angle_0->begin<T>();
  auto itr_angle_1 = angle_1->begin<T>();
  auto itr_pa = phase_advance->begin<T>();
  for (int ind = 0, input_cnt = 0; itr_angle_0 != angle_0->end<T>(); itr_angle_0++, itr_angle_1++, ind++) {
    if (ind != 0 && ind % phase_shape[2] == 0) {
      itr_pa++;
      if (itr_pa == phase_advance->end<T>()) {
        itr_pa = phase_advance->begin<T>();
      }
      input_cnt++;
    }
    phase[ind] = (*itr_angle_1) - (*itr_angle_0) - (*itr_pa);
    phase[ind] = phase[ind] - 2 * PI * round(phase[ind] / (2 * PI)) + (*itr_pa);
  }

  // concat phase time 0
  int ind = 0;
  auto itr_p0 = phase_time0->begin<T>();
  phase.insert(phase.begin(), (*itr_p0));
  while (itr_p0 != phase_time0->end<T>()) {
    itr_p0++;
    ind += phase_shape[2];
    phase[ind] = (*itr_p0);
  }
  phase.erase(phase.begin() + static_cast<int>(angle_0->Size()), phase.end());

  // cal phase accum
  for (ind = 0; ind < phase.size(); ind++) {
    if (ind % phase_shape[2] != 0) {
      phase[ind] = phase[ind] + phase[ind - 1];
    }
  }
  std::shared_ptr<Tensor> phase_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(phase, phase_shape, &phase_tensor));
  *output = phase_tensor;
  return Status::OK();
}

/// \brief Calculate magnitude.
/// \param[in] alphas - The alphas.
/// \param[in] abs_0 - The norm.
/// \param[in] abs_1 - The norm.
/// \param[out] output - Magnitude tensor.
/// \return Status return code.
template <typename T>
Status Mag(const std::shared_ptr<Tensor> &abs_0, const std::shared_ptr<Tensor> &abs_1, std::shared_ptr<Tensor> *output,
           const std::vector<T> &alphas) {
  TensorShape mag_shape = abs_0->shape();
  std::vector<T> mag(mag_shape[0] * mag_shape[1] * mag_shape[2]);
  auto itr_abs_0 = abs_0->begin<T>();
  auto itr_abs_1 = abs_1->begin<T>();
  for (int ind = 0; itr_abs_0 != abs_0->end<T>(); itr_abs_0++, itr_abs_1++, ind++) {
    mag[ind] = alphas[ind % mag_shape[2]] * (*itr_abs_1) + (1 - alphas[ind % mag_shape[2]]) * (*itr_abs_0);
  }
  std::shared_ptr<Tensor> mag_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(mag, mag_shape, &mag_tensor));
  *output = mag_tensor;
  return Status::OK();
}

template <typename T>
Status TimeStretch(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, float rate,
                   std::shared_ptr<Tensor> phase_advance) {
  // pack <..., freq, time, complex>
  TensorShape input_shape = input->shape();
  TensorShape toShape({input->Size() / (input_shape[-1] * input_shape[-2] * input_shape[-3]), input_shape[-3],
                       input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(toShape));
  if (rate == 1.0) {
    *output = input;
    return Status::OK();
  }
  // calculate time step and alphas
  int ind = 0;
  std::vector<dsize_t> time_steps_0, time_steps_1;
  std::vector<T> alphas;
  for (T val = 0;; ind++) {
    val = ind * rate;
    if (val >= input_shape[-2]) break;
    int val_int = static_cast<int>(val);
    time_steps_0.push_back(val_int);
    time_steps_1.push_back(val_int + 1);
    alphas.push_back(fmod(val, 1));
  }

  // calculate phase on time 0
  std::shared_ptr<Tensor> spec_time0, phase_time0;
  RETURN_IF_NOT_OK(
    input->Slice(&spec_time0, std::vector<SliceOption>({SliceOption(true), SliceOption(true),
                                                        SliceOption(std::vector<dsize_t>{0}), SliceOption(true)})));
  RETURN_IF_NOT_OK(ComplexAngle<T>(spec_time0, &phase_time0));

  // time pad: add zero to time dim
  RETURN_IF_NOT_OK(PadComplexTensor<T>(input, &input, 2, 2));

  // slice
  std::shared_ptr<Tensor> spec_0;
  RETURN_IF_NOT_OK(input->Slice(&spec_0, std::vector<SliceOption>({SliceOption(true), SliceOption(true),
                                                                   SliceOption(time_steps_0), SliceOption(true)})));
  std::shared_ptr<Tensor> spec_1;
  RETURN_IF_NOT_OK(input->Slice(&spec_1, std::vector<SliceOption>({SliceOption(true), SliceOption(true),
                                                                   SliceOption(time_steps_1), SliceOption(true)})));

  // new slices angle and abs <channel, freq, time>
  std::shared_ptr<Tensor> angle_0, angle_1, abs_0, abs_1;
  RETURN_IF_NOT_OK(ComplexAngle<T>(spec_0, &angle_0));
  RETURN_IF_NOT_OK(ComplexAbs<T>(spec_0, &abs_0));
  RETURN_IF_NOT_OK(ComplexAngle<T>(spec_1, &angle_1));
  RETURN_IF_NOT_OK(ComplexAbs<T>(spec_1, &abs_1));

  // cal phase, there exists precision loss between mindspore and pytorch
  std::shared_ptr<Tensor> phase_tensor;
  RETURN_IF_NOT_OK(Phase<T>(angle_0, angle_1, phase_advance, phase_time0, &phase_tensor));

  // calculate magnitude
  std::shared_ptr<Tensor> mag_tensor;
  RETURN_IF_NOT_OK(Mag<T>(abs_0, abs_1, &mag_tensor, alphas));

  // reconstruct complex from norm and angle
  std::shared_ptr<Tensor> complex_spec_stretch;
  RETURN_IF_NOT_OK(Polar<T>(mag_tensor, phase_tensor, &complex_spec_stretch));

  // unpack
  auto output_shape_vec = input_shape.AsVector();
  output_shape_vec.pop_back();
  output_shape_vec.pop_back();
  output_shape_vec.push_back(complex_spec_stretch->shape()[-2]);
  output_shape_vec.push_back(input_shape[-1]);
  RETURN_IF_NOT_OK(complex_spec_stretch->Reshape(TensorShape(output_shape_vec)));
  *output = complex_spec_stretch;
  return Status::OK();
}

Status TimeStretch(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, float rate, float hop_length,
                   float n_freq) {
  std::shared_ptr<Tensor> phase_advance;
  switch (input->type().value()) {
    case DataType::DE_FLOAT32:
      RETURN_IF_NOT_OK(Linspace<float>(&phase_advance, 0, PI * hop_length, n_freq));
      RETURN_IF_NOT_OK(TimeStretch<float>(input, output, rate, phase_advance));
      break;
    case DataType::DE_FLOAT64:
      RETURN_IF_NOT_OK(Linspace<double>(&phase_advance, 0, PI * hop_length, n_freq));
      RETURN_IF_NOT_OK(TimeStretch<double>(input, output, rate, phase_advance));
      break;
    default:
      RETURN_STATUS_UNEXPECTED("TimeStretch: input tensor type should be float or double, but got: " +
                               input->type().ToString());
  }
  return Status::OK();
}

Status RandomMaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_param,
                           float mask_value, int axis, std::mt19937 rnd) {
  std::uniform_int_distribution<int32_t> mask_width_value(0, mask_param);
  TensorShape input_shape = input->shape();
  int32_t mask_dim_size = axis == 1 ? input_shape[-2] : input_shape[-1];
  int32_t mask_width = mask_width_value(rnd);
  std::uniform_int_distribution<int32_t> min_freq_value(0, mask_dim_size - mask_width);
  int32_t mask_start = min_freq_value(rnd);

  return MaskAlongAxis(input, output, mask_width, mask_start, mask_value, axis);
}

Status MaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_width,
                     int32_t mask_start, float mask_value, int32_t axis) {
  if (axis != 2 && axis != 1) {
    RETURN_STATUS_UNEXPECTED("MaskAlongAxis: only support Time and Frequency masking, axis should be 1 or 2.");
  }
  TensorShape input_shape = input->shape();
  // squeeze input
  TensorShape squeeze_shape = TensorShape({-1, input_shape[-2], input_shape[-1]});
  input->Reshape(squeeze_shape);

  int check_dim_ind = (axis == 1) ? -2 : -1;
  CHECK_FAIL_RETURN_UNEXPECTED(0 <= mask_start && mask_start <= input_shape[check_dim_ind],
                               "MaskAlongAxis: mask_start should be less than the length of chosen dimension.");
  CHECK_FAIL_RETURN_UNEXPECTED(mask_start + mask_width <= input_shape[check_dim_ind],
                               "MaskAlongAxis: the sum of mask_start and mask_width is out of bounds.");

  int32_t cell_size = input->type().SizeInBytes();

  if (axis == 1) {
    // freq
    for (int ind = 0; ind < input->Size() / input_shape[-2] * mask_width; ind++) {
      int block_num = ind / (mask_width * input_shape[-1]);
      auto start_pos = ind % (mask_width * input_shape[-1]) + mask_start * input_shape[-1] +
                       input_shape[-1] * input_shape[-2] * block_num;
      auto start_mem_pos = const_cast<uchar *>(input->GetBuffer() + start_pos * cell_size);
      if (input->type() != DataType::DE_FLOAT64) {
        // tensor float 32
        auto mask_val = static_cast<float>(mask_value);
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_val, cell_size) == 0,
                                     "MaskAlongAxis: mask failed, memory copy error.");
      } else {
        // tensor float 64
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_value, cell_size) == 0,
                                     "MaskAlongAxis: mask failed, memory copy error.");
      }
    }
  } else {
    // time
    for (int ind = 0; ind < input->Size() / input_shape[-1] * mask_width; ind++) {
      int row_num = ind / mask_width;
      auto start_pos = ind % mask_width + mask_start + input_shape[-1] * row_num;
      auto start_mem_pos = const_cast<uchar *>(input->GetBuffer() + start_pos * cell_size);
      if (input->type() != DataType::DE_FLOAT64) {
        // tensor float 32
        auto mask_val = static_cast<float>(mask_value);
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_val, cell_size) == 0,
                                     "MaskAlongAxis: mask failed, memory copy error.");
      } else {
        // tensor float 64
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_value, cell_size) == 0,
                                     "MaskAlongAxis: mask failed, memory copy error.");
      }
    }
  }
  // unsqueeze input
  input->Reshape(input_shape);
  *output = input;
  return Status::OK();
}

template <typename T>
Status Norm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power) {
  // calculate the output dimension
  auto input_size = input->shape().AsVector();
  int32_t dim_back = input_size.back();
  CHECK_FAIL_RETURN_UNEXPECTED(
    dim_back == 2, "ComplexNorm: expect complex input of shape <..., 2>, but got: " + std::to_string(dim_back));
  input_size.pop_back();
  TensorShape out_shape = TensorShape(input_size);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(out_shape, input->type(), output));

  // calculate norm, using: .pow(2.).sum(-1).pow(0.5 * power)
  auto itr_out = (*output)->begin<T>();
  auto itr_in = input->begin<T>();

  for (; itr_out != (*output)->end<T>(); ++itr_out) {
    auto a = static_cast<T>(*itr_in);
    ++itr_in;
    auto b = static_cast<T>(*itr_in);
    ++itr_in;
    auto res = pow(a, 2) + pow(b, 2);
    *itr_out = static_cast<T>(pow(res, (0.5 * power)));
  }

  return Status::OK();
}

Status ComplexNorm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power) {
  try {
    if (input->type().value() >= DataType::DE_INT8 && input->type().value() <= DataType::DE_FLOAT16) {
      // convert the data type to float
      std::shared_ptr<Tensor> input_tensor;
      RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));

      Norm<float>(input_tensor, output, power);
    } else if (input->type().value() == DataType::DE_FLOAT32) {
      Norm<float>(input, output, power);
    } else if (input->type().value() == DataType::DE_FLOAT64) {
      Norm<double>(input, output, power);
    } else {
      RETURN_STATUS_UNEXPECTED("ComplexNorm: input tensor type should be int, float or double, but got: " +
                               input->type().ToString());
    }
    return Status::OK();
  } catch (std::runtime_error &e) {
    RETURN_STATUS_UNEXPECTED("ComplexNorm: " + std::string(e.what()));
  }
}

template <typename T>
float sgn(T val) {
  return (static_cast<T>(0) < val) - (val < static_cast<T>(0));
}

template <typename T>
Status Decoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T mu) {
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto itr_out = (*output)->begin<T>();
  auto itr = input->begin<T>();
  auto end = input->end<T>();

  while (itr != end) {
    auto x_mu = *itr;
    CHECK_FAIL_RETURN_SYNTAX_ERROR(mu != 0, "mu can not be zero.");
    x_mu = ((x_mu) / mu) * 2 - 1.0;
    x_mu = sgn(x_mu) * expm1(fabs(x_mu) * log1p(mu)) / mu;
    *itr_out = x_mu;
    ++itr_out;
    ++itr;
  }
  return Status::OK();
}

Status MuLawDecoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int quantization_channels) {
  if (input->type().value() >= DataType::DE_INT8 && input->type().value() <= DataType::DE_FLOAT32) {
    float f_mu = static_cast<float>(quantization_channels) - 1;

    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));

    RETURN_IF_NOT_OK(Decoding<float>(input_tensor, output, f_mu));
  } else if (input->type().value() == DataType::DE_FLOAT64) {
    double f_mu = static_cast<double>(quantization_channels) - 1;

    RETURN_IF_NOT_OK(Decoding<double>(input, output, f_mu));
  } else {
    RETURN_STATUS_UNEXPECTED("MuLawDecoding: input tensor type should be int, float or double, but got: " +
                             input->type().ToString());
  }
  return Status::OK();
}

template <typename T>
Status FadeIn(std::shared_ptr<Tensor> *output, int32_t fade_in_len, FadeShape fade_shape) {
  T start = 0;
  T end = 1;
  RETURN_IF_NOT_OK(Linspace<T>(output, start, end, fade_in_len));
  for (auto iter = (*output)->begin<T>(); iter != (*output)->end<T>(); iter++) {
    switch (fade_shape) {
      case FadeShape::kLinear:
        break;
      case FadeShape::kExponential:
        // Compute the scale factor of the exponential function, pow(2.0, *in_ter - 1.0) * (*in_ter)
        *iter = static_cast<T>(std::pow(2.0, *iter - 1.0) * (*iter));
        break;
      case FadeShape::kLogarithmic:
        // Compute the scale factor of the logarithmic function, log(*in_iter + 0.1) + 1.0
        *iter = static_cast<T>(std::log10(*iter + 0.1) + 1.0);
        break;
      case FadeShape::kQuarterSine:
        // Compute the scale factor of the quarter_sine function, sin((*in_iter - 1.0) * PI / 2.0)
        *iter = static_cast<T>(std::sin((*iter) * PI / 2.0));
        break;
      case FadeShape::kHalfSine:
        // Compute the scale factor of the half_sine function, sin((*in_iter) * PI - PI / 2.0) / 2.0 + 0.5
        *iter = static_cast<T>(std::sin((*iter) * PI - PI / 2.0) / 2.0 + 0.5);
        break;
    }
  }
  return Status::OK();
}

template <typename T>
Status FadeOut(std::shared_ptr<Tensor> *output, int32_t fade_out_len, FadeShape fade_shape) {
  T start = 0;
  T end = 1;
  RETURN_IF_NOT_OK(Linspace<T>(output, start, end, fade_out_len));
  for (auto iter = (*output)->begin<T>(); iter != (*output)->end<T>(); iter++) {
    switch (fade_shape) {
      case FadeShape::kLinear:
        // In fade out, invert *out_iter
        *iter = static_cast<T>(1.0 - *iter);
        break;
      case FadeShape::kExponential:
        // Compute the scale factor of the exponential function
        *iter = static_cast<T>(std::pow(2.0, -*iter) * (1.0 - *iter));
        break;
      case FadeShape::kLogarithmic:
        // Compute the scale factor of the logarithmic function
        *iter = static_cast<T>(std::log10(1.1 - *iter) + 1.0);
        break;
      case FadeShape::kQuarterSine:
        // Compute the scale factor of the quarter_sine function
        *iter = static_cast<T>(std::sin((*iter) * PI / 2.0 + PI / 2.0));
        break;
      case FadeShape::kHalfSine:
        // Compute the scale factor of the half_sine function
        *iter = static_cast<T>(std::sin((*iter) * PI + PI / 2.0) / 2.0 + 0.5);
        break;
    }
  }
  return Status::OK();
}

template <typename T>
Status Fade(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t fade_in_len,
            int32_t fade_out_len, FadeShape fade_shape) {
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(input, output));
  const TensorShape input_shape = input->shape();
  int32_t waveform_length = static_cast<int32_t>(input_shape[-1]);
  CHECK_FAIL_RETURN_UNEXPECTED(fade_in_len <= waveform_length, "Fade: fade_in_len exceeds waveform length.");
  CHECK_FAIL_RETURN_UNEXPECTED(fade_out_len <= waveform_length, "Fade: fade_out_len exceeds waveform length.");
  int32_t num_waveform = static_cast<int32_t>(input->Size() / waveform_length);
  TensorShape toShape = TensorShape({num_waveform, waveform_length});
  RETURN_IF_NOT_OK((*output)->Reshape(toShape));
  TensorPtr fade_in;
  RETURN_IF_NOT_OK(FadeIn<T>(&fade_in, fade_in_len, fade_shape));
  TensorPtr fade_out;
  RETURN_IF_NOT_OK(FadeOut<T>(&fade_out, fade_out_len, fade_shape));

  // Add fade in to input tensor
  auto output_iter = (*output)->begin<T>();
  for (auto fade_in_iter = fade_in->begin<T>(); fade_in_iter != fade_in->end<T>(); fade_in_iter++) {
    *output_iter = (*output_iter) * (*fade_in_iter);
    for (int32_t j = 1; j < num_waveform; j++) {
      output_iter += waveform_length;
      *output_iter = (*output_iter) * (*fade_in_iter);
    }
    output_iter -= ((num_waveform - 1) * waveform_length);
    ++output_iter;
  }

  // Add fade out to input tensor
  output_iter = (*output)->begin<T>();
  output_iter += (waveform_length - fade_out_len);
  for (auto fade_out_iter = fade_out->begin<T>(); fade_out_iter != fade_out->end<T>(); fade_out_iter++) {
    *output_iter = (*output_iter) * (*fade_out_iter);
    for (int32_t j = 1; j < num_waveform; j++) {
      output_iter += waveform_length;
      *output_iter = (*output_iter) * (*fade_out_iter);
    }
    output_iter -= ((num_waveform - 1) * waveform_length);
    ++output_iter;
  }
  (*output)->Reshape(input_shape);
  return Status::OK();
}

Status Fade(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t fade_in_len,
            int32_t fade_out_len, FadeShape fade_shape) {
  if (DataType::DE_INT8 <= input->type().value() && input->type().value() <= DataType::DE_FLOAT32) {
    std::shared_ptr<Tensor> waveform;
    RETURN_IF_NOT_OK(TypeCast(input, &waveform, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Fade<float>(waveform, output, fade_in_len, fade_out_len, fade_shape));
  } else if (input->type().value() == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(Fade<double>(input, output, fade_in_len, fade_out_len, fade_shape));
  } else {
    RETURN_STATUS_UNEXPECTED("Fade: input tensor type should be int, float or double, but got: " +
                             input->type().ToString());
  }
  return Status::OK();
}

Status Magphase(const TensorRow &input, TensorRow *output, float power) {
  std::shared_ptr<Tensor> mag;
  std::shared_ptr<Tensor> phase;

  RETURN_IF_NOT_OK(ComplexNorm(input[0], &mag, power));
  if (input[0]->type() == DataType(DataType::DE_FLOAT64)) {
    RETURN_IF_NOT_OK(Angle<double>(input[0], &phase));
  } else {
    std::shared_ptr<Tensor> tmp;
    RETURN_IF_NOT_OK(TypeCast(input[0], &tmp, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Angle<float>(tmp, &phase));
  }
  (*output).push_back(mag);
  (*output).push_back(phase);

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore

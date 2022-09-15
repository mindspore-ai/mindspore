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

#include <fstream>

#include "mindspore/core/base/float16.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/util/random.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
/// \brief Calculate complex tensor angle.
/// \param[in] input - Input tensor, must be complex, <channel, freq, time, complex=2>.
/// \param[out] output - Complex tensor angle.
/// \return Status return code.
template <typename T>
Status ComplexAngle(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  // check complex
  RETURN_IF_NOT_OK(ValidateTensorShape("ComplexAngle", input->IsComplex(), "<..., complex=2>"));
  TensorShape input_shape = input->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2]});
  std::vector<T> phase(input_shape[0] * input_shape[1] * input_shape[2]);
  size_t ind = 0;

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
  RETURN_IF_NOT_OK(ValidateTensorShape("ComplexAngle", input->IsComplex(), "<..., complex=2>"));
  TensorShape input_shape = input->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2]});
  std::vector<T> abs(input_shape[0] * input_shape[1] * input_shape[2]);
  size_t ind = 0;
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

Status CreateLinearFbanks(std::shared_ptr<Tensor> *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                          int32_t sample_rate) {
  // all_freqs is equivalent filterbank construction.
  std::shared_ptr<Tensor> all_freqs;
  // the sampling frequency is at least twice the highest frequency of the signal.
  const double signal_times = 2;
  RETURN_IF_NOT_OK(Linspace<float>(&all_freqs, 0, sample_rate / signal_times, n_freqs));
  // f_pts is mel value sequence in linspace of (m_min, m_max).
  std::shared_ptr<Tensor> f_pts;
  const int32_t bias = 2;
  RETURN_IF_NOT_OK(Linspace<float>(&f_pts, f_min, f_max, n_filter + bias));
  // create filterbank
  std::shared_ptr<Tensor> fb;
  RETURN_IF_NOT_OK(CreateTriangularFilterbank<float>(&fb, all_freqs, f_pts));
  *output = fb;
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
    std::string err_msg = "Polar: the shape of input tensor abs and angle should be the same, but got: abs " +
                          abs->shape().ToString() + " and angle " + angle->shape().ToString();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  TensorShape input_shape = abs->shape();
  TensorShape out_shape({input_shape[0], input_shape[1], input_shape[2], 2});
  std::vector<T> complex_vec(input_shape[0] * input_shape[1] * input_shape[2] * 2);
  size_t ind = 0;
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
Status PadComplexTensor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int length, size_t dim) {
  TensorShape input_shape = input->shape();
  std::vector<int64_t> pad_shape_vec = {input_shape[0], input_shape[1], input_shape[2], input_shape[3]};
  pad_shape_vec[dim] += static_cast<int64_t>(length);
  TensorShape input_shape_with_pad(pad_shape_vec);
  std::vector<T> in_vect(input_shape_with_pad[0] * input_shape_with_pad[1] * input_shape_with_pad[2] *
                         input_shape_with_pad[3]);
  auto itr_input = input->begin<T>();
  int64_t input_cnt = 0;
  /*lint -e{446} ind is modified in the body of the for loop */
  for (auto ind = 0; ind < static_cast<int>(in_vect.size()); ind++) {
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
  size_t ind = 0;
  auto itr_p0 = phase_time0->begin<T>();
  (void)phase.insert(phase.begin(), (*itr_p0));
  itr_p0++;
  while (itr_p0 != phase_time0->end<T>()) {
    ind += phase_shape[2];
    phase[ind] = (*itr_p0);
    itr_p0++;
  }
  (void)phase.erase(phase.begin() + static_cast<int>(angle_0->Size()), phase.end());

  // cal phase accum
  for (ind = 0; ind < static_cast<int64_t>(phase.size()); ind++) {
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
  for (auto ind = 0; itr_abs_0 != abs_0->end<T>(); itr_abs_0++, itr_abs_1++, ind++) {
    mag[ind] = alphas[ind % mag_shape[2]] * (*itr_abs_1) + (1 - alphas[ind % mag_shape[2]]) * (*itr_abs_0);
  }
  std::shared_ptr<Tensor> mag_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(mag, mag_shape, &mag_tensor));
  *output = mag_tensor;
  return Status::OK();
}

template <typename T>
Status TimeStretch(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, float rate,
                   const std::shared_ptr<Tensor> &phase_advance) {
  // pack <..., freq, time, complex>
  TensorShape input_shape = input->shape();
  TensorShape toShape({input->Size() / (input_shape[-1] * input_shape[-2] * input_shape[-3]), input_shape[-3],
                       input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(toShape));
  if (std::fabs(rate - 1.0) <= std::numeric_limits<float>::epsilon()) {
    *output = input;
    return Status::OK();
  }
  // calculate time step and alphas
  std::vector<dsize_t> time_steps_0, time_steps_1;
  std::vector<T> alphas;
  for (int ind = 0;; ind++) {
    T val = static_cast<float>(ind) * rate;
    if (val >= input_shape[-2]) {
      break;
    }
    auto val_int = static_cast<dsize_t>(val);
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

Status TimeStretch(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate, float hop_length,
                   int32_t n_freq) {
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
      RETURN_IF_NOT_OK(ValidateTensorFloat("TimeStretch", input));
  }
  return Status::OK();
}

Status PhaseVocoder(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate,
                    const std::shared_ptr<Tensor> &phase_advance) {
  const int32_t kFrequencePosInComplex = -3;
  RETURN_IF_NOT_OK(ValidateTensorShape("PhaseVocoder", input->shape().Size() > kDefaultAudioDim && input->IsComplex(),
                                       "<..., freq, num_frame, complex=2>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("PhaseVocoder", input));
  RETURN_IF_NOT_OK(ValidateEqual("PhaseVocoder", "first dimension of 'phase_advance'", phase_advance->shape()[0],
                                 "freq dimension length of input tensor", input->shape()[kFrequencePosInComplex]));
  CHECK_FAIL_RETURN_UNEXPECTED(phase_advance->type() == input->type(),
                               "PhaseVocoder: invalid parameter, data type of phase_advance should be equal to data "
                               "type of input tensor, but got: data type of phase_advance " +
                                 phase_advance->type().ToString() + " while data type of input tensor " +
                                 input->type().ToString() + ".");
  std::shared_ptr<Tensor> input_tensor;
  if (input->type().value() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(TimeStretch<float>(input_tensor, output, rate, phase_advance));
  } else {
    RETURN_IF_NOT_OK(TimeStretch<double>(input, output, rate, phase_advance));
  }
  return Status::OK();
}

Status Dct(std::shared_ptr<Tensor> *output, int n_mfcc, int n_mels, NormMode norm) {
  TensorShape dct_shape({n_mels, n_mfcc});
  Tensor::CreateEmpty(dct_shape, DataType(DataType::DE_FLOAT32), output);
  auto iter = (*output)->begin<float>();
  const float sqrt_2 = 1 / sqrt(2.0f);
  auto sqrt_2_n_mels = static_cast<float>(sqrt(2.0 / n_mels));
  for (int i = 0; i < n_mels; i++) {
    for (int j = 0; j < n_mfcc; j++) {
      // calculate temp:
      // 1. while norm = None, use 2*cos(PI*(i+0.5)*j/n_mels)
      // 2. while norm = Ortho, divide the first row by sqrt(2),
      //    then using sqrt(2.0 / n_mels)*cos(PI*(i+0.5)*j/n_mels)
      auto temp = static_cast<float>(PI / n_mels * (i + 0.5) * j);
      temp = cos(temp);
      if (norm == NormMode::kOrtho) {
        if (j == 0) {
          temp *= sqrt_2;
        }
        temp *= sqrt_2_n_mels;
      } else {
        temp *= 2;
      }
      (*iter++) = temp;
    }
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
  if (mask_width == 0) {
    *output = input;
    return Status::OK();
  }
  if (axis != 2 && axis != 1) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(
      "MaskAlongAxis: invalid parameter, 'axis' can only be 1 for Frequency Masking or 2 for Time Masking.");
  }
  TensorShape input_shape = input->shape();
  // squeeze input
  TensorShape squeeze_shape = TensorShape({-1, input_shape[-2], input_shape[-1]});
  (void)input->Reshape(squeeze_shape);

  int check_dim_ind = (axis == 1) ? -2 : -1;
  CHECK_FAIL_RETURN_SYNTAX_ERROR(mask_start >= 0 && mask_start <= input_shape[check_dim_ind],
                                 "MaskAlongAxis: invalid parameter, 'mask_start' should be less than the length of the "
                                 "masked dimension, but got: 'mask_start' " +
                                   std::to_string(mask_start) + " and length " +
                                   std::to_string(input_shape[check_dim_ind]));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    mask_start + mask_width <= input_shape[check_dim_ind],
    "MaskAlongAxis: invalid parameter, the sum of 'mask_start' and 'mask_width' should be no more "
    "than the length of the masked dimension, but got: 'mask_start' " +
      std::to_string(mask_start) + ", 'mask_width' " + std::to_string(mask_width) + " and length " +
      std::to_string(input_shape[check_dim_ind]));

  size_t cell_size = input->type().SizeInBytes();

  if (axis == 1) {
    // freq
    for (auto ind = 0; ind < input->Size() / input_shape[-2] * mask_width; ind++) {
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
        auto mask_val = static_cast<double>(mask_value);
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_val, cell_size) == 0,
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
        auto mask_val = static_cast<double>(mask_value);
        CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(start_mem_pos, cell_size, &mask_val, cell_size) == 0,
                                     "MaskAlongAxis: mask failed, memory copy error.");
      }
    }
  }
  // unsqueeze input
  (void)input->Reshape(input_shape);
  *output = input;
  return Status::OK();
}

template <typename T>
Status Norm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power) {
  // calculate the output dimension
  auto input_size = input->shape().AsVector();
  auto dim_back = input_size.back();
  RETURN_IF_NOT_OK(
    ValidateTensorShape("ComplexNorm", input->IsComplex(), "<..., complex=2>", std::to_string(dim_back)));
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
  if (input->type().value() >= DataType::DE_BOOL && input->type().value() <= DataType::DE_FLOAT16) {
    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));

    RETURN_IF_NOT_OK(Norm<float>(input_tensor, output, power));
  } else if (input->type().value() == DataType::DE_FLOAT32) {
    RETURN_IF_NOT_OK(Norm<float>(input, output, power));
  } else if (input->type().value() == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(Norm<double>(input, output, power));
  } else {
    RETURN_IF_NOT_OK(ValidateTensorNumeric("ComplexNorm", input));
  }
  return Status::OK();
}

template <typename T>
inline float sgn(T val) {
  return (val > 0) ? 1 : ((val < 0) ? -1 : 0);
}

template <typename T>
Status Decoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T mu) {
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto itr_out = (*output)->begin<T>();
  auto itr = input->begin<T>();
  auto end = input->end<T>();

  while (itr != end) {
    auto x_mu = *itr;
    CHECK_FAIL_RETURN_SYNTAX_ERROR(mu != 0, "Decoding: invalid parameter, 'mu' can not be zero.");
    x_mu = ((x_mu) / mu) * 2 - 1.0;
    x_mu = sgn(x_mu) * expm1(fabs(x_mu) * log1p(mu)) / mu;
    *itr_out = x_mu;
    ++itr_out;
    ++itr;
  }
  return Status::OK();
}

Status MuLawDecoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                     int32_t quantization_channels) {
  if (input->type() != DataType::DE_FLOAT64) {
    float f_mu = static_cast<float>(quantization_channels) - 1;
    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Decoding<float>(input_tensor, output, f_mu));
  } else {
    double f_mu = static_cast<double>(quantization_channels) - 1;
    RETURN_IF_NOT_OK(Decoding<double>(input, output, f_mu));
  }
  return Status::OK();
}

template <typename T>
Status Encoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T mu) {
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), DataType(DataType::DE_INT32), output));
  auto itr_out = (*output)->begin<int32_t>();
  auto itr = input->begin<T>();
  auto end = input->end<T>();

  while (itr != end) {
    auto x = *itr;
    x = sgn(x) * log1p(mu * fabs(x)) / log1p(mu);
    x = (x + 1) / 2 * mu + 0.5;
    *itr_out = static_cast<int32_t>(x);
    ++itr_out;
    ++itr;
  }
  return Status::OK();
}

Status MuLawEncoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                     int32_t quantization_channels) {
  if (input->type() != DataType::DE_FLOAT64) {
    float f_mu = static_cast<float>(quantization_channels) - 1;
    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Encoding<float>(input_tensor, output, f_mu));
  } else {
    double f_mu = static_cast<double>(quantization_channels) - 1;
    RETURN_IF_NOT_OK(Encoding<double>(input, output, f_mu));
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
  auto waveform_length = static_cast<int32_t>(input_shape[-1]);
  RETURN_IF_NOT_OK(ValidateNoGreaterThan("Fade", "fade_in_len", fade_in_len, "length of waveform", waveform_length));
  RETURN_IF_NOT_OK(ValidateNoGreaterThan("Fade", "fade_out_len", fade_out_len, "length of waveform", waveform_length));
  auto num_waveform = static_cast<int32_t>(input->Size() / waveform_length);
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
  if (DataType::DE_BOOL <= input->type().value() && input->type().value() <= DataType::DE_FLOAT32) {
    std::shared_ptr<Tensor> waveform;
    RETURN_IF_NOT_OK(TypeCast(input, &waveform, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Fade<float>(waveform, output, fade_in_len, fade_out_len, fade_shape));
  } else if (input->type().value() == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(Fade<double>(input, output, fade_in_len, fade_out_len, fade_shape));
  } else {
    RETURN_IF_NOT_OK(ValidateTensorNumeric("Fade", input));
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

Status MedianSmoothing(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t win_length) {
  auto channel = input->shape()[0];
  auto num_of_frames = input->shape()[1];
  // Centered windowed
  int32_t pad_length = (win_length - 1) / 2;
  int32_t out_length = num_of_frames + pad_length - win_length + 1;
  TensorShape out_shape({channel, out_length});
  std::vector<int> signal;
  std::vector<int> out;
  std::vector<int> indices(channel * (num_of_frames + pad_length), 0);
  // "replicate" padding in any dimension
  for (auto itr = input->begin<int>(); itr != input->end<int>(); ++itr) {
    signal.push_back(*itr);
  }
  for (int i = 0; i < channel; ++i) {
    for (int j = 0; j < pad_length; ++j) {
      indices[i * (num_of_frames + pad_length) + j] = signal[i * num_of_frames];
    }
  }
  for (int i = 0; i < channel; ++i) {
    for (int j = 0; j < num_of_frames; ++j) {
      indices[i * (num_of_frames + pad_length) + j + pad_length] = signal[i * num_of_frames + j];
    }
  }
  for (int i = 0; i < channel; ++i) {
    int32_t index = i * (num_of_frames + pad_length);
    for (int j = 0; j < out_length; ++j) {
      std::vector<int> tem(indices.begin() + index, indices.begin() + win_length + index);
      std::sort(tem.begin(), tem.end());
      out.push_back(tem[pad_length]);
      ++index;
    }
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out, out_shape, output));
  return Status::OK();
}

Status DetectPitchFrequency(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                            float frame_time, int32_t win_length, int32_t freq_low, int32_t freq_high) {
  std::shared_ptr<Tensor> nccf;
  std::shared_ptr<Tensor> indices;
  std::shared_ptr<Tensor> smooth_indices;
  // pack batch
  TensorShape input_shape = input->shape();
  TensorShape to_shape({input->Size() / input_shape[-1], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));
  if (input->type() == DataType(DataType::DE_FLOAT32)) {
    RETURN_IF_NOT_OK(ComputeNccf<float>(input, &nccf, sample_rate, frame_time, freq_low));
    RETURN_IF_NOT_OK(FindMaxPerFrame<float>(nccf, &indices, sample_rate, freq_high));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    RETURN_IF_NOT_OK(ComputeNccf<double>(input, &nccf, sample_rate, frame_time, freq_low));
    RETURN_IF_NOT_OK(FindMaxPerFrame<double>(nccf, &indices, sample_rate, freq_high));
  } else {
    RETURN_IF_NOT_OK(ComputeNccf<float16>(input, &nccf, sample_rate, frame_time, freq_low));
    RETURN_IF_NOT_OK(FindMaxPerFrame<float16>(nccf, &indices, sample_rate, freq_high));
  }
  RETURN_IF_NOT_OK(MedianSmoothing(indices, &smooth_indices, win_length));

  // Convert indices to frequency
  constexpr double EPSILON = 1e-9;
  TensorShape freq_shape = smooth_indices->shape();
  std::vector<float> out;
  for (auto itr_fre = smooth_indices->begin<int>(); itr_fre != smooth_indices->end<int>(); ++itr_fre) {
    out.push_back(sample_rate / (EPSILON + *itr_fre));
  }

  // unpack batch
  auto shape_vec = input_shape.AsVector();
  shape_vec[shape_vec.size() - 1] = freq_shape[-1];
  TensorShape out_shape(shape_vec);
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out, out_shape, output));
  return Status::OK();
}

Status GenerateWaveTable(std::shared_ptr<Tensor> *output, const DataType &type, Modulation modulation,
                         int32_t table_size, float min, float max, float phase) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(table_size > 0,
                               "table_size must be more than 0, but got: " + std::to_string(table_size));
  auto phase_offset = static_cast<int32_t>(phase / PI / 2 * table_size + 0.5);
  // get the offset of the i-th
  std::vector<int32_t> point;
  for (auto i = 0; i < table_size; i++) {
    point.push_back((i + phase_offset) % table_size);
  }

  std::shared_ptr<Tensor> wave_table;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({table_size}), DataType(DataType::DE_FLOAT32), &wave_table));

  auto iter = wave_table->begin<float>();

  if (modulation == Modulation::kSinusoidal) {
    for (int i = 0; i < table_size; iter++, i++) {
      // change phase
      *iter = (sin(point[i] * PI / table_size * 2) + 1) / 2;
    }
  } else {
    for (int i = 0; i < table_size; iter++, i++) {
      // change phase
      *iter = point[i] * 2.0 / table_size;
      // get complete offset
      auto value = static_cast<int>(4 * point[i] / table_size);
      // change the value of the square wave according to the number of complete offsets
      if (value == 0) {
        *iter = *iter + 0.5;
      } else if (value == 1 || value == 2) {
        *iter = 1.5 - *iter;
      } else if (value == 3) {
        *iter = *iter - 1.5;
      }
    }
  }
  for (iter = wave_table->begin<float>(); iter != wave_table->end<float>(); iter++) {
    *iter = *iter * (max - min) + min;
  }
  if (type.IsInt()) {
    for (iter = wave_table->begin<float>(); iter != wave_table->end<float>(); iter++) {
      if (*iter < 0) {
        *iter = *iter - 0.5;
      } else {
        *iter = *iter + 0.5;
      }
    }
    RETURN_IF_NOT_OK(TypeCast(wave_table, output, DataType(DataType::DE_INT32)));
  } else if (type.IsFloat()) {
    RETURN_IF_NOT_OK(TypeCast(wave_table, output, DataType(DataType::DE_FLOAT32)));
  }

  return Status::OK();
}

Status ReadWaveFile(const std::string &wav_file_dir, std::vector<float> *waveform_vec, int32_t *sample_rate) {
  RETURN_UNEXPECTED_IF_NULL(waveform_vec);
  RETURN_UNEXPECTED_IF_NULL(sample_rate);
  auto wav_realpath = FileUtils::GetRealPath(wav_file_dir.c_str());
  if (!wav_realpath.has_value()) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Invalid file path, get real path failed: " + wav_file_dir);
  }

  const float kMaxVal = 32767.0;
  Path file_path(wav_realpath.value());
  CHECK_FAIL_RETURN_UNEXPECTED(file_path.Exists() && !file_path.IsDirectory(),
                               "Invalid file path, failed to find waveform file: " + file_path.ToString());
  std::ifstream in(file_path.ToString(), std::ios::in | std::ios::binary);
  CHECK_FAIL_RETURN_UNEXPECTED(in.is_open(), "Invalid file, failed to open waveform file: " + file_path.ToString() +
                                               ", make sure the file not damaged or permission denied.");
  auto *header = new WavHeader();
  in.read(reinterpret_cast<char *>(header), sizeof(WavHeader));
  *sample_rate = header->sample_rate;
  float bytes_per_sample = header->bits_per_sample / 8;
  auto sub_chunk2_size = header->sub_chunk2_size;
  if (bytes_per_sample == 0) {
    in.close();
    delete header;
    RETURN_STATUS_UNEXPECTED("ReadWaveFile: zero division error, bits per sample of the audio can not be zero.");
  }
  int num_samples = sub_chunk2_size / bytes_per_sample;
  std::unique_ptr<int16_t[]> data = std::make_unique<int16_t[]>(num_samples);
  in.read(reinterpret_cast<char *>(data.get()), sizeof(int16_t) * num_samples);
  waveform_vec->resize(num_samples);
  for (int i = 0; i < num_samples; i++) {
    (*waveform_vec)[i] = data[i] / kMaxVal;
  }
  in.close();
  delete header;
  return Status::OK();
}

Status ComputeCmnStartAndEnd(int32_t cmn_window, int32_t min_cmn_window, bool center, int32_t idx, int32_t num_frames,
                             int32_t *cmn_window_start_p, int32_t *cmn_window_end_p) {
  RETURN_UNEXPECTED_IF_NULL(cmn_window_start_p);
  RETURN_UNEXPECTED_IF_NULL(cmn_window_end_p);
  RETURN_IF_NOT_OK(ValidateNonNegative("SlidingWindowCmn", "cmn_window", cmn_window));
  RETURN_IF_NOT_OK(ValidateNonNegative("SlidingWindowCmn", "min_cmn_window", min_cmn_window));
  int32_t cmn_window_start = 0, cmn_window_end = 0;
  const constexpr int window_center = 2;
  if (center) {
    cmn_window_start = idx - cmn_window / window_center;
    cmn_window_end = cmn_window_start + cmn_window;
  } else {
    cmn_window_start = idx - cmn_window;
    cmn_window_end = idx + 1;
  }
  if (cmn_window_start < 0) {
    cmn_window_end -= cmn_window_start;
    cmn_window_start = 0;
  }
  if (!center) {
    if (cmn_window_end > idx) {
      cmn_window_end = std::max(idx + 1, min_cmn_window);
    }
  }
  if (cmn_window_end > num_frames) {
    cmn_window_start -= (cmn_window_end - num_frames);
    cmn_window_end = num_frames;
    if (cmn_window_start < 0) {
      cmn_window_start = 0;
    }
  }

  *cmn_window_start_p = cmn_window_start;
  *cmn_window_end_p = cmn_window_end;
  return Status::OK();
}

template <typename T>
Status ComputeCmnWaveform(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *cmn_waveform_p,
                          int32_t num_channels, int32_t num_frames, int32_t num_feats, int32_t cmn_window,
                          int32_t min_cmn_window, bool center, bool norm_vars) {
  using ArrayXT = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  constexpr int square_num = 2;
  int32_t last_window_start = -1, last_window_end = -1;
  ArrayXT cur_sum = ArrayXT(num_channels, num_feats);
  ArrayXT cur_sum_sq;
  if (norm_vars) {
    cur_sum_sq = ArrayXT(num_channels, num_feats);
  }
  for (int i = 0; i < num_frames; ++i) {
    int32_t cmn_window_start = 0, cmn_window_end = 0;
    RETURN_IF_NOT_OK(
      ComputeCmnStartAndEnd(cmn_window, min_cmn_window, center, i, num_frames, &cmn_window_start, &cmn_window_end));
    int32_t row = cmn_window_end - cmn_window_start * 2;
    int32_t cmn_window_frames = cmn_window_end - cmn_window_start;
    for (int32_t m = 0; m < num_channels; ++m) {
      if (last_window_start == -1) {
        auto it = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));
        it += (m * num_frames * num_feats + cmn_window_start * num_feats);
        auto tmp_map = Eigen::Map<ArrayXT>(it, row, num_feats);
        if (i > 0) {
          cur_sum.row(m) += tmp_map.colwise().sum();
          if (norm_vars) {
            cur_sum_sq.row(m) += tmp_map.pow(square_num).colwise().sum();
          }
        } else {
          cur_sum.row(m) = tmp_map.colwise().sum();
          if (norm_vars) {
            cur_sum_sq.row(m) = tmp_map.pow(square_num).colwise().sum();
          }
        }
      } else {
        if (cmn_window_start > last_window_start) {
          auto it = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));
          it += (m * num_frames * num_feats + last_window_start * num_feats);
          auto tmp_map = Eigen::Map<ArrayXT>(it, 1, num_feats);
          cur_sum.row(m) -= tmp_map;
          if (norm_vars) {
            cur_sum_sq.row(m) -= tmp_map.pow(square_num);
          }
        }
        if (cmn_window_end > last_window_end) {
          auto it = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));
          it += (m * num_frames * num_feats + last_window_end * num_feats);
          auto tmp_map = Eigen::Map<ArrayXT>(it, 1, num_feats);
          cur_sum.row(m) += tmp_map;
          if (norm_vars) {
            cur_sum_sq.row(m) += tmp_map.pow(square_num);
          }
        }
      }

      auto it = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));
      auto cmn_it = reinterpret_cast<T *>(const_cast<uchar *>((*cmn_waveform_p)->GetBuffer()));
      it += (m * num_frames * num_feats + i * num_feats);
      cmn_it += (m * num_frames * num_feats + i * num_feats);
      CHECK_FAIL_RETURN_UNEXPECTED(cmn_window_frames != 0,
                                   "SlidingWindowCmn: invalid parameter, 'cmn_window_frames' can not be zero.");
      Eigen::Map<ArrayXT>(cmn_it, 1, num_feats) =
        Eigen::Map<ArrayXT>(it, 1, num_feats) - cur_sum.row(m) / cmn_window_frames;
      if (norm_vars) {
        if (cmn_window_frames == 1) {
          auto cmn_it_1 = reinterpret_cast<T *>(const_cast<uchar *>((*cmn_waveform_p)->GetBuffer()));
          cmn_it_1 += (m * num_frames * num_feats + i * num_feats);
          Eigen::Map<ArrayXT>(cmn_it_1, 1, num_feats).setZero();
        } else {
          auto variance = (Eigen::Map<ArrayXT>(cur_sum_sq.data(), num_channels, num_feats) / cmn_window_frames) -
                          (cur_sum.pow(2) / std::pow(cmn_window_frames, 2));
          auto cmn_it_2 = reinterpret_cast<T *>(const_cast<uchar *>((*cmn_waveform_p)->GetBuffer()));
          cmn_it_2 += (m * num_frames * num_feats + i * num_feats);
          Eigen::Map<ArrayXT>(cmn_it_2, 1, num_feats) =
            Eigen::Map<ArrayXT>(cmn_it_2, 1, num_feats) * (1 / variance.sqrt()).row(m);
        }
      }
    }
    last_window_start = cmn_window_start;
    last_window_end = cmn_window_end;
  }
  return Status::OK();
}

template <typename T>
Status SlidingWindowCmnHelper(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t cmn_window,
                              int32_t min_cmn_window, bool center, bool norm_vars) {
  int32_t num_frames = input->shape()[Tensor::HandleNeg(-2, input->shape().Size())];
  int32_t num_feats = input->shape()[Tensor::HandleNeg(-1, input->shape().Size())];

  int32_t first_index = 1;
  std::vector<dsize_t> input_shape = input->shape().AsVector();
  std::for_each(input_shape.begin(), input_shape.end(), [&first_index](const dsize_t &item) { first_index *= item; });
  RETURN_IF_NOT_OK(
    input->Reshape(TensorShape({static_cast<int>(first_index / (num_frames * num_feats)), num_frames, num_feats})));

  auto num_channels = static_cast<int32_t>(input->shape()[0]);
  TensorPtr cmn_waveform;
  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({num_channels, num_frames, num_feats}), input->type(), &cmn_waveform));
  RETURN_IF_NOT_OK(ComputeCmnWaveform<T>(input, &cmn_waveform, num_channels, num_frames, num_feats, cmn_window,
                                         min_cmn_window, center, norm_vars));

  std::vector<dsize_t> re_shape = input_shape;
  auto r_it = re_shape.rbegin();
  *r_it++ = num_feats;
  *r_it = num_frames;
  RETURN_IF_NOT_OK(cmn_waveform->Reshape(TensorShape(re_shape)));

  const constexpr int specify_input_shape = 2;
  const constexpr int specify_first_shape = 1;
  if (input_shape.size() == specify_input_shape && cmn_waveform->shape()[0] == specify_first_shape) {
    cmn_waveform->Squeeze();
  }
  *output = cmn_waveform;
  return Status::OK();
}

Status SlidingWindowCmn(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t cmn_window,
                        int32_t min_cmn_window, bool center, bool norm_vars) {
  RETURN_IF_NOT_OK(ValidateLowRank("SlidingWindowCmn", input, kDefaultAudioDim, "<..., freq, time>"));

  if (input->type().IsNumeric() && input->type().value() != DataType::DE_FLOAT64) {
    std::shared_ptr<Tensor> temp;
    RETURN_IF_NOT_OK(TypeCast(input, &temp, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(SlidingWindowCmnHelper<float>(temp, output, cmn_window, min_cmn_window, center, norm_vars));
  } else if (input->type().value() == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(SlidingWindowCmnHelper<double>(input, output, cmn_window, min_cmn_window, center, norm_vars));
  } else {
    RETURN_IF_NOT_OK(ValidateTensorNumeric("SlidingWindowCmn", input));
  }
  return Status::OK();
}

template <typename T>
Status Pad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t pad_left, int32_t pad_right,
           BorderType padding_mode, T value = 0) {
  RETURN_IF_NOT_OK(ValidateLowRank("Pad", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Pad", input));
  RETURN_IF_NOT_OK(ValidateNonNegative("Pad", "pad_left", pad_left));
  RETURN_IF_NOT_OK(ValidateNonNegative("Pad", "pad_right", pad_right));
  TensorShape input_shape = input->shape();
  int32_t wave_length = input_shape[-1];
  auto num_wavs = static_cast<int32_t>(input->Size() / wave_length);
  TensorShape to_shape = TensorShape({num_wavs, wave_length});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));
  int32_t pad_length = wave_length + pad_left + pad_right;
  TensorShape new_shape = TensorShape({num_wavs, pad_length});
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(new_shape, input->type(), output));
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Eigen::Map;
  constexpr int pad_mul = 2;
  T *input_data = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));
  T *output_data = reinterpret_cast<T *>(const_cast<uchar *>((*output)->GetBuffer()));
  auto input_map = Map<MatrixXT>(input_data, num_wavs, wave_length);
  auto output_map = Map<MatrixXT>(output_data, num_wavs, pad_length);
  output_map.block(0, pad_left, num_wavs, wave_length) = input_map;
  if (padding_mode == BorderType::kConstant) {
    output_map.block(0, 0, num_wavs, pad_left).setConstant(value);
    output_map.block(0, pad_left + wave_length, num_wavs, pad_right).setConstant(value);
  } else if (padding_mode == BorderType::kEdge) {
    output_map.block(0, 0, num_wavs, pad_left).colwise() = input_map.col(0);
    output_map.block(0, pad_left + wave_length, num_wavs, pad_right).colwise() = input_map.col(wave_length - 1);
  } else if (padding_mode == BorderType::kReflect) {
    // First, deal with the pad operation on the right.
    int32_t current_pad = wave_length - 1;
    while (pad_right >= current_pad) {
      // current_pad: the length of pad required for current loop.
      // pad_right: the length of the remaining pad on the right.
      output_map.block(0, pad_left + current_pad + 1, num_wavs, current_pad) =
        output_map.block(0, pad_left, num_wavs, current_pad).rowwise().reverse();
      pad_right -= current_pad;
      current_pad += current_pad;
    }
    output_map.block(0, pad_length - pad_right, num_wavs, pad_right) =
      output_map.block(0, pad_length - pad_right * pad_mul - 1, num_wavs, pad_right).rowwise().reverse();
    // Next, deal with the pad operation on the left.
    current_pad = wave_length - 1;
    while (pad_left >= current_pad) {
      // current_pad: the length of pad required for current loop.
      // pad_left: the length of the remaining pad on the left.
      output_map.block(0, pad_left - current_pad, num_wavs, current_pad) =
        output_map.block(0, pad_left + 1, num_wavs, current_pad).rowwise().reverse();
      pad_left -= current_pad;
      current_pad += current_pad;
    }
    output_map.block(0, 0, num_wavs, pad_left) =
      output_map.block(0, pad_left + 1, num_wavs, pad_left).rowwise().reverse();
  } else if (padding_mode == BorderType::kSymmetric) {
    // First, deal with the pad operation on the right.
    int32_t current_pad = wave_length;
    while (pad_right >= current_pad) {
      // current_pad: the length of pad required for current loop.
      // pad_right: the length of the remaining pad on the right.
      output_map.block(0, pad_left + current_pad, num_wavs, current_pad) =
        output_map.block(0, pad_left, num_wavs, current_pad).rowwise().reverse();
      pad_right -= current_pad;
      current_pad += current_pad;
    }
    output_map.block(0, pad_length - pad_right, num_wavs, pad_right) =
      output_map.block(0, pad_length - pad_right * pad_mul, num_wavs, pad_right).rowwise().reverse();
    // Next, deal with the pad operation on the left.
    current_pad = wave_length;
    while (pad_left >= current_pad) {
      // current_pad: the length of pad required for current loop.
      // pad_left: the length of the remaining pad on the left.
      output_map.block(0, pad_left - current_pad, num_wavs, current_pad) =
        output_map.block(0, pad_left, num_wavs, current_pad).rowwise().reverse();
      pad_left -= current_pad;
      current_pad += current_pad;
    }
    output_map.block(0, 0, num_wavs, pad_left) = output_map.block(0, pad_left, num_wavs, pad_left).rowwise().reverse();
  } else {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Pad: invalid padding_mode value, check the optional value of BorderType.");
  }
  std::vector<dsize_t> shape_vec = input_shape.AsVector();
  shape_vec[shape_vec.size() - 1] = static_cast<dsize_t>(pad_length);
  TensorShape output_shape(shape_vec);
  RETURN_IF_NOT_OK((*output)->Reshape(output_shape));
  return Status::OK();
}

template <typename T>
Status ComputeDeltasImpl(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int all_freqs,
                         int n_frame, int n) {
  using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Eigen::Map;
  int32_t denom = n * (n + 1) * (n * 2 + 1) / 3;
  // twice sum of integer squared
  VectorXT kernel = VectorXT::LinSpaced(2 * n + 1, -n, n);                         // 2n+1
  T *input_data = reinterpret_cast<T *>(const_cast<uchar *>(input->GetBuffer()));  // [all_freq,n_fram+2n]
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape{all_freqs, n_frame}, input->type(), output));
  T *output_data = reinterpret_cast<T *>(const_cast<uchar *>((*output)->GetBuffer()));
  for (int freq = 0; freq < all_freqs; ++freq) {  // conv with im2col
    auto input_map = Map<MatrixXT, 0, Eigen::OuterStride<1>>(input_data + freq * (n_frame + 2 * n), n_frame,
                                                             2 * n + 1);  // n_frmae,2n+1
    Map<VectorXT>(output_data + freq * n_frame, n_frame) = (input_map * kernel).array() / T(denom);
  }
  return Status::OK();
}

Status Bartlett(std::shared_ptr<Tensor> *output, int len) {
  CHECK_FAIL_RETURN_UNEXPECTED(len != 0, "Bartlett: len can not be zero.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), DataType(DataType::DE_FLOAT32), output));
  // Bartlett window function.
  auto iter = (*output)->begin<float>();
  float twice = 2.0;
  for (ptrdiff_t i = 0; i < len; ++i) {
    *(iter + i) = 1.0 - std::abs(twice * i / len - 1.0);
  }
  return Status::OK();
}

Status Blackman(std::shared_ptr<Tensor> *output, int len) {
  CHECK_FAIL_RETURN_UNEXPECTED(len != 0, "Blackman: len can not be zero.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), DataType(DataType::DE_FLOAT32), output));
  // Blackman window function.
  auto iter = (*output)->begin<float>();
  const float alpha = 0.42;
  const float half = 0.5;
  const float delta = 0.08;
  for (ptrdiff_t i = 0; i < len; ++i) {
    *(iter + i) = alpha - half * std::cos(TWO * PI * i / len) + delta * std::cos(TWO * TWO * PI * i / len);
  }
  return Status::OK();
}

Status Hamming(std::shared_ptr<Tensor> *output, int len, float alpha = 0.54, float beta = 0.46) {
  CHECK_FAIL_RETURN_UNEXPECTED(len != 0, "Hamming: len can not be zero.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), DataType(DataType::DE_FLOAT32), output));
  // Hamming window function.
  auto iter = (*output)->begin<float>();
  for (ptrdiff_t i = 0; i < len; ++i) {
    *(iter + i) = alpha - beta * std::cos(TWO * PI * i / len);
  }
  return Status::OK();
}

Status Hann(std::shared_ptr<Tensor> *output, int len) {
  CHECK_FAIL_RETURN_UNEXPECTED(len != 0, "Hann: len can not be zero.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), DataType(DataType::DE_FLOAT32), output));
  // Hann window function.
  auto iter = (*output)->begin<float>();
  const float half = 0.5;
  for (ptrdiff_t i = 0; i < len; ++i) {
    *(iter + i) = half - half * std::cos(TWO * PI * i / len);
  }
  return Status::OK();
}

Status Kaiser(std::shared_ptr<Tensor> *output, int len, float beta = 12.0) {
#ifdef __APPLE__
  return Status(StatusCode::kMDNotImplementedYet, "For macOS, Kaiser window is not supported.");
#else
  CHECK_FAIL_RETURN_UNEXPECTED(len != 0, "Kaiser: len can not be zero.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), DataType(DataType::DE_FLOAT32), output));
  // Kaiser window function.
  auto iter = (*output)->begin<float>();
  float twice = 2.0;
  for (ptrdiff_t i = 0; i < len; ++i) {
    *(iter + i) =
      std::cyl_bessel_i(0, beta * std::sqrt(1 - std::pow(i * twice / (len)-1.0, TWO))) / std::cyl_bessel_i(0, beta);
  }
  return Status::OK();
#endif
}

Status Window(std::shared_ptr<Tensor> *output, WindowType window_type, int len) {
  switch (window_type) {
    case WindowType::kBartlett:
      return Bartlett(output, len);
    case WindowType::kBlackman:
      return Blackman(output, len);
    case WindowType::kHamming:
      return Hamming(output, len);
    case WindowType::kHann:
      return Hann(output, len);
    case WindowType::kKaiser:
      return Kaiser(output, len);
    default:
      return Hann(output, len);
  }
}

// control whether return half of results after stft.
template <typename T>
Status Onesided(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int n_fft, int n_columns) {
  std::shared_ptr<Tensor> output_onsided;
  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({input->shape()[0], n_fft, n_columns, 2}), input->type(), &output_onsided));
  auto onside_begin = output_onsided->begin<T>();
  auto spec_f_begin = input->begin<T>();
  std::vector<int> spec_f_slice = {(n_fft / 2 + 1) * n_columns * 2, n_columns * 2, 2};
  for (int r = 0; r < input->shape()[0]; r++) {
    for (int i = 0; i < (n_fft / TWO + 1); i++) {
      for (int j = 0; j < n_columns; j++) {
        ptrdiff_t onside_offset_0 = r * n_fft * n_columns * 2 + i * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t spec_f_offset_0 = r * spec_f_slice[0] + i * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t onside_offset_1 = onside_offset_0 + 1;
        ptrdiff_t spec_f_offset_1 = spec_f_offset_0 + 1;
        *(onside_begin + onside_offset_0) = *(spec_f_begin + spec_f_offset_0);
        *(onside_begin + onside_offset_1) = *(spec_f_begin + spec_f_offset_1);
      }
    }
    for (int i = n_fft / 2 + 1; i < n_fft; i++) {
      for (int j = 0; j < n_columns; j++) {
        ptrdiff_t onside_offset_0 = r * n_fft * n_columns * 2 + i * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t spec_f_offset_0 = r * spec_f_slice[0] + (n_fft - i) * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t onside_offset_1 = onside_offset_0 + 1;
        ptrdiff_t spec_f_offset_1 = spec_f_offset_0 + 1;
        *(onside_begin + onside_offset_0) = *(spec_f_begin + spec_f_offset_0);
        *(onside_begin + onside_offset_1) = *(spec_f_begin + spec_f_offset_1);
      }
    }
  }
  *output = output_onsided;
  return Status::OK();
}

template <typename T>
Status PowerStft(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power, int n_columns,
                 int n_length) {
  auto spec_f_begin = input->begin<T>();
  std::vector<int> spec_f_slice = {n_length * n_columns * 2, n_columns * 2, 2};
  std::vector<int> spec_p_slice = {n_length * n_columns, n_columns};
  std::shared_ptr<Tensor> spec_p;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input->shape()[0], n_length, n_columns}), input->type(), &spec_p));
  auto spec_p_begin = spec_p->begin<T>();
  for (int r = 0; r < input->shape()[0]; r++) {
    for (int i = 0; i < n_length; i++) {
      for (int j = 0; j < n_columns; j++) {
        ptrdiff_t spec_f_offset_0 = r * spec_f_slice[0] + i * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t spec_f_offset_1 = spec_f_offset_0 + 1;
        ptrdiff_t spec_p_offset = r * spec_p_slice[0] + i * spec_p_slice[1] + j;
        T spec_power_0 = *(spec_f_begin + spec_f_offset_0);
        T spec_power_1 = *(spec_f_begin + spec_f_offset_1);
        *(spec_p_begin + spec_p_offset) =
          std::pow(std::sqrt(std::pow(spec_power_0, TWO) + std::pow(spec_power_1, TWO)), power);
      }
    }
  }
  *output = spec_p;
  return Status::OK();
}

template <typename T>
Status Stft(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int n_fft,
            const std::shared_ptr<Tensor> &win, int win_length, int n_columns, bool normalized, float power,
            bool onesided) {
  CHECK_FAIL_RETURN_UNEXPECTED(win_length != 0, "Spectrogram: win_length can not be zero.");
  double win_sum = 0.;
  float twice = 2.0;
  for (auto iter_win = win->begin<float>(); iter_win != win->end<float>(); iter_win++) {
    win_sum += (*iter_win) * (*iter_win);
  }
  win_sum = std::sqrt(win_sum);
  std::shared_ptr<Tensor> spec_f;

  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({input->shape()[0], n_fft / 2 + 1, n_columns, 2}), input->type(), &spec_f));

  auto spec_f_begin = spec_f->begin<T>();
  auto input_win_begin = input->begin<T>();
  std::vector<int> spec_f_slice = {(n_fft / 2 + 1) * n_columns * 2, n_columns * 2, 2};
  std::vector<int> input_win_slice = {n_columns * win_length, win_length};
  std::shared_ptr<Tensor> spec_p;
  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({input->shape()[0], n_fft / 2 + 1, n_columns}), input->type(), &spec_p));
  std::shared_ptr<Tensor> exp_complex;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({n_fft / 2 + 1, win_length, 2}), input->type(), &exp_complex));
  auto exp_complex_begin = exp_complex->begin<T>();
  std::vector<int> exp_complex_slice = {win_length * 2, 2};
  for (int i = 0; i < (n_fft / TWO + 1); i++) {
    for (int k = 0; k <= win_length - 1; k++) {
      ptrdiff_t exp_complex_offset_0 = i * exp_complex_slice[0] + k * exp_complex_slice[1];
      ptrdiff_t exp_complex_offset_1 = exp_complex_offset_0 + 1;
      *(exp_complex_begin + exp_complex_offset_0) = std::cos(twice * PI * i * k / win_length);
      *(exp_complex_begin + exp_complex_offset_1) = std::sin(twice * PI * i * k / win_length);
    }
  }
  for (int r = 0; r < input->shape()[0]; r++) {
    for (int i = 0; i < (n_fft / TWO + 1); i++) {
      for (int j = 0; j < n_columns; j++) {
        T spec_f_0 = 0.;
        T spec_f_1 = 0.;
        ptrdiff_t exp_complex_offset_0 = i * exp_complex_slice[0];
        for (int k = 0; k < win_length; k++) {
          ptrdiff_t exp_complex_offset_1 = exp_complex_offset_0 + 1;
          T exp_complex_a = *(exp_complex_begin + exp_complex_offset_0);
          T exp_complex_b = *(exp_complex_begin + exp_complex_offset_1);
          ptrdiff_t input_win_offset = r * input_win_slice[0] + j * input_win_slice[1] + k;
          T input_value = *(input_win_begin + input_win_offset);
          spec_f_0 += input_value * exp_complex_a;
          spec_f_1 += -input_value * exp_complex_b;
          exp_complex_offset_0 = exp_complex_offset_1 + 1;
        }
        ptrdiff_t spec_f_offset_0 = r * spec_f_slice[0] + i * spec_f_slice[1] + j * spec_f_slice[2];
        ptrdiff_t spec_f_offset_1 = spec_f_offset_0 + 1;
        *(spec_f_begin + spec_f_offset_0) = spec_f_0;
        *(spec_f_begin + spec_f_offset_1) = spec_f_1;
      }
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(win_sum != 0, "Window: the total value of window function can not be zero.");
  if (normalized) {
    for (int r = 0; r < input->shape()[0]; r++) {
      for (int i = 0; i < (n_fft / TWO + 1); i++) {
        for (int j = 0; j < n_columns; j++) {
          ptrdiff_t spec_f_offset_0 = r * spec_f_slice[0] + i * spec_f_slice[1] + j * spec_f_slice[2];
          ptrdiff_t spec_f_offset_1 = spec_f_offset_0 + 1;
          T spec_norm_a = *(spec_f_begin + spec_f_offset_0);
          T spec_norm_b = *(spec_f_begin + spec_f_offset_1);
          *(spec_f_begin + spec_f_offset_0) = spec_norm_a / win_sum;
          *(spec_f_begin + spec_f_offset_1) = spec_norm_b / win_sum;
        }
      }
    }
  }
  std::shared_ptr<Tensor> output_onsided;
  if (!onesided) {
    RETURN_IF_NOT_OK(Onesided<T>(spec_f, &output_onsided, n_fft, n_columns));
    if (power == 0) {
      *output = output_onsided;
      return Status::OK();
    }
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input->shape()[0], n_fft, n_columns}), input->type(), &spec_p));
    RETURN_IF_NOT_OK(PowerStft<T>(output_onsided, &spec_p, power, n_columns, n_fft));
    *output = spec_p;
    return Status::OK();
  }
  if (power == 0) {
    *output = spec_f;
    return Status::OK();
  }
  RETURN_IF_NOT_OK(PowerStft<T>(spec_f, &spec_p, power, n_columns, n_fft / TWO + 1));
  *output = spec_p;
  return Status::OK();
}

template <typename T>
Status SpectrogramImpl(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int pad,
                       WindowType window, int n_fft, int hop_length, int win_length, float power, bool normalized,
                       bool center, BorderType pad_mode, bool onesided) {
  std::shared_ptr<Tensor> fft_window_tensor;
  std::shared_ptr<Tensor> fft_window_later;
  TensorShape shape = input->shape();
  std::vector output_shape = shape.AsVector();
  output_shape.pop_back();
  int input_len = input->shape()[-1];

  RETURN_IF_NOT_OK(input->Reshape(TensorShape({input->Size() / input_len, input_len})));

  DataType data_type = input->type();
  // get the windows
  RETURN_IF_NOT_OK(Window(&fft_window_tensor, window, win_length));
  if (win_length == 1) {
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::DE_FLOAT32), &fft_window_tensor));
    auto win = fft_window_tensor->begin<float>();
    *(win) = 1;
  }

  // Pad window length
  int pad_left = (n_fft - win_length) / 2;
  int pad_right = n_fft - win_length - pad_left;
  RETURN_IF_NOT_OK(fft_window_tensor->Reshape(TensorShape({1, win_length})));
  RETURN_IF_NOT_OK(Pad<float>(fft_window_tensor, &fft_window_later, pad_left, pad_right, BorderType::kConstant));
  RETURN_IF_NOT_OK(fft_window_later->Reshape(TensorShape({n_fft})));

  int length = input_len + pad * 2 + n_fft;

  std::shared_ptr<Tensor> input_data_tensor;
  std::shared_ptr<Tensor> input_data_tensor_pad;
  RETURN_IF_NOT_OK(
    Tensor::CreateEmpty(TensorShape({input->shape()[0], input_len + pad * 2}), data_type, &input_data_tensor_pad));
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input->shape()[0], length}), data_type, &input_data_tensor));

  RETURN_IF_NOT_OK(Pad<T>(input, &input_data_tensor_pad, pad, pad, BorderType::kConstant));

  if (center) {
    RETURN_IF_NOT_OK(Pad<T>(input_data_tensor_pad, &input_data_tensor, n_fft / TWO, n_fft / TWO, pad_mode));
  } else {
    input_data_tensor = input_data_tensor_pad;
  }

  CHECK_FAIL_RETURN_UNEXPECTED(n_fft <= input_data_tensor->shape()[-1],
                               "Spectrogram: n_fft should be more than 0 and less than " +
                                 std::to_string(input_data_tensor->shape()[-1]) +
                                 ", but got n_fft: " + std::to_string(n_fft) + ".");

  // calculate the sliding times of the window function
  CHECK_FAIL_RETURN_UNEXPECTED(hop_length != 0, "Spectrogram: hop_length can not be zero.");
  int n_columns = 0;
  while ((1 + n_columns++) * hop_length + n_fft <= input_data_tensor->shape()[-1]) {
  }
  std::shared_ptr<Tensor> stft_compute;

  auto input_begin = input_data_tensor->begin<T>();
  std::vector<int> input_win_slice = {n_columns * n_fft, n_fft};
  auto iter_win = fft_window_later->begin<float>();
  std::shared_ptr<Tensor> input_win;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input_data_tensor->shape()[0], n_columns, n_fft}),
                                       input_data_tensor->type(), &input_win));
  auto input_win_begin = input_win->begin<T>();
  for (int r = 0; r < input_data_tensor->shape()[0]; r++) {
    for (int j = 0; j < n_columns; j++) {
      for (int k = 0; k < n_fft; k++) {
        ptrdiff_t win_offset = k;
        float win_value = *(iter_win + win_offset);
        ptrdiff_t input_stft_offset = r * input_data_tensor->shape()[-1] + j * hop_length + k;
        T input_value = *(input_begin + input_stft_offset);
        ptrdiff_t input_win_offset = r * input_win_slice[0] + j * input_win_slice[1] + k;
        *(input_win_begin + input_win_offset) = win_value * input_value;
      }
    }
  }
  RETURN_IF_NOT_OK(
    Stft<T>(input_win, &stft_compute, n_fft, fft_window_later, n_fft, n_columns, normalized, power, onesided));
  if (onesided) {
    output_shape.push_back(n_fft / TWO + 1);
  } else {
    output_shape.push_back(n_fft);
  }
  output_shape.push_back(n_columns);
  if (power == 0) {
    output_shape.push_back(TWO);
  }
  // reshape the output
  RETURN_IF_NOT_OK(stft_compute->Reshape(TensorShape({output_shape})));
  *output = stft_compute;
  return Status::OK();
}

Status Spectrogram(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int pad, WindowType window,
                   int n_fft, int hop_length, int win_length, float power, bool normalized, bool center,
                   BorderType pad_mode, bool onesided) {
  TensorShape input_shape = input->shape();

  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type().IsNumeric(),
    "Spectrogram: input tensor type should be int, float or double, but got: " + input->type().ToString());

  CHECK_FAIL_RETURN_UNEXPECTED(input_shape.Size() > 0, "Spectrogram: input tensor is not in shape of <..., time>.");

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return SpectrogramImpl<float>(input_tensor, output, pad, window, n_fft, hop_length, win_length, power, normalized,
                                  center, pad_mode, onesided);
  } else {
    input_tensor = input;
    return SpectrogramImpl<double>(input_tensor, output, pad, window, n_fft, hop_length, win_length, power, normalized,
                                   center, pad_mode, onesided);
  }
}

template <typename T>
Status SpectralCentroidImpl(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int sample_rate,
                            int n_fft, int win_length, int hop_length, int pad, WindowType window) {
  std::shared_ptr<Tensor> output_tensor;
  std::shared_ptr<Tensor> spectrogram_tensor;
  if (input->type() == DataType::DE_FLOAT64) {
    SpectrogramImpl<double>(input, &spectrogram_tensor, pad, window, n_fft, hop_length, win_length, 1.0, false, true,
                            BorderType::kReflect, true);
  } else {
    SpectrogramImpl<float>(input, &spectrogram_tensor, pad, window, n_fft, hop_length, win_length, 1.0, false, true,
                           BorderType::kReflect, true);
  }
  std::shared_ptr<Tensor> freqs;
  // sample_rate / TWO is half of sample_rate and n_fft / TWO is half of n_fft
  RETURN_IF_NOT_OK(Linspace<T>(&freqs, 0, sample_rate / TWO, 1 + n_fft / TWO));
  auto itr_freq = freqs->begin<T>();
  int num = freqs->Size();
  TensorShape spectrogram_shape = spectrogram_tensor->shape();
  int waveform = spectrogram_shape[-1];
  int channals = spectrogram_shape[-2];
  std::vector output_shape = spectrogram_shape.AsVector();
  output_shape[output_shape.size() - TWO] = 1;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape{output_shape}, input->type(), &output_tensor));
  Eigen::MatrixXd freqs_r = Eigen::MatrixXd::Zero(num, 1);
  for (int i = 0; i < num; ++i) {
    freqs_r(i, 0) = *itr_freq;
    itr_freq++;
  }
  int k_num = spectrogram_tensor->Size() / (waveform * channals);
  std::vector<Eigen::MatrixXd> specgram;
  std::vector<Eigen::MatrixXd> specgram_result;
  std::vector<Eigen::MatrixXd> specgram_sum;
  Eigen::MatrixXd tmp = Eigen::MatrixXd::Zero(channals, waveform);
  auto itr_spectrogram = spectrogram_tensor->begin<T>();
  for (int k = 0; k < k_num; k++) {
    for (int i = 0; i < channals; ++i) {
      for (int j = 0; j < waveform; ++j) {
        tmp(i, j) = *itr_spectrogram;
        itr_spectrogram++;
      }
    }
    specgram.push_back(tmp);
    specgram_sum.emplace_back(specgram[k].colwise().sum());
  }
  for (int k = 0; k < k_num; k++) {
    for (int i = 0; i < channals; ++i) {
      for (int j = 0; j < waveform; ++j) {
        tmp(i, j) = freqs_r(i, 0) * specgram[k](i, j);
      }
    }
    specgram_result.emplace_back((tmp).colwise().sum());
  }
  auto itr_output = output_tensor->begin<T>();
  for (int k = 0; k < k_num; k++) {
    for (int i = 0; i < waveform; ++i) {
      *itr_output = specgram_result[k](0, i) / specgram_sum[k](0, i);
      itr_output++;
    }
  }
  *output = output_tensor;
  return Status::OK();
}

Status SpectralCentroid(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int sample_rate,
                        int n_fft, int win_length, int hop_length, int pad, WindowType window) {
  RETURN_IF_NOT_OK(ValidateLowRank("SpectralCentroid", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("SpectralCentroid", input));

  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return SpectralCentroidImpl<float>(input_tensor, output, sample_rate, n_fft, win_length, hop_length, pad, window);
  } else {
    input_tensor = input;
    return SpectralCentroidImpl<double>(input_tensor, output, sample_rate, n_fft, win_length, hop_length, pad, window);
  }
}

Status ComputeDeltas(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t win_length,
                     const BorderType &mode) {
  RETURN_IF_NOT_OK(ValidateLowRank("ComputeDeltas", input, kDefaultAudioDim, "<..., freq, time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("ComputeDeltas", input));

  // reshape Tensor from <..., freq, time> to <-1, time>
  auto raw_shape = input->shape();
  int32_t n_frames = raw_shape[-1];
  int32_t all_freqs = raw_shape.NumOfElements() / n_frames;
  RETURN_IF_NOT_OK(input->Reshape(TensorShape{all_freqs, n_frames}));

  int32_t n = (win_length - 1) / 2;

  std::shared_ptr<Tensor> specgram_local_pad;
  if (input->type() == DataType(DataType::DE_FLOAT64)) {
    RETURN_IF_NOT_OK(Pad<double>(input, &specgram_local_pad, n, n, mode));
    RETURN_IF_NOT_OK(ComputeDeltasImpl<double>(specgram_local_pad, output, all_freqs, n_frames, n));
  } else {
    std::shared_ptr<Tensor> float_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &float_tensor, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(Pad<float>(float_tensor, &specgram_local_pad, n, n, mode));
    RETURN_IF_NOT_OK(ComputeDeltasImpl<float>(specgram_local_pad, output, all_freqs, n_frames, n));
  }
  RETURN_IF_NOT_OK((*output)->Reshape(raw_shape));
  return Status::OK();
}

/// \brief IRFFT.
Status IRFFT(const Eigen::MatrixXcd &stft_matrix, Eigen::MatrixXd *inverse) {
  int32_t n = 2 * (stft_matrix.rows() - 1);
  int32_t s = stft_matrix.rows() - 1;
  Eigen::FFT<double> fft;
  for (int k = 0; k < stft_matrix.cols(); ++k) {
    Eigen::VectorXcd output_complex(n);
    // pad input
    Eigen::VectorXcd input(n);
    input.head(s + 1) = stft_matrix.col(k);
    auto reverse_pad = stft_matrix.col(k).segment(1, s - 1).colwise().reverse().conjugate();
    input.segment(s + 1, s - 1) = reverse_pad;
    fft.inv(output_complex, input);
    Eigen::VectorXd output_real = output_complex.real().eval();
    inverse->col(k) = Eigen::Map<Eigen::MatrixXd>(output_real.data(), n, 1);
  }
  return Status::OK();
}

/// \brief Overlap Add
Status OverlapAdd(Eigen::VectorXd *out_buf, const Eigen::MatrixXd &win_inverse_stft, int32_t hop_lengh) {
  int32_t n_fft = win_inverse_stft.rows();
  for (int frame = 0; frame < win_inverse_stft.cols(); frame++) {
    int32_t sample = frame * hop_lengh;
    out_buf->middleRows(sample, n_fft) += win_inverse_stft.col(frame);
  }
  return Status::OK();
}

/// \brief Window Sum Square
Status WindowSumSquare(const Eigen::MatrixXf &window_matrix, Eigen::VectorXf *win_sum_square, const int32_t n_frames,
                       int32_t n_fft, int32_t hop_length) {
  Eigen::MatrixXf win_norm = window_matrix.array().pow(2);
  // window sum square fill
  int32_t n = n_fft + hop_length * (n_frames - 1);
  // check n_fft
  CHECK_FAIL_RETURN_UNEXPECTED(
    n_fft == win_norm.rows(),
    "GriffinLim: n_fft must be equal to the length of the window during window sum square calculation.");
  for (int ind = 0; ind < n_frames; ind++) {
    int sample = ind * hop_length;
    int end_ss = std::min(n, sample + n_fft);
    int end_win = std::max(0, std::min(n_fft, n - sample));
    win_sum_square->segment(sample, end_ss - sample) += win_norm.col(0).head(end_win);
  }
  return Status::OK();
}

/// \brief ISTFT.
/// \param input: Complex matrix of eigen, shape of <freq, time>.
/// \param output: Tensor of shape <time>.
/// \param n_fft: Size of Fourier transform.
/// \param hop_length: The distance between neighboring sliding window frames.
/// \param win_length: The size of window frame and STFT filter.
/// \param window_type: The type of window function.
/// \param center:  Whether input was padded on both sides so that the `t`-th frame is centered at time
///     `t * hop_length`.
/// \param normalized: Whether the STFT was normalized.
/// \param onesided: Whether the STFT was onesided.
/// \param length: The amount to trim the signal by (i.e. the original signal length).
/// \return Status code.
template <typename T>
Status ISTFT(const Eigen::MatrixXcd &stft_matrix, std::shared_ptr<Tensor> *output, int32_t n_fft, int32_t hop_length,
             int32_t win_length, WindowType window_type, bool center, int32_t length) {
  // check input
  CHECK_FAIL_RETURN_UNEXPECTED(n_fft == ((stft_matrix.rows() - 1) * 2),
                               "GriffinLim: the frequency of the input should equal to n_fft / 2 + 1");

  // window
  std::shared_ptr<Tensor> ifft_window_tensor;
  RETURN_IF_NOT_OK(Window(&ifft_window_tensor, window_type, win_length));
  if (win_length == 1) {
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::DE_FLOAT32), &ifft_window_tensor));
    auto win = ifft_window_tensor->begin<float>();
    *(win) = 1;
  }
  // pad window to match n_fft, and add a broadcasting axis
  std::shared_ptr<Tensor> ifft_window_pad;
  ifft_window_pad = ifft_window_tensor;
  if (win_length < n_fft) {
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({n_fft}), DataType(DataType::DE_FLOAT32), &ifft_window_pad));
    int pad_left = (n_fft - win_length) / 2;
    int pad_right = n_fft - win_length - pad_left;
    RETURN_IF_NOT_OK(Pad<float>(ifft_window_tensor, &ifft_window_pad, pad_left, pad_right, BorderType::kConstant));
  }

  int32_t n_frames = 0;
  if ((length != 0) && (hop_length != 0)) {
    int32_t padded_length = center ? (length + n_fft) : length;
    n_frames = std::min(static_cast<int32_t>(stft_matrix.cols()),
                        static_cast<int32_t>(std::ceil(static_cast<float>(padded_length) / hop_length)));
  } else {
    n_frames = stft_matrix.cols();
  }
  int32_t expected_signal_len = n_fft + hop_length * (n_frames - 1);
  Eigen::VectorXd y(expected_signal_len);
  y.setZero();

  // constrain STFT block sizes to 256 KB
  int32_t max_mem_block = std::pow(2, 8) * std::pow(2, 10);
  int32_t n_columns = max_mem_block / (stft_matrix.rows() * sizeof(T) * TWO);
  n_columns = std::max(n_columns, 1);

  // turn window to eigen matrix
  auto data_ptr = &*ifft_window_pad->begin<float>();
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> ifft_window_matrix(data_ptr,
                                                                                      ifft_window_pad->shape()[0], 1);
  for (int bl_s = 0, frame = 0; bl_s < n_frames;) {
    int bl_t = std::min(bl_s + n_columns, n_frames);
    // calculate ifft
    Eigen::MatrixXcd stft_temp = stft_matrix.middleCols(bl_s, bl_t - bl_s).eval();
    Eigen::MatrixXd inverse(TWO * (stft_temp.rows() - 1), stft_temp.cols());
    inverse.setZero();
    RETURN_IF_NOT_OK(IRFFT(stft_temp, &inverse));
    auto ytmp = ifft_window_matrix.template cast<double>().replicate(1, inverse.cols()).cwiseProduct(inverse);
    RETURN_IF_NOT_OK(OverlapAdd(&y, ytmp, hop_length));
    frame += bl_t - bl_s;
    bl_s += n_columns;
  }
  // normalize by sum of squared window
  int32_t n = n_fft + hop_length * (n_frames - 1);
  Eigen::VectorXf ifft_win_sum(n);
  ifft_win_sum.setZero();
  RETURN_IF_NOT_OK(WindowSumSquare(ifft_window_matrix, &ifft_win_sum, n_frames, n_fft, hop_length));

  for (int32_t ind = 0; ind < y.rows(); ind++) {
    if (ifft_win_sum[ind] > std::numeric_limits<float>::min()) {
      y[ind] /= ifft_win_sum[ind];
    }
  }

  std::shared_ptr<Tensor> res_tensor;
  if (length == 0 && center) {
    int32_t y_start = n_fft / 2;
    int32_t y_end = y.rows() - y_start;
    auto tmp = y.middleRows(y_start, y_end - y_start);
    std::vector<T> y_res(tmp.data(), tmp.data() + tmp.rows() * tmp.cols());
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(y_res, TensorShape({tmp.size()}), &res_tensor));
  } else {
    int32_t start = center ? n_fft / 2 : 0;
    auto tmp = y.tail(y.rows() - start);
    // fix length
    std::vector<T> y_res(tmp.data(), tmp.data() + tmp.rows() * tmp.cols());
    if (length > y_res.size()) {
      while (y_res.size() != length) {
        y_res.push_back(0);
      }
    } else if (length < tmp.rows()) {
      while (y_res.size() != length) {
        y_res.pop_back();
      }
    }
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(y_res, TensorShape({length}), &res_tensor));
  }
  *output = res_tensor;
  return Status::OK();
}

template <typename T>
Status GriffinLimImpl(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t n_fft,
                      int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type, float power,
                      float momentum, int32_t length, bool rand_init, std::mt19937 rnd) {
  // pack
  TensorShape shape = input->shape();
  TensorShape new_shape({input->Size() / shape[-1] / shape[-2], shape[-2], shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(new_shape));
  // power
  CHECK_FAIL_RETURN_UNEXPECTED(power != 0, "GriffinLim: power can not be zero.");
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    *itr = pow(*itr, 1 / power);
  }
  // window
  std::shared_ptr<Tensor> fft_window_tensor;
  RETURN_IF_NOT_OK(Window(&fft_window_tensor, window_type, win_length));
  if (win_length == 1) {
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1}), DataType(DataType::DE_FLOAT32), &fft_window_tensor));
    auto win = fft_window_tensor->begin<float>();
    *(win) = 1;
  }
  std::shared_ptr<Tensor> final_results;
  for (int dim = 0; dim < new_shape[0]; dim++) {
    // init complex phase
    Eigen::MatrixXcd angles(shape[(-1) * TWO], shape[-1]);
    if (rand_init) {
      // static std::default_random_engine e;
      std::uniform_real_distribution<double> dist(0, 1);
      angles = angles.unaryExpr(
        [&dist, &rnd](std::complex<double> value) { return std::complex<double>(dist(rnd), dist(rnd)); });
    } else {
      angles = angles.unaryExpr([](std::complex<double> value) { return std::complex<double>(1, 0); });
    }
    // slice and squeeze the first dim
    std::shared_ptr<Tensor> spec_tensor_slice;
    RETURN_IF_NOT_OK(input->Slice(
      &spec_tensor_slice,
      std::vector<SliceOption>({SliceOption(std::vector<dsize_t>{dim}), SliceOption(true), SliceOption(true)})));
    TensorShape new_slice_shape({shape[-2], shape[-1]});
    RETURN_IF_NOT_OK(spec_tensor_slice->Reshape(new_slice_shape));
    // turn tensor into eigen MatrixXd
    auto data_ptr = &*spec_tensor_slice->begin<T>();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> spec_matrix_transpose(data_ptr, shape[-1],
                                                                                       shape[(-1) * TWO]);
    Eigen::MatrixXd spec_matrix = spec_matrix_transpose.transpose().template cast<double>();
    auto stft_complex = angles.cwiseProduct(spec_matrix);

    // init tprev zero mat
    Eigen::MatrixXcd tprev(shape[(-1) * TWO], shape[-1]);
    Eigen::MatrixXcd rebuilt(shape[(-1) * TWO], shape[-1]);
    tprev.setZero();
    rebuilt.setZero();
    for (int iter = 0; iter < n_iter; iter++) {
      // istft
      std::shared_ptr<Tensor> inverse;
      RETURN_IF_NOT_OK(ISTFT<T>(stft_complex, &inverse, n_fft, hop_length, win_length, window_type, true, length));
      // stft
      std::shared_ptr<Tensor> stft_out;
      RETURN_IF_NOT_OK(SpectrogramImpl<T>(inverse, &stft_out, 0, window_type, n_fft, hop_length, win_length, 0, false,
                                          true, BorderType::kReflect, true));

      rebuilt.transposeInPlace();
      Tensor::TensorIterator<T> itr = stft_out->begin<T>();
      rebuilt = rebuilt.unaryExpr([&itr](std::complex<double> value) {
        T real = *(itr++);
        T img = *(itr++);
        return std::complex<double>(real, img);
      });
      rebuilt.transposeInPlace();
      angles = rebuilt.array();

      if (momentum != 0) {
        tprev = tprev * ((momentum) / (1 + momentum));
        angles = angles.array() - tprev.array();
      }

      float eps = 1e-16;
      auto angles_abs = angles.cwiseAbs().eval();
      angles_abs.array() += eps;
      angles = angles.array() / angles_abs.array();
      tprev = rebuilt.array();
    }

    // istft calculate final phase
    auto stft_complex_fin = angles.cwiseProduct(spec_matrix);
    std::shared_ptr<Tensor> waveform;
    RETURN_IF_NOT_OK(ISTFT<T>(stft_complex_fin, &waveform, n_fft, hop_length, win_length, window_type, true, length));

    if (shape.Rank() == TWO) {
      // do not expand dim
      final_results = waveform;
      continue;
    }

    if (final_results != nullptr) {
      RETURN_IF_NOT_OK(final_results->InsertTensor({dim}, waveform));
    } else {
      RETURN_IF_NOT_OK(
        Tensor::CreateEmpty(TensorShape({new_shape[0], waveform->shape()[0]}), waveform->type(), &final_results));
      RETURN_IF_NOT_OK(final_results->InsertTensor({dim}, waveform));
    }
  }
  *output = final_results;
  return Status::OK();
}

Status GriffinLim(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t n_fft, int32_t n_iter,
                  int32_t win_length, int32_t hop_length, WindowType window_type, float power, float momentum,
                  int32_t length, bool rand_init, std::mt19937 rnd) {
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return GriffinLimImpl<float>(input_tensor, output, n_fft, n_iter, win_length, hop_length, window_type, power,
                                 momentum, length, rand_init, rnd);
  } else {
    input_tensor = input;
    return GriffinLimImpl<double>(input_tensor, output, n_fft, n_iter, win_length, hop_length, window_type, power,
                                  momentum, length, rand_init, rnd);
  }
}

template <typename T>
Status InverseMelScaleImpl(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t n_stft,
                           int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t max_iter,
                           float tolerance_loss, float tolerance_change, float sgd_lr, float sgd_momentum,
                           NormType norm, MelType mel_type, std::mt19937 rnd) {
  f_max = f_max == 0 ? static_cast<T>(std::floor(sample_rate / 2)) : f_max;
  // create fb mat <freq, n_mels>
  std::shared_ptr<Tensor> freq_bin_mat;
  RETURN_IF_NOT_OK(CreateFbanks<T>(&freq_bin_mat, n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_type));

  auto fb_ptr = &*freq_bin_mat->begin<float>();
  Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_fb(fb_ptr, n_mels, n_stft);
  // pack melspec <n, n_mels, time>
  TensorShape input_shape = input->shape();
  TensorShape input_reshape({input->Size() / input_shape[-1] / input_shape[-2], input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(input_reshape));
  CHECK_FAIL_RETURN_UNEXPECTED(n_mels == input_shape[-1 * TWO],
                               "InverseMelScale: n_mels must be equal to the penultimate dimension of input.");

  int time = input_shape[-1];
  int freq = matrix_fb.cols();
  // input matrix 3d
  std::vector<T> specgram;
  // engine for random matrix
  std::uniform_real_distribution<T> dist(0, 1);
  for (int channel = 0; channel < input_reshape[0]; channel++) {
    // slice by first dimension
    auto data_ptr = &*input->begin<T>();
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> input_channel(data_ptr + time * n_mels * channel, time,
                                                                               n_mels);
    // init specgram at n=channel
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat_channel =
      Eigen::MatrixXd::Zero(time, freq).unaryExpr([&rnd, &dist](double dummy) { return dist(rnd); });
    std::vector<T> vec_channel(mat_channel.data(), mat_channel.data() + mat_channel.size());
    std::shared_ptr<Tensor> param_channel;
    TensorShape output_shape = TensorShape({freq, time});
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(vec_channel, TensorShape({freq * time}), &param_channel));
    // sgd
    T loss = std::numeric_limits<T>::max();
    for (int epoch = 0; epoch < max_iter; epoch++) {
      auto pred = mat_channel * (matrix_fb.transpose().template cast<T>());
      // cal loss with pred and gt
      auto diff = input_channel - pred;
      T new_loss = diff.array().square().mean();
      // cal grad
      auto grad = diff * (matrix_fb.template cast<T>()) * (-1) / time;
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat_grad = grad;
      std::vector<T> vec_grad(mat_grad.data(), mat_grad.data() + mat_grad.size());
      std::shared_ptr<Tensor> tensor_grad;
      RETURN_IF_NOT_OK(Tensor::CreateFromVector(vec_grad, TensorShape({grad.size()}), &tensor_grad));

      std::shared_ptr<Tensor> nspec;
      RETURN_IF_NOT_OK(SGD<T>(param_channel, &nspec, tensor_grad, sgd_lr, sgd_momentum));

      T diff_loss = std::abs(loss - new_loss);
      if ((new_loss < tolerance_loss) || (diff_loss < tolerance_change)) {
        break;
      }
      loss = new_loss;
      data_ptr = &*nspec->begin<T>();
      mat_channel = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(data_ptr, time, freq);
      // use new mat_channel to update param_channel
      RETURN_IF_NOT_OK(Tensor::CreateFromTensor(nspec, &param_channel));
    }
    // clamp and transpose
    auto res = mat_channel.cwiseMax(0);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat_res = res;
    std::vector<T> spec_channel(mat_res.data(), mat_res.data() + mat_res.size());
    specgram.insert(specgram.end(), spec_channel.begin(), spec_channel.end());
  }
  std::shared_ptr<Tensor> final_out;
  if (input_shape.Size() > TWO) {
    std::vector<int64_t> out_shape_vec = input_shape.AsVector();
    out_shape_vec[input_shape.Size() - 1] = time;
    out_shape_vec[input_shape.Size() - TWO] = freq;
    TensorShape output_shape(out_shape_vec);
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(specgram, output_shape, &final_out));
  } else {
    TensorShape output_shape = TensorShape({input_reshape[0], freq, time});
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(specgram, output_shape, &final_out));
  }
  *output = final_out;
  return Status::OK();
}

Status InverseMelScale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t n_stft,
                       int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t max_iter,
                       float tolerance_loss, float tolerance_change, float sgd_lr, float sgd_momentum, NormType norm,
                       MelType mel_type, std::mt19937 rnd) {
  std::shared_ptr<Tensor> input_tensor;
  if (input->type() != DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));
    return InverseMelScaleImpl<float>(input_tensor, output, n_stft, n_mels, sample_rate, f_min, f_max, max_iter,
                                      tolerance_loss, tolerance_change, sgd_lr, sgd_momentum, norm, mel_type, rnd);
  } else {
    input_tensor = input;
    return InverseMelScaleImpl<double>(input_tensor, output, n_stft, n_mels, sample_rate, f_min, f_max, max_iter,
                                       tolerance_loss, tolerance_change, sgd_lr, sgd_momentum, norm, mel_type, rnd);
  }
}

template <typename T>
Status GetSincResampleKernel(int32_t orig_freq, int32_t des_freq, ResampleMethod resample_method,
                             int32_t lowpass_filter_width, float rolloff, float beta, DataType datatype,
                             std::shared_ptr<Tensor> *kernel, int32_t *width) {
  CHECK_FAIL_RETURN_UNEXPECTED(orig_freq != 0, "Resample: invalid parameter, 'orig_freq' can not be zero.");
  CHECK_FAIL_RETURN_UNEXPECTED(des_freq != 0, "Resample: invalid parameter, 'des_freq' can not be zero.");
  float base_freq = static_cast<float>(std::min(orig_freq, des_freq));
  // Removing the highest frequencies to perform antialiasing filtering. This is needed in both upsampling and
  // downsampling
  base_freq *= rolloff;
  // The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor) using the sinc
  // interpolation formula:
  //  x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
  // We can then sample the function x(t) with a different sample rate:
  //  y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / des_freq))
  *width = static_cast<int32_t>(ceil(static_cast<float>(lowpass_filter_width * orig_freq) / base_freq));
  int32_t kernel_length = 2 * (*width) + orig_freq;
  std::shared_ptr<Tensor> idx;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1, kernel_length}), datatype, &idx));
  auto idx_it = idx->begin<T>();
  for (int32_t i = -(*width); i < (*width + orig_freq); i++) {
    *idx_it = static_cast<T>(i);
    idx_it++;
  }
  std::vector<std::shared_ptr<Tensor>> kernels_vec;
  std::shared_ptr<Tensor> filter;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(idx, &filter));
  auto filter_begin = filter->begin<T>();
  auto filter_end = filter->end<T>();
  std::shared_ptr<Tensor> window;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(filter->shape(), filter->type(), &window));
  auto window_it = window->begin<T>();
  idx_it = idx->begin<T>();
  // Compute the filter for resample, sinc interpolation
  for (int32_t i = 0; i < des_freq; i++) {
    while (filter_begin != filter_end) {
      *filter_begin = (static_cast<T>(-i) / des_freq + *idx_it / orig_freq) * base_freq;
      *filter_begin =
        std::clamp(*filter_begin, static_cast<T>(-lowpass_filter_width), static_cast<T>(lowpass_filter_width));
      // Build window
      if (resample_method == ResampleMethod::kSincInterpolation) {
        *window_it = *filter_begin * PI / lowpass_filter_width / 2.0;
        *window_it = pow(cos(*window_it), 2);
      } else {
#ifdef __APPLE__
        RETURN_STATUS_ERROR(StatusCode::kMDNotImplementedYet,
                            "Resample: ResampleMethod of Kaiser Window is not supported on MacOS yet.");
#else
        *window_it =
          static_cast<T>(std::cyl_bessel_i(3.0, beta * sqrt(1 - pow((*filter_begin / lowpass_filter_width), 2)))) /
          (static_cast<T>(std::cyl_bessel_if(3.0, beta)));
#endif
      }
      *filter_begin *= PI;
      // sinc function
      if (*filter_begin == 0) {
        *filter_begin = 1.0;
      } else {
        *filter_begin = sin(*filter_begin) / *filter_begin;
      }
      *filter_begin *= (*window_it);
      ++window_it;
      ++filter_begin;
      ++idx_it;
    }
    std::shared_ptr<Tensor> temp_filter;
    RETURN_IF_NOT_OK(Tensor::CreateFromTensor(filter, &temp_filter));
    kernels_vec.emplace_back(temp_filter);
    window_it = window->begin<T>();
    filter_begin = filter->begin<T>();
    idx_it = idx->begin<T>();
  }
  T scale = static_cast<T>(base_freq) / orig_freq;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({des_freq, kernel_length}), datatype, kernel));
  T temp;
  for (int32_t i = 0; i < des_freq; i++) {
    for (int32_t j = 0; j < kernel_length; j++) {
      RETURN_IF_NOT_OK(kernels_vec.at(i)->GetItemAt<T>(&temp, {0, j}));
      temp *= scale;
      RETURN_IF_NOT_OK((*kernel)->SetItemAt<T>({i, j}, temp));
    }
  }
  return Status::OK();
}

template <typename T>
Status ResampleSingle(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                      const std::shared_ptr<Tensor> &kernel, int32_t orig_freq, int32_t target_length) {
  int32_t pad_length = input->shape()[-1];
  RETURN_IF_NOT_OK(input->Reshape(TensorShape({pad_length})));
  int32_t kernel_x = kernel->shape()[0];
  int32_t kernel_y = kernel->shape()[1];
  std::shared_ptr<Tensor> multi_input;
  int32_t resample_num = static_cast<int32_t>(ceil(static_cast<float>(target_length) / kernel_x));
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({resample_num, kernel_y}), input->type(), &multi_input));
  int x_dim = 0;
  // Slice the waveform with stride orig_freq and length kernel_y
  for (int i = 0; i + kernel_y < pad_length && x_dim < resample_num; i += orig_freq) {
    std::vector<dsize_t> slice_idx(kernel_y);
    std::iota(slice_idx.begin(), slice_idx.end(), i);
    std::shared_ptr<Tensor> input_slice;
    RETURN_IF_NOT_OK(input->Slice(&input_slice, std::vector<SliceOption>({SliceOption(slice_idx)})));
    RETURN_IF_NOT_OK(multi_input->InsertTensor({x_dim}, input_slice));
    ++x_dim;
  }
  auto multi_input_ptr = &*multi_input->begin<T>();
  auto kernel_ptr = &*kernel->begin<T>();
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> multi_input_matrix(multi_input_ptr, kernel_y,
                                                                                  resample_num);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> kernel_matrix(kernel_ptr, kernel_y, kernel_x);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mul_matrix =
    (multi_input_matrix.transpose() * (kernel_matrix)).transpose();
  std::vector<T> mul_mat(mul_matrix.data(), mul_matrix.data() + mul_matrix.size());
  std::shared_ptr<Tensor> resampled;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(mul_mat, &resampled));
  std::vector<dsize_t> resampled_idx(target_length);
  std::iota(resampled_idx.begin(), resampled_idx.end(), 0);
  RETURN_IF_NOT_OK(resampled->Slice(output, std::vector<SliceOption>({SliceOption(resampled_idx)})));
  return Status::OK();
}

template <typename T>
Status Resample(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float orig_freq, float des_freq,
                ResampleMethod resample_method, int32_t lowpass_filter_width, float rolloff, float beta) {
  TensorShape input_shape = input->shape();
  int32_t waveform_length = input_shape[-1];
  int32_t num_waveform = input->Size() / waveform_length;
  TensorShape to_shape = TensorShape({num_waveform, waveform_length});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));

  int32_t gcd = std::gcd(static_cast<int32_t>(orig_freq), static_cast<int32_t>(des_freq));
  CHECK_FAIL_RETURN_UNEXPECTED(gcd != 0, "Resample: gcd cannet be equal to 0.");
  int32_t orig_freq_prime = static_cast<int32_t>(floor(orig_freq / gcd));
  int32_t des_freq_prime = static_cast<int32_t>(floor(des_freq / gcd));
  CHECK_FAIL_RETURN_UNEXPECTED(orig_freq_prime != 0, "Resample: invalid parameter, 'orig_freq_prime' can not be zero.");
  std::shared_ptr<Tensor> kernel;
  int32_t width = 0;
  RETURN_IF_NOT_OK(GetSincResampleKernel<T>(orig_freq_prime, des_freq_prime, resample_method, lowpass_filter_width,
                                            rolloff, beta, input->type(), &kernel, &width));
  // padding
  int32_t pad_length = waveform_length + 2 * width + orig_freq_prime;
  std::shared_ptr<Tensor> waveform_pad;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({1, pad_length}), input->type(), &waveform_pad));
  RETURN_IF_NOT_OK(Pad<T>(input, &waveform_pad, width, width + orig_freq_prime, BorderType::kConstant));

  int32_t target_length =
    static_cast<int32_t>(std::ceil(static_cast<double>(des_freq_prime * waveform_length) / orig_freq_prime));
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({num_waveform, target_length}), input->type(), output));
  for (int32_t i = 0; i < num_waveform; i++) {
    std::shared_ptr<Tensor> single_waveform;
    RETURN_IF_NOT_OK(waveform_pad->Slice(
      &single_waveform, std::vector<SliceOption>({SliceOption(std::vector<dsize_t>{i}), SliceOption(true)})));
    std::shared_ptr<Tensor> resampled_single;
    RETURN_IF_NOT_OK(ResampleSingle<T>(single_waveform, &resampled_single, kernel, orig_freq_prime, target_length));
    RETURN_IF_NOT_OK((*output)->InsertTensor({i}, resampled_single));
  }
  std::vector<dsize_t> shape_vec = input_shape.AsVector();
  shape_vec.at(shape_vec.size() - 1) = static_cast<dsize_t>(target_length);
  TensorShape output_shape(shape_vec);
  RETURN_IF_NOT_OK((*output)->Reshape(output_shape));
  return Status::OK();
}

Status Resample(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float orig_freq, float des_freq,
                ResampleMethod resample_method, int32_t lowpass_filter_width, float rolloff, float beta) {
  RETURN_IF_NOT_OK(ValidateLowRank("Resample", input, kMinAudioDim, "<..., time>"));
  RETURN_IF_NOT_OK(ValidateTensorNumeric("Resample", input));
  if (input->type().value() == DataType::DE_FLOAT64) {
    RETURN_IF_NOT_OK(
      Resample<double>(input, output, orig_freq, des_freq, resample_method, lowpass_filter_width, rolloff, beta));
  } else {
    std::shared_ptr<Tensor> waveform;
    RETURN_IF_NOT_OK(TypeCast(input, &waveform, DataType(DataType::DE_FLOAT32)));
    RETURN_IF_NOT_OK(
      Resample<float>(waveform, output, orig_freq, des_freq, resample_method, lowpass_filter_width, rolloff, beta));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore

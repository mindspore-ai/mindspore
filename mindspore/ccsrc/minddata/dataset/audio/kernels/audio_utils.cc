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
  RETURN_IF_NOT_OK(ValidateTensorShape("ComplexAngle", input->IsComplex(), "<..., complex=2>"));
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
    std::string err_msg = "Polar: the shape of input tensor abs and angle should be the same, but got: abs " +
                          abs->shape().ToString() + " and angle " + angle->shape().ToString();
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
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
  pad_shape_vec[dim] += static_cast<int64_t>(length);
  TensorShape input_shape_with_pad(pad_shape_vec);
  std::vector<T> in_vect(input_shape_with_pad[0] * input_shape_with_pad[1] * input_shape_with_pad[2] *
                         input_shape_with_pad[3]);
  auto itr_input = input->begin<T>();
  int64_t input_cnt = 0;
  /*lint -e{446} ind is modified in the body of the for loop */
  for (int ind = 0; ind < static_cast<int>(in_vect.size()); ind++) {
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
  int64_t ind = 0;
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

template <typename T>
Status CreateTriangularFilterbank(std::shared_ptr<Tensor> *output, const std::shared_ptr<Tensor> &all_freqs,
                                  const std::shared_ptr<Tensor> &f_pts) {
  // calculate the difference between each mel point and each stft freq point in hertz.
  std::vector<T> f_diff;
  auto iter_fpts1 = f_pts->begin<T>();
  auto iter_fpts2 = f_pts->begin<T>();
  ++iter_fpts2;
  for (size_t i = 1; i < f_pts->Size(); i++) {
    f_diff.push_back(*iter_fpts2 - *iter_fpts1);
    ++iter_fpts2;
    ++iter_fpts1;
  }

  std::vector<T> slopes;
  TensorShape slopes_shape({all_freqs->Size(), f_pts->Size()});
  auto iter_all_freq = all_freqs->begin<T>();
  for (; iter_all_freq != all_freqs->end<T>(); ++iter_all_freq) {
    auto iter_f_pts = f_pts->begin<T>();
    for (; iter_f_pts != f_pts->end<T>(); ++iter_f_pts) {
      slopes.push_back(*iter_f_pts - *iter_all_freq);
    }
  }

  // calculate up and down slopes for creating overlapping triangles.
  std::vector<T> down_slopes;
  TensorShape down_slopes_shape({all_freqs->Size(), f_pts->Size() - 2});
  for (size_t row = 0; row < down_slopes_shape[0]; row++)
    for (size_t col = 0; col < down_slopes_shape[1]; col++) {
      down_slopes.push_back(-slopes[col + row * f_pts->Size()] / f_diff[col]);
    }
  std::vector<T> up_slopes;
  TensorShape up_slopes_shape({all_freqs->Size(), f_pts->Size() - 2});
  for (size_t row = 0; row < up_slopes_shape[0]; row++)
    for (size_t col = 2; col < f_pts->Size(); col++) {
      up_slopes.push_back(slopes[col + row * f_pts->Size()] / f_diff[col - 1]);
    }

  // clip the value of triangles and save into fb.
  std::vector<T> fb;
  TensorShape fb_shape({all_freqs->Size(), f_pts->Size() - 2});
  for (size_t i = 0; i < down_slopes.size(); i++) {
    fb.push_back(std::max(0.0f, std::min(down_slopes[i], up_slopes[i])));
  }

  std::shared_ptr<Tensor> fb_tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(fb, fb_shape, &fb_tensor));
  *output = fb_tensor;
  return Status::OK();
}

Status CreateFbanks(std::shared_ptr<Tensor> *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                    int32_t sample_rate, NormType norm, MelType mel_type) {
  // min_log_hz, min_log_mel, logstep and f_sp are the const of the mel value equation.
  const double min_log_hz = 1000.0;
  const double min_log_mel = 1000 / (200.0 / 3);
  const double logstep = log2(6.4) / 27.0;
  const double f_sp = 200.0 / 3;

  // hez_to_mel_c and mel_to_hz_c are the const coefficient of mel frequency cepstrum.
  const double hz_to_mel_c = 2595.0;
  const double mel_to_hz_c = 700.0;

  // all_freqs is equivalent filterbank construction.
  std::shared_ptr<Tensor> all_freqs;
  // the sampling frequency is at least twice the highest frequency of the signal.
  const double signal_times = 2;
  RETURN_IF_NOT_OK(Linspace<float>(&all_freqs, 0, sample_rate / signal_times, n_freqs));

  // calculate mel value by f_min and f_max.
  double m_min = 0.0;
  double m_max = 0.0;
  if (mel_type == MelType::kHtk) {
    m_min = hz_to_mel_c * log10(1.0 + (f_min / mel_to_hz_c));
    m_max = hz_to_mel_c * log10(1.0 + (f_max / mel_to_hz_c));
  } else {
    m_min = (f_min - 0.0) / f_sp;
    m_max = (f_max - 0.0) / f_sp;
    if (m_min >= min_log_hz) {
      m_min = min_log_mel + log2(f_min / min_log_hz) / logstep;
    }
    if (m_max >= min_log_hz) {
      m_max = min_log_mel + log2(f_max / min_log_hz) / logstep;
    }
  }

  // m_pts is mel value sequence in linspace of  (m_min, m_max).
  std::shared_ptr<Tensor> m_pts;
  const int32_t bias = 2;
  RETURN_IF_NOT_OK(Linspace<float>(&m_pts, m_min, m_max, n_mels + bias));

  // f_pts saves hertz(mel) though 700.0 * (10.0 **(mel/ 2595.0) - 1.).
  std::shared_ptr<Tensor> f_pts;
  const double htk_mel_c = 10.0;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(m_pts->shape(), DataType(DataType::DE_FLOAT32), &f_pts));

  if (mel_type == MelType::kHtk) {
    auto iter_f = f_pts->begin<float>();
    auto iter_m = m_pts->begin<float>();
    for (; iter_m != m_pts->end<float>(); ++iter_m) {
      *iter_f = mel_to_hz_c * (pow(htk_mel_c, *iter_m / hz_to_mel_c) - 1.0);
      ++iter_f;
    }
  } else {
    auto iter_f = f_pts->begin<float>();
    auto iter_m = m_pts->begin<float>();
    for (; iter_m != m_pts->end<float>(); iter_m++, iter_f++) {
      *iter_f = f_sp * (*iter_m);
    }
    iter_f = f_pts->begin<float>();
    iter_m = m_pts->begin<float>();
    for (; iter_m != m_pts->end<float>(); iter_m++, iter_f++) {
      if (*iter_m >= min_log_mel) {
        *iter_f = min_log_hz * exp(logstep * (*iter_m - min_log_mel));
      }
    }
  }

  // create filterbank
  TensorShape fb_shape({all_freqs->Size(), f_pts->Size() - 2});
  std::shared_ptr<Tensor> fb;
  RETURN_IF_NOT_OK(CreateTriangularFilterbank<float>(&fb, all_freqs, f_pts));

  // normalize with Slaney
  std::vector<float> enorm;
  if (norm == NormType::kSlaney) {
    auto iter_f_pts_0 = f_pts->begin<float>();
    auto iter_f_pts_2 = f_pts->begin<float>();
    iter_f_pts_2++;
    iter_f_pts_2++;
    for (; iter_f_pts_2 != f_pts->end<float>(); iter_f_pts_0++, iter_f_pts_2++) {
      enorm.push_back(2.0f / (*iter_f_pts_2 - *iter_f_pts_0));
    }
    auto iter_fb = fb->begin<float>();
    for (size_t row = 0; row < fb_shape[0]; row++) {
      for (size_t col = 0; col < fb_shape[1]; col++) {
        *iter_fb = (*iter_fb) * enorm[col];
        iter_fb++;
      }
    }
    enorm.clear();
  }

  // anomaly detection.
  auto iter_fb = fb->begin<float>();
  std::vector<float> max_val(fb_shape[1], 0);
  for (size_t row = 0; row < fb_shape[0]; row++) {
    for (size_t col = 0; col < fb_shape[1]; col++) {
      max_val[col] = std::max(max_val[col], *iter_fb);
      iter_fb++;
    }
  }
  for (size_t col = 0; col < fb_shape[1]; col++) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      max_val[col] >= 1e-8,
      "MelscaleFbanks: at least one mel filterbank is all zeros, check if the value for 'n_mels' " +
        std::to_string(n_mels) + " is set too high or the value for 'n_freqs' " + std::to_string(n_freqs) +
        " is set too low.");
  }

  *output = fb;
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
  std::vector<dsize_t> time_steps_0, time_steps_1;
  std::vector<T> alphas;
  for (int ind = 0;; ind++) {
    auto val = ind * rate;
    if (val >= input_shape[-2]) {
      break;
    }
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

Status TimeStretch(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate, float hop_length,
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
  const float sqrt_2 = 1 / sqrt(2);
  float sqrt_2_n_mels = sqrt(2.0 / n_mels);
  for (int i = 0; i < n_mels; i++) {
    for (int j = 0; j < n_mfcc; j++) {
      // calculate temp:
      // 1. while norm = None, use 2*cos(PI*(i+0.5)*j/n_mels)
      // 2. while norm = Ortho, divide the first row by sqrt(2),
      //    then using sqrt(2.0 / n_mels)*cos(PI*(i+0.5)*j/n_mels)
      float temp = PI / n_mels * (i + 0.5) * j;
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
  (void)input->Reshape(input_shape);
  *output = input;
  return Status::OK();
}

template <typename T>
Status Norm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power) {
  // calculate the output dimension
  auto input_size = input->shape().AsVector();
  int32_t dim_back = static_cast<int32_t>(input_size.back());
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
  if (input->type().value() >= DataType::DE_INT8 && input->type().value() <= DataType::DE_FLOAT16) {
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
float sgn(T val) {
  return static_cast<float>(static_cast<T>(0) < val) - static_cast<float>(val < static_cast<T>(0));
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
  if (input->type().IsInt() || input->type() == DataType(DataType::DE_FLOAT16) ||
      input->type() == DataType(DataType::DE_FLOAT32)) {
    float f_mu = static_cast<float>(quantization_channels) - 1;

    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));

    RETURN_IF_NOT_OK(Decoding<float>(input_tensor, output, f_mu));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    double f_mu = static_cast<double>(quantization_channels) - 1;

    RETURN_IF_NOT_OK(Decoding<double>(input, output, f_mu));
  } else {
    RETURN_IF_NOT_OK(ValidateTensorNumeric("MuLawDecoding", input));
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
  if (input->type().IsInt() || input->type() == DataType(DataType::DE_FLOAT16)) {
    float f_mu = static_cast<float>(quantization_channels) - 1;

    // convert the data type to float
    std::shared_ptr<Tensor> input_tensor;
    RETURN_IF_NOT_OK(TypeCast(input, &input_tensor, DataType(DataType::DE_FLOAT32)));

    RETURN_IF_NOT_OK(Encoding<float>(input_tensor, output, f_mu));
  } else if (input->type() == DataType(DataType::DE_FLOAT32)) {
    float f_mu = static_cast<float>(quantization_channels) - 1;

    RETURN_IF_NOT_OK(Encoding<float>(input, output, f_mu));
  } else if (input->type() == DataType(DataType::DE_FLOAT64)) {
    double f_mu = static_cast<double>(quantization_channels) - 1;

    RETURN_IF_NOT_OK(Encoding<double>(input, output, f_mu));
  } else {
    RETURN_IF_NOT_OK(ValidateTensorNumeric("MuLawEncoding", input));
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
  RETURN_IF_NOT_OK(ValidateNoGreaterThan("Fade", "fade_in_len", fade_in_len, "length of waveform", waveform_length));
  RETURN_IF_NOT_OK(ValidateNoGreaterThan("Fade", "fade_out_len", fade_out_len, "length of waveform", waveform_length));
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
  int32_t phase_offset = static_cast<int32_t>(phase / PI / 2 * table_size + 0.5);
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
      int32_t value = static_cast<int>(4 * point[i] / table_size);
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
  auto wav_realpath = FileUtils::GetRealPath(wav_file_dir.data());
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
  WavHeader *header = new WavHeader();
  in.read(reinterpret_cast<char *>(header), sizeof(WavHeader));
  *sample_rate = header->sample_rate;
  float bytes_per_sample = header->bits_per_sample / 8;
  if (bytes_per_sample == 0) {
    in.close();
    delete header;
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "ReadWaveFile: zero division error, bits per sample of the audio can not be zero.");
  }
  int num_samples = header->sub_chunk2_size / bytes_per_sample;
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

  int32_t num_channels = static_cast<int32_t>(input->shape()[0]);
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
  int32_t num_wavs = static_cast<int32_t>(input->Size() / wave_length);
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
Status PowerStft(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power, int n_fft,
                 int n_columns, int n_length) {
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
            const std::shared_ptr<Tensor> &win, int win_length, int hop_length, int n_columns, bool normalized,
            float power, bool onesided) {
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
    RETURN_IF_NOT_OK(PowerStft<T>(output_onsided, &spec_p, power, n_fft, n_columns, n_fft));
    *output = spec_p;
    return Status::OK();
  }
  if (power == 0) {
    *output = spec_f;
    return Status::OK();
  }
  RETURN_IF_NOT_OK(PowerStft<T>(spec_f, &spec_p, power, n_fft, n_columns, n_fft / TWO + 1));
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
  RETURN_IF_NOT_OK(Stft<T>(input_win, &stft_compute, n_fft, fft_window_later, n_fft, hop_length, n_columns, normalized,
                           power, onesided));
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
    specgram_sum.push_back(specgram[k].colwise().sum());
  }
  for (int k = 0; k < k_num; k++) {
    for (int i = 0; i < channals; ++i) {
      for (int j = 0; j < waveform; ++j) {
        tmp(i, j) = freqs_r(i, 0) * specgram[k](i, j);
      }
    }
    specgram_result.push_back((tmp).colwise().sum());
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
}  // namespace dataset
}  // namespace mindspore

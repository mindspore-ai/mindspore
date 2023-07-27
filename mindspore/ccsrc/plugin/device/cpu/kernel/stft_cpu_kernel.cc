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

#include "plugin/device/cpu/kernel/stft_cpu_kernel.h"

#include "mindspore/core/ops/math_ops.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/stft.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kSTFTInputsNum = 2;
const size_t kSTFTOutputsNum = 1;
const int64_t kSTFTInputMaxDim = 2;
const int64_t kSTFTInputMinDim = 1;
const int64_t kSTFTNum0 = 0;
const int64_t kSTFTNum1 = 1;
const int64_t kSTFTNum2 = 2;
const double KSTFTPI = 3.14159265358979323846;

template <class T>
struct GetCFromR {
  void operator()(T input_num, complex128 *res) const {
    res->real(static_cast<double>(input_num));
    res->imag(0);
  }
};

template <class T>
struct GetCFromC64 {
  void operator()(T input_num, complex128 *res) const {
    res->real(static_cast<double>(input_num.real()));
    res->imag(static_cast<double>(input_num.imag()));
  }
};

template <class T>
struct GetCFromC128 {
  void operator()(T input_num, complex128 *res) const { *res = input_num; }
};

template <class R>
struct Trans2C128 {
  void operator()(R *output, int64_t *index, const complex128 temp) const {
    output[*index] = temp;
    *index += 1;
  }
};

template <class R>
struct Trans2C64 {
  void operator()(R *output, int64_t *index, const complex128 temp) const {
    output[*index].real(static_cast<float>(temp.real()));
    output[*index].imag(static_cast<float>(temp.imag()));
    *index += 1;
  }
};

template <class R>
struct Trans2R {
  void operator()(R *output, int64_t *index, const complex128 temp) const {
    output[*index] = static_cast<R>(temp.real());
    *index += 1;
    output[*index] = static_cast<R>(temp.imag());
    *index += 1;
  }
};
}  // namespace

bool STFTCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  if (!base_operator) {
    MS_LOG(ERROR) << "For " << kernel_type_ << ", cast " << kernel_type_ << " ops failed!";
    return false;
  }
  kernel_name_ = base_operator->name();
  batch_rank_ = base_operator->get_batch_rank();

  auto kernel_ptr = std::dynamic_pointer_cast<ops::STFT>(base_operator);
  n_fft_ = kernel_ptr->get_n_fft();
  hop_length_ = kernel_ptr->get_hop_length();
  if (hop_length_ <= 0) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ", expected hop_length > 0, but got hop_length=" << hop_length_ << ".";
    return false;
  }
  win_length_ = kernel_ptr->get_win_length();
  if (win_length_ <= 0 || win_length_ > n_fft_) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ", expected 0 < win_length <= n_fft_, but got win_length=" << win_length_
                  << ".";
    return false;
  }
  if (win_length_ < n_fft_) {
    pad_window_ = true;
  }
  normalized_ = kernel_ptr->get_normalized();
  if (normalized_) {
    norm_coe_.real(static_cast<double>(1.0 / std::sqrt(n_fft_)));
  } else {
    norm_coe_.real(1.0);
  }
  onesided_ = kernel_ptr->get_onesided();
  if (onesided_) {
    fft_length_ = n_fft_ / kSTFTNum2 + kSTFTNum1;
  } else {
    fft_length_ = n_fft_;
  }
  return_complex_ = kernel_ptr->get_return_complex();

  if (inputs.size() != kSTFTInputsNum || outputs.size() != kSTFTOutputsNum) {
    MS_LOG(ERROR) << "For" << kernel_name_ << ": input and output size should be " << kSTFTInputsNum << " and "
                  << kSTFTOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  input_type_1_ = inputs[kIndex0]->GetDtype();
  input_type_2_ = inputs[kIndex1]->GetDtype();
  output_type_ = outputs[kIndex0]->GetDtype();

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  return true;
}

int STFTCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != 0) {
    return ret;
  }

  MS_EXCEPTION_IF_NULL(inputs[kIndex0]);
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  input_shape_1_ = inputs[kIndex0]->GetShapeVector();
  input_shape_2_ = inputs[kIndex1]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();
  // input shape (vmap_B, [batches], input_len)
  // window shape (vmap_B, win_length)
  // output shape (vmap_B, [batches], fft_length_, n_frames, [2])
  // get vmap batch size
  for (int64_t index = 0; index < batch_rank_; index++) {
    if (input_shape_1_[index] != input_shape_2_[index]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the batch dimensions of two input should be the same but got "
                    << "one batch dimension shape input=" << input_shape_1_[index]
                    << " window=" << input_shape_2_[index];
      return KRET_RESIZE_FAILED;
    }
    vmap_batches_ *= input_shape_1_[index];
  }
  int64_t in_shape_size_1 = static_cast<int64_t>(input_shape_1_.size());
  int64_t in_shape_size_2 = static_cast<int64_t>(input_shape_2_.size());
  // check input is 1D or 2D
  if (in_shape_size_1 - batch_rank_ == kSTFTInputMinDim) {
    has_batches_ = false;
    batches_ = kSTFTNum1;
    input_len_ = input_shape_1_[batch_rank_];
  } else if (in_shape_size_1 - batch_rank_ == kSTFTInputMaxDim) {
    has_batches_ = true;
    batches_ = input_shape_1_[batch_rank_];
    input_len_ = input_shape_1_[batch_rank_ + kIndex1];
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input should be in ["
                  << kSTFTInputMinDim + batch_rank_ << ", " << kSTFTInputMaxDim + batch_rank_ << "], but got "
                  << in_shape_size_1 << ".";
    return KRET_RESIZE_FAILED;
  }
  // check window is 1D
  if (in_shape_size_2 != kSTFTInputMinDim + batch_rank_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of window should be "
                  << kSTFTInputMinDim + batch_rank_ << ", but got a" << in_shape_size_2 << "D Tensor.";
    return KRET_RESIZE_FAILED;
  }
  if (input_shape_2_[batch_rank_] != win_length_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', expected the size of window equal to win_length=" << win_length_
                  << ", but got window with size " << input_shape_2_[batch_rank_] << ".";
    return KRET_RESIZE_FAILED;
  }

  if (n_fft_ <= kSTFTNum0 || n_fft_ >= input_len_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', expected 0 < n_fft < " << input_len_ << ", but got n_fft=" << n_fft_
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  // for window padding
  switch (input_type_2_) {
    case kNumberTypeFloat32:
      (void)workspace_size_list_.emplace_back((sizeof(float) * n_fft_));
      break;
    case kNumberTypeFloat64:
      (void)workspace_size_list_.emplace_back((sizeof(double) * n_fft_));
      break;
    case kNumberTypeComplex64:
      (void)workspace_size_list_.emplace_back((sizeof(complex64) * n_fft_));
      break;
    case kNumberTypeComplex128:
      (void)workspace_size_list_.emplace_back((sizeof(complex128) * n_fft_));
      break;
    default:
      MS_LOG(ERROR) << "STFT kernel does not support " << TypeIdToString(input_type_2_);
      return KRET_RESIZE_FAILED;
  }

  n_frames_ = 1 + (input_len_ - n_fft_) / hop_length_;
  window_left_ = (n_fft_ - win_length_) / kSTFTNum2;

  // for calculate index
  w_skip_ = n_frames_;
  if (!return_complex_) {
    w_skip_ *= kSTFTNum2;
  }
  parallel_num_ = static_cast<size_t>(batches_ * fft_length_);
  return KRET_OK;
}

template <typename T, typename S, typename R, typename DataFT, typename DataFS, typename DataFR>
bool STFTCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                    const std::vector<AddressPtr> &workspace,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSTFTInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSTFTOutputsNum, kernel_name_);
  const auto *input1 = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const auto *window = reinterpret_cast<S *>(inputs[kIndex1]->addr);
  auto *window_paded = reinterpret_cast<S *>(workspace[kIndex0]->addr);
  auto *output = reinterpret_cast<R *>(outputs[kIndex0]->addr);

  for (int64_t vmap_b = 0; vmap_b < vmap_batches_; vmap_b++) {
    // padding
    int64_t index = 0;
    if (pad_window_) {
      S window_zero = static_cast<S>(0);
      for (; index < window_left_; index++) {
        window_paded[index] = window_zero;
      }
      for (; index < window_left_ + win_length_; index++) {
        window_paded[index] = window[index - window_left_];
      }
      for (; index < n_fft_; index++) {
        window_paded[index] = window_zero;
      }
    } else {
      for (; index < n_fft_; index++) {
        window_paded[index] = window[index];
      }
    }

    // FFt
    DataFT input_d_f;
    DataFS win_d_f;
    DataFR output_d_f;
    auto task = [this, &input1, &window_paded, &output, input_d_f, win_d_f, output_d_f](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        int64_t batch_index = i / this->fft_length_;
        int64_t w = i % this->fft_length_;
        int64_t input_index_start = batch_index * this->input_len_;
        int64_t output_index = i * this->w_skip_;
        for (int64_t m = 0; m < this->n_frames_; m++) {
          this->temp_ = kSTFTComplexZero;
          for (int64_t k = 0; k < this->n_fft_; k++) {
            this->complex_w_ = kSTFTComplexZero;
            this->complex_input_ = kSTFTComplexZero;
            input_d_f(input1[input_index_start + m * this->hop_length_ + k], &this->complex_input_);
            win_d_f(window_paded[k], &this->complex_w_);
            this->temp_ +=
              this->complex_w_ * this->complex_input_ * exp(kSTFTNegI * (2.0 * KSTFTPI * w * k / this->n_fft_));
          }
          this->temp_ *= this->norm_coe_;
          output_d_f(output, &output_index, this->temp_);
        }
      }
    };
    ParallelLaunchAutoSearch(task, parallel_num_, this, &parallel_search_info_, pool_);

    input1 += batches_ * input_len_;
    window += win_length_;
    output += batches_ * fft_length_ * w_skip_;
  }
  return true;
}

#define STFT_CPU_MATCH(MS_T, MS_S, MS_R, T, S, R, DataFT, DataFS, DataFR) \
  KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_S).AddOutputAttr(MS_R), \
    &STFTCpuKernelMod::LaunchKernel<T, S, R, DataFT<T>, DataFS<S>, DataFR<R>>

const std::vector<std::pair<KernelAttr, STFTCpuKernelMod::KernelRunFunc>> &STFTCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, STFTCpuKernelMod::KernelRunFunc>> func_list = {
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, float, float, float, GetCFromR,
                    GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeComplex64, float, float, complex64, GetCFromR,
                    GetCFromR, Trans2C64)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeFloat64, float, double, double, GetCFromR,
                    GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex128, float, double, complex128, GetCFromR,
                    GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeComplex64, kNumberTypeFloat32, float, complex64, float, GetCFromR,
                    GetCFromC64, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeComplex64, kNumberTypeComplex64, float, complex64, complex64,
                    GetCFromR, GetCFromC64, Trans2C64)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeComplex128, kNumberTypeFloat64, float, complex128, double, GetCFromR,
                    GetCFromC128, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat32, kNumberTypeComplex128, kNumberTypeComplex128, float, complex128, complex128,
                    GetCFromR, GetCFromC128, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeFloat64, double, float, double, GetCFromR,
                    GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeFloat32, kNumberTypeComplex128, double, float, complex128, GetCFromR,
                    GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64, double, double, double, GetCFromR,
                    GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeComplex128, double, double, complex128,
                    GetCFromR, GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeFloat64, double, complex64, double, GetCFromR,
                    GetCFromC64, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeComplex128, double, complex64, complex128,
                    GetCFromR, GetCFromC64, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeComplex128, kNumberTypeFloat64, double, complex128, double,
                    GetCFromR, GetCFromC128, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeFloat64, kNumberTypeComplex128, kNumberTypeComplex128, double, complex128, complex128,
                    GetCFromR, GetCFromC128, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeFloat32, kNumberTypeFloat32, complex64, float, float, GetCFromC64,
                    GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeFloat32, kNumberTypeComplex64, complex64, float, complex64,
                    GetCFromC64, GetCFromR, Trans2C64)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeFloat64, kNumberTypeFloat64, complex64, double, double,
                    GetCFromC64, GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeFloat64, kNumberTypeComplex128, complex64, double, complex128,
                    GetCFromC64, GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeComplex64, kNumberTypeFloat32, complex64, complex64, float,
                    GetCFromC64, GetCFromC64, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeComplex64, kNumberTypeComplex64, complex64, complex64, complex64,
                    GetCFromC64, GetCFromC64, Trans2C64)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeFloat64, complex64, complex128, double,
                    GetCFromC64, GetCFromC128, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex64, kNumberTypeComplex128, kNumberTypeComplex128, complex64, complex128,
                    complex128, GetCFromC64, GetCFromC128, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeFloat32, kNumberTypeFloat64, complex128, float, double,
                    GetCFromC128, GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeFloat32, kNumberTypeComplex128, complex128, float, complex128,
                    GetCFromC128, GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeFloat64, kNumberTypeFloat64, complex128, double, double,
                    GetCFromC128, GetCFromR, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeFloat64, kNumberTypeComplex128, complex128, double, complex128,
                    GetCFromC128, GetCFromR, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeComplex64, kNumberTypeFloat64, complex128, complex64, double,
                    GetCFromC128, GetCFromC64, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeComplex64, kNumberTypeComplex128, complex128, complex64,
                    complex128, GetCFromC128, GetCFromC64, Trans2C128)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeComplex128, kNumberTypeFloat64, complex128, complex128, double,
                    GetCFromC128, GetCFromC128, Trans2R)},
    {STFT_CPU_MATCH(kNumberTypeComplex128, kNumberTypeComplex128, kNumberTypeComplex128, complex128, complex128,
                    complex128, GetCFromC128, GetCFromC128, Trans2C128)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, STFT,
                                 []() { return std::make_shared<STFTCpuKernelMod>(prim::kPrimSTFT->name()); });
}  // namespace kernel
}  // namespace mindspore

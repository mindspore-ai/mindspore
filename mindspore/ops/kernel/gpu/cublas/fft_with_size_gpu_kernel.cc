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

#include "kernel/gpu/cublas/fft_with_size_gpu_kernel.h"
#include <cmath>
#include <utility>
#include <algorithm>
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int RANK_MIN = 1;
constexpr int RANK_MAX = 3;
constexpr int K_HALF = 2;

#ifndef CHECK_CUBLAS_RET_WITH_ERROR_RET_FALSE
#define CHECK_CUBLAS_RET_WITH_ERROR_RET_FALSE(expression, message)                          \
  do {                                                                                      \
    cublasStatus_t status = (expression);                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                  \
      MS_LOG(ERROR) << "cuBLAS Error: " << message << " | Error Number: " << status << " "; \
      return false;                                                                         \
    }                                                                                       \
  } while (0)
#endif  // CHECK_CUBLAS_RET_WITH_ERROR_RET_FALSE

#ifndef CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE
#define CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE(expression, message)                          \
  do {                                                                                     \
    cufftResult status = (expression);                                                     \
    if (status != CUFFT_SUCCESS) {                                                         \
      MS_LOG(ERROR) << "CUFFT Error: " << message << " | Error Number: " << status << " "; \
      return false;                                                                        \
    }                                                                                      \
  } while (0)
#endif  // CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE

enum class FFTNormMode {
  none,       // no normalization
  by_n,       // divide by the product of the lengths of all transformation dimensions
  by_root_n,  // same as above, but sqrt the product
};

FFTNormMode GetNormModeFromString(const ops::NormMode &norm_type, const bool is_inverse) {
  if (norm_type == ops::NormMode::FORWARD) {
    return is_inverse ? FFTNormMode::none : FFTNormMode::by_n;
  }
  if (norm_type == ops::NormMode::BACKWARD) {
    return is_inverse ? FFTNormMode::by_n : FFTNormMode::none;
  }
  if (norm_type == ops::NormMode::ORTHO) {
    return FFTNormMode::by_root_n;
  }
  MS_LOG(ERROR) << "For 'FFTWithSize', the fft norm type " << norm_type << " is unsupported!";
  return FFTNormMode::none;
}

double GetNormScale(const ops::NormMode &norm_type, const bool is_inverse, const int n) {
  FFTNormMode norm_mode = GetNormModeFromString(norm_type, is_inverse);
  if (norm_mode == FFTNormMode::none) {
    return 1.0;
  }
  double scale_denom = (norm_mode == FFTNormMode::by_root_n) ? std::sqrt(n) : static_cast<double>(n);
  return 1.0 / scale_denom;
}

FFTVariety GetFFTVariety(const bool &is_inverse, const bool &is_real) {
  if (is_real) {
    if (is_inverse) {
      return FFTVariety::irfft;
    } else {
      return FFTVariety::rfft;
    }
  } else {
    if (is_inverse) {
      return FFTVariety::ifft;
    } else {
      return FFTVariety::fft;
    }
  }
  return FFTVariety::unknown;
}
}  // namespace

bool FFTWithSizeGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  fit_index_ = index;
  MS_EXCEPTION_IF_NULL(inputs[0]);
  data_type_bytes_ = GetTypeByte(TypeIdToType(inputs[0]->dtype_id()));
  return true;
}

bool FFTWithSizeGpuKernelMod::FFTVarietyInResize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  rank_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  is_inverse_ = inputs[kIndex2]->GetValueWithCheck<bool>();
  is_real_ = inputs[kIndex3]->GetValueWithCheck<bool>();
  norm_type_ = static_cast<ops::NormMode>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
  is_onesided_ = inputs[kIndex5]->GetValueWithCheck<bool>();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto index = fit_index_;
  fft_variety_ = GetFFTVariety(is_inverse_, is_real_);

  switch (fft_variety_) {
    case FFTVariety::fft:
      resize_func_ = &FFTWithSizeGpuKernelMod::ResizeFFT;
      if (index == kIndex4) {  // input_dtype == kNumberTypeComplex64 && output_dtype == kNumberTypeComplex64
        cufft_type_ = CUFFT_C2C;
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchFFT<cufftComplex>;
      } else if (index == kIndex5) {  // input_dtype == kNumberTypeComplex128 && output_dtype == kNumberTypeComplex128
        cufft_type_ = CUFFT_Z2Z;
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchFFT<cufftDoubleComplex>;
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "(fft)', it does not support this kernel type: " << kernel_attr;
        return false;
      }
      break;
    case FFTVariety::ifft:
      resize_func_ = &FFTWithSizeGpuKernelMod::ResizeIFFT;
      if (index == kIndex4) {  // input_dtype == kNumberTypeComplex64 && output_dtype == kNumberTypeComplex64
        cufft_type_ = CUFFT_C2C;
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchIFFT<cufftComplex>;
      } else if (index == kIndex5) {  // input_dtype == kNumberTypeComplex128 && output_dtype == kNumberTypeComplex128
        cufft_type_ = CUFFT_Z2Z;
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchIFFT<cufftDoubleComplex>;
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "(ifft)', it does not support this kernel type: " << kernel_attr;
        return false;
      }
      break;
    case FFTVariety::rfft:
      resize_func_ = &FFTWithSizeGpuKernelMod::ResizeRFFT;
      if (index == kIndex0) {  // input_dtype == kNumberTypeFloat32 && output_dtype == kNumberTypeComplex64
        cufft_type_ = is_onesided_ ? CUFFT_R2C : CUFFT_C2C;  // true: R2C, false: R->C2C
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchRFFT<float, cufftComplex>;
      } else if (index == kIndex1) {  // input_dtype == kNumberTypeFloat64 && output_dtype == kNumberTypeComplex128
        cufft_type_ = is_onesided_ ? CUFFT_D2Z : CUFFT_Z2Z;  // true: D2Z, false: D->Z2Z
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchRFFT<double, cufftDoubleComplex>;
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "(rfft)', it does not support this kernel type: " << kernel_attr;
        return false;
      }
      break;
    case FFTVariety::irfft:
      resize_func_ = &FFTWithSizeGpuKernelMod::ResizeIRFFT;
      if (index == kIndex2) {  // input_dtype == kNumberTypeComplex64 && output_dtype == kNumberTypeFloat32
        cufft_type_ = is_onesided_ ? CUFFT_C2R : CUFFT_C2C;  // true: C2R, false: C2C->R
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchIRFFT<cufftComplex, float>;
      } else if (index == kIndex3) {  // input_dtype == kNumberTypeComplex128 && output_dtype == kNumberTypeFloat64
        cufft_type_ = is_onesided_ ? CUFFT_Z2D : CUFFT_Z2Z;  // true: Z2D, false: Z2Z->D
        launch_func_ = &FFTWithSizeGpuKernelMod::LaunchIRFFT<cufftDoubleComplex, double>;
      } else {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "(irfft)', it does not support this kernel type: " << kernel_attr;
        return false;
      }
      break;
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
      return false;
  }
  return true;
}

void FFTWithSizeGpuKernelMod::ResetResource() noexcept {
  if (cufft_plan_ != 0) {
    cufftDestroy(cufft_plan_);
    cufft_plan_ = 0;
  }
  if (scale_plan_ != nullptr) {
    cublasDestroy_v2(scale_plan_);
    scale_plan_ = nullptr;
  }
  scale_factor_ = 1.0;
  x_elements_ = 0;
  y_elements_ = 0;
  x_shape_.clear();
  y_shape_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

bool FFTWithSizeGpuKernelMod::MakeCufftPlan(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  if (cufft_plan_ != 0) {  // if there is already a plan, destroy it.
    CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE(cufftDestroy(cufft_plan_),
                                         "For '" << kernel_name_ << "', it failed to call cufftDestroy.");
    cufft_plan_ = 0;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  auto output_shape = outputs[0]->GetShapeVector();
  int batch = std::accumulate(input_shape.begin(), input_shape.end() - rank_, 1, std::multiplies<int>());
  // array of dimensions for fft transformation
  std::vector<int> shape_for_trans(input_shape.end() - rank_, input_shape.end());
  shape_for_trans.back() =
    static_cast<int>(input_shape.back() > output_shape.back() ? input_shape.back() : output_shape.back());
  std::vector<int> inembed(input_shape.end() - rank_, input_shape.end());
  std::vector<int> onembed(output_shape.end() - rank_, output_shape.end());
  // product of all dimensions to be transformed.
  int idist = std::accumulate(inembed.begin(), inembed.end(), 1, std::multiplies<int>());
  // product of all dimensions to be transformed.
  int odist = std::accumulate(onembed.begin(), onembed.end(), 1, std::multiplies<int>());
  CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE(cufftCreate(&cufft_plan_),
                                       "For '" << kernel_name_ << "', it failed to call cufftCreate.");
  CHECK_CUFFT_RET_WITH_ERROR_RET_FALSE(
    cufftPlanMany(&cufft_plan_, rank_, shape_for_trans.data(), inembed.data(), 1, idist,  // NULL, 1, 0,
                  onembed.data(), 1, odist,                                               // NULL, 1, 0,
                  cufft_type_, batch),
    "For '" << kernel_name_ << "', it failed to call cufftPlanMany.");

  if (scale_plan_ != nullptr) {  // if there is already a plan, destroy it.
    CHECK_CUBLAS_RET_WITH_ERROR_RET_FALSE(cublasDestroy_v2(scale_plan_),
                                          "For '" << kernel_name_ << "', it failed to call cublasDestroy_v2.");
    scale_plan_ = nullptr;
  }
  CHECK_CUBLAS_RET_WITH_ERROR_RET_FALSE(cublasCreate_v2(&scale_plan_),
                                        "For '" << kernel_name_ << "', it failed to call cublasCreate_v2.");
  x_elements_ = batch * idist;
  y_elements_ = batch * odist;
  int n = idist > odist ? idist : odist;
  scale_factor_ = GetNormScale(norm_type_, is_inverse_, n);
  return true;
}

int FFTWithSizeGpuKernelMod::ResizeBase(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (rank_ > RANK_MAX || rank_ < RANK_MIN) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the value of signal_ndim must be in range of "
                  << "[" << RANK_MIN << ", " << RANK_MAX << "].\n";
    return KRET_RESIZE_FAILED;
  }
  x_shape_ = inputs[0]->GetShapeVector();
  y_shape_ = outputs[0]->GetShapeVector();
  if (SizeToLong(x_shape_.size()) < rank_) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimensions of input cannot be less than "
                  << "the number of dimensions to be transformed.\n";
    return KRET_RESIZE_FAILED;
  }
  if (x_shape_.size() != y_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimensions of the input must be same as the output.\n";
    return KRET_RESIZE_FAILED;
  }
  for (size_t axis = 0; axis + 1 < x_shape_.size(); ++axis) {  // axis [0, x_shape_.size() - 1)
    if (x_shape_[axis] != y_shape_[axis]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of the input and output "
                    << "at the axis " << axis << " must be same.\n";
      return KRET_RESIZE_FAILED;
    }
  }
  return ret;
}

int FFTWithSizeGpuKernelMod::ResizeFFT(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = ResizeBase(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (x_shape_.back() != y_shape_.back()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "(fft)', the length of the input and output "
                  << "at the last axis must be same.\n";
    return KRET_RESIZE_FAILED;
  }
  if (!MakeCufftPlan(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return ret;
}
int FFTWithSizeGpuKernelMod::ResizeIFFT(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = ResizeBase(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (x_shape_.back() != y_shape_.back()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "(ifft)', the length of the input and output "
                  << "at the last axis must be same.\n";
    return KRET_RESIZE_FAILED;
  }
  if (!MakeCufftPlan(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return ret;
}

int FFTWithSizeGpuKernelMod::ResizeRFFT(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  int ret = ResizeBase(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (is_onesided_) {
    if ((x_shape_.back() / K_HALF) + 1 != y_shape_.back()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "(rfft, onesided=true)', the length of the input and output "
                    << "at the last axis must satisfy (ni/2)+1=no.\n";
      return KRET_RESIZE_FAILED;
    }
  } else {
    if (x_shape_.back() != y_shape_.back()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "(rfft, onesided=false)', the length of the input and output "
                    << "at the last axis must be same.\n";
      return KRET_RESIZE_FAILED;
    }
  }
  if (is_onesided_) {
    workspace_size_list_ = {};
  } else {
    // cast float(input) to complex(workspace), and C2C transform
    MS_EXCEPTION_IF_CHECK_FAIL(!output_size_list_.empty(), "output_size_list_ must be not empty!");
    workspace_size_list_ = {output_size_list_[0]};
  }
  if (!MakeCufftPlan(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return ret;
}

int FFTWithSizeGpuKernelMod::ResizeIRFFT(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  int ret = ResizeBase(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  if (is_onesided_) {
    if (x_shape_.back() != (y_shape_.back() / K_HALF) + 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "(irfft)', the length of the input and output "
                    << "at the last axis should satisfy ni=(no/2)+1.\n";
      return KRET_RESIZE_FAILED;
    }
  } else {
    if (x_shape_.back() != y_shape_.back()) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "(irfft, onesided=false)', the length of the input and output "
                    << "at the last axis must be same.\n";
      return KRET_RESIZE_FAILED;
    }
  }
  // onesided: true
  // Chapter 2.4. of cufft's documentation(https://docs.nvidia.com/cuda/cufft/index.html#data-layout) says that
  // Out-of-place complex-to-real FFT will always overwrite input buffer.
  // We copy input buffer to avoid cufft overwriting, while complex-to-real.
  // onesided: false
  // C2C transform, and cast complex(workspace) to float(output).
  MS_EXCEPTION_IF_CHECK_FAIL(!inputs.empty(), "The inputs must be not empty!");
  workspace_size_list_ = {inputs[0]->size()};
  if (!MakeCufftPlan(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  return ret;
}

template <typename T>
bool FFTWithSizeGpuKernelMod::LaunchFFT(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (!IsValidShape(x_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the shape of output is invalid, since all the inputs are not ready.";
    return false;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto x_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  if (x_ptr == nullptr || y_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the address of output or input is nullptr.";
    return false;
  }
  auto status =
    CalculateFFT(x_ptr, y_ptr, scale_factor_, y_elements_, cufft_plan_, scale_plan_, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaGetLastError(),
                                           "For '" << kernel_name_ << "', it failed to CalculateFFT.");
  return true;
}

template <typename T>
bool FFTWithSizeGpuKernelMod::LaunchIFFT(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (!IsValidShape(x_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the shape of output is invalid, since all the inputs are not ready.";
    return false;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto x_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  if (x_ptr == nullptr || y_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the address of output or input is nullptr.";
    return false;
  }
  auto status =
    CalculateIFFT(x_ptr, y_ptr, scale_factor_, y_elements_, cufft_plan_, scale_plan_, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaGetLastError(),
                                           "For '" << kernel_name_ << "', it failed to CalculateIFFT.");
  return true;
}

template <typename S, typename T>
bool FFTWithSizeGpuKernelMod::LaunchRFFT(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (!IsValidShape(x_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the shape of output is invalid, since all the inputs are not ready.";
    return false;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto x_ptr = GetDeviceAddress<S>(inputs, kIndex0);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  if (x_ptr == nullptr || y_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the address of output or input is nullptr.";
    return false;
  }
  T *w_ptr = nullptr;
  if (!is_onesided_) {
    w_ptr = GetDeviceAddress<T>(workspace, kIndex0);
    if (w_ptr == nullptr) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "(rfft, onesided=false)', the address of workspace is nullptr.";
      return false;
    }
  }
  auto status = CalculateRFFT(x_ptr, w_ptr, y_ptr, is_onesided_, scale_factor_, x_elements_, y_elements_, cufft_plan_,
                              scale_plan_, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaGetLastError(),
                                           "For '" << kernel_name_ << "', it failed to CalculateRFFT.");
  return true;
}
template <typename S, typename T>
bool FFTWithSizeGpuKernelMod::LaunchIRFFT(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (!IsValidShape(x_shape_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', "
                  << "the shape of output is invalid, since all the inputs are not ready.";
    return false;
  }
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  auto x_ptr = GetDeviceAddress<S>(inputs, kIndex0);
  auto w_ptr = GetDeviceAddress<S>(workspace, kIndex0);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  if (x_ptr == nullptr || w_ptr == nullptr || y_ptr == nullptr) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the address of output or input is nullptr.";
    return false;
  }
  auto status = CalculateIRFFT(x_ptr, w_ptr, y_ptr, is_onesided_, scale_factor_, x_elements_, y_elements_, cufft_plan_,
                               scale_plan_, device_id_, cuda_stream);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaGetLastError(),
                                           "For '" << kernel_name_ << "', it failed to CalculateIRFFT.");
  return true;
}

#define FFT_GPU_REG(MS_I, MS_O)                                           \
  KernelAttr()                                                            \
    .AddInputAttr(MS_I)                                /* x */            \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* signal_ndim */  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* inverse */      \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* real */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* norm */         \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* onesided */     \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  /* signal_sizes */ \
    .AddOutputAttr(MS_O)

std::vector<KernelAttr> FFTWithSizeGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    FFT_GPU_REG(kNumberTypeFloat32, kNumberTypeComplex64),     // R2C, rfft
    FFT_GPU_REG(kNumberTypeFloat64, kNumberTypeComplex128),    // D2Z
    FFT_GPU_REG(kNumberTypeComplex64, kNumberTypeFloat32),     // C2R, irfft
    FFT_GPU_REG(kNumberTypeComplex128, kNumberTypeFloat64),    // Z2D
    FFT_GPU_REG(kNumberTypeComplex64, kNumberTypeComplex64),   // C2C, fft & ifft
    FFT_GPU_REG(kNumberTypeComplex128, kNumberTypeComplex128)  // Z2Z
  };
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, FFTWithSize, FFTWithSizeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

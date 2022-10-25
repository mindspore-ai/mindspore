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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_

#include <cmath>
#include <limits>
#include <string>

namespace mindspore {
namespace kernel {
enum KernelType { Lanczos1, Lanczos3, Lanczos5, Gaussian, Box, Triangle, KeysCubic, MitchellCubic, TypeEnd };
KernelType KernelTypeFromString(const std::string &str);
static constexpr float kRValue0 = 0.0f;
static constexpr float kRValue1 = 1.0f;
static constexpr float kRValue2 = 2.0f;

struct ComputerLanczosKernel {
  explicit ComputerLanczosKernel(float _radius) : radius(_radius) {}
  float operator()(float input) const {
    constexpr float PI = 3.14159265359;
    input = std::abs(input);
    if (input > radius) {
      return 0.0f;
    }
    // Need to special case the limit case of sin(input) / input when input is zero.
    if (input <= 1e-3) {
      return 1.0f;
    }
    return radius * std::sin(PI * input) * std::sin(PI * input / radius) / (PI * PI * input * input);
  }
  float Radius() const { return radius; }
  const float radius;
};

struct ComputerGaussianKernel {
  static constexpr float kRadiusMultiplier = 3.0f;
  /**
   * https://en.wikipedia.org/wiki/Gaussian_function
   * We use sigma = 0.5, as suggested on p. 4 of Ken Turkowski's "Filters
   * for Common Resampling Tasks" for kernels with a support of 3 pixels:
   * www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
   * This implies a radius of 1.5,
   */
  explicit ComputerGaussianKernel(float _radius = 1.5f) : radius(_radius), sigma(_radius / kRadiusMultiplier) {}
  float operator()(float input) const {
    input = std::abs(input);
    if (input >= radius) {
      return 0.0;
    }
    return std::exp(-input * input / (2.0f * sigma * sigma));
  }
  float Radius() const { return radius; }
  const float radius;
  // Gaussian standard deviation
  const float sigma;
};

struct ComputerBoxKernel {
  float operator()(float input) const {
    float result;
    input = std::abs(input);
    if (input < 0.5f) {
      result = kRValue1;
    } else if (std::fabs(input - 0.5f) <= std::numeric_limits<float>::epsilon()) {
      result = 0.5f;
    } else {
      result = kRValue0;
    }
    return result;
  }
  float Radius() const { return kRValue1; }
};

struct ComputetTriangleKernel {
  // https://en.wikipedia.org/wiki/Triangle_function
  float operator()(float input) const {
    float result;
    input = std::abs(input);
    if (input < kRValue1) {
      result = kRValue1 - input;
    } else {
      result = kRValue0;
    }
    return result;
  }
  float Radius() const { return kRValue1; }
};

struct ComputerKeysCubicKernel {
  /**
   * http://ieeexplore.ieee.org/document/1163711/
   * R. G. Keys. Cubic convolution interpolation for digital image
   * processing. IEEE Transactions on Acoustics, Speech, and Signal
   * Processing, 29(6):1153–1160, 1981.
   */
  float operator()(float input) const {
    input = std::abs(input);
    float result;
    if (input >= kRValue2) {
      result = kRValue0;
    } else if (input >= kRValue1) {
      result = -0.5f * input + 2.5f;
      result = result * input - 4.0f;
      result = result * input + kRValue2;
    } else {
      result = (1.5f * input - 2.5f) * input;
      result = result * input + kRValue1;
    }
    return result;
  }
  float Radius() const { return kRValue2; }
};

struct ComputerMitchellCubicKernel {
  /**
   * https://doi.org/10.1145/378456.378514
   * D. P. Mitchell and A. N. Netravali. Reconstruction filters in computer
   * graphics.  Computer Graphics (Proceedings of ACM SIGGRAPH 1988),
   * 22(4):221–228, 1988.
   */
  float operator()(float input) const {
    input = std::abs(input);
    if (input >= 2.0f) {
      return 0.0f;
    } else if (input >= 1.0f) {
      return (((-7.0f / 18.0f) * input + 2.0f) * input - 10.0f / 3.0f) * input + 16.0f / 9.0f;
    } else {
      return (((7.0f / 6.0f) * input - 2.0f) * input) * input + 8.0f / 9.0f;
    }
  }
  float Radius() const { return 2.f; }
};

inline ComputerLanczosKernel CreateLanczos1Kernel() { return ComputerLanczosKernel(1.0f); }

inline ComputerLanczosKernel CreateLanczos3Kernel() { return ComputerLanczosKernel(3.0f); }

inline ComputerLanczosKernel CreateLanczos5Kernel() { return ComputerLanczosKernel(5.0f); }

inline ComputerGaussianKernel CreateGaussianKernel() { return ComputerGaussianKernel(1.5f); }

inline ComputerBoxKernel CreateBoxKernel() { return ComputerBoxKernel(); }

inline ComputetTriangleKernel CreateTriangleKernel() { return ComputetTriangleKernel(); }

inline ComputerKeysCubicKernel CreateKeysCubicKernel() { return ComputerKeysCubicKernel(); }

inline ComputerMitchellCubicKernel CreateMitchellCubicKernel() { return ComputerMitchellCubicKernel(); }
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UTILS_SAMPLING_KERNELS_H_

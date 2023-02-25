/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/arithmetic_self_cpu_kernel.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/mkldnn/eltwise_cpu_kernel.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

constexpr float kMaxNegSerialSize = 5000.0f;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

constexpr auto kSquare = "Square";
constexpr auto kNeg = "Neg";
constexpr auto kSign = "Sign";
constexpr auto kFloor = "Floor";
constexpr auto kRint = "Rint";
constexpr auto kRound = "Round";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kInv = "Inv";
constexpr auto kInvert = "Invert";
constexpr auto kGeLU = "GeLU";
constexpr auto kLogicalNot = "LogicalNot";
constexpr auto kAsin = "Asin";
constexpr auto kACos = "ACos";
constexpr auto kAtan = "Atan";
constexpr auto kSin = "Sin";
constexpr auto kCos = "Cos";
constexpr auto kTan = "Tan";
constexpr auto kLog = "Log";
constexpr auto kExp = "Exp";
constexpr auto kSinh = "Sinh";
constexpr auto kCosh = "Cosh";
constexpr auto kTanh = "Tanh";
constexpr auto kAsinh = "Asinh";
constexpr auto kAcosh = "Acosh";
constexpr auto kAtanh = "Atanh";
constexpr auto kAbs = "Abs";
constexpr auto kSqrt = "Sqrt";
constexpr auto kRsqrt = "Rsqrt";
constexpr auto kErf = "Erf";
constexpr auto kErfc = "Erfc";
constexpr auto kSoftsign = "Softsign";
constexpr auto kReLU = "ReLU";

class ArithmeticSelfCpuKernelFunc : public CpuKernelFunc {
 public:
  ArithmeticSelfCpuKernelFunc() = default;
  ~ArithmeticSelfCpuKernelFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void LaunchLogicalNot(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  void LaunchKernelComplex(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  void LaunchKernelFloat16(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  std::string kernel_name_{kUnknown};
  TypeId dtype_{kTypeUnknown};
};

template <typename T>
void Square(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(in[i] * in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sign(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    auto zero = static_cast<T>(0);
    for (size_t i = start; i < end; i++) {
      if (in[i] < zero) {
        out[i] = static_cast<T>(-1);
      } else if (in[i] > zero) {
        out[i] = static_cast<T>(1);
      } else {
        out[i] = zero;
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Neg(ArithmeticSelfCpuKernelFunc *, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = -in[i];
    }
  };
  ParallelLaunch(task, size, kMaxNegSerialSize);
}

void LogicalNot(ArithmeticSelfCpuKernelFunc *content, const bool *in, bool *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = !in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Floor(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(floor(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Rint(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  if constexpr ((std::is_same_v<T, float16>)) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<T>(rint(static_cast<float>(in[i])));
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<T>(rint(in[i]));
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  }
}

template <typename T>
void Round(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(nearbyint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Reciprocal(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr ((std::is_same_v<T, double>) || (std::is_same_v<T, float>) || (std::is_same_v<T, complex64>) ||
                    (std::is_same_v<T, complex128>)) {
        T one = static_cast<T>(1.0);
        out[i] = static_cast<T>(one / in[i]);
      } else {
        out[i] = static_cast<T>(1.0 / in[i]);
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Inv(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  Reciprocal<T>(content, in, out, size);
}

template <typename T>
void Invert(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  if constexpr ((std::is_same_v<T, double>) || (std::is_same_v<T, float>) || (std::is_same_v<T, float16>)) {
    MS_LOG(EXCEPTION) << "'Invert' cannot be instantiated.";
  } else {
    auto task = [&in, &out](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = ~in[i];
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  }
}

template <typename T>
void Gelu(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    auto factor_a = static_cast<T>(0.7978845608);
    auto factor_b = static_cast<T>(0.044715);
    for (size_t i = start; i < end; i++) {
      T x = in[i];
      auto double_x = static_cast<T>(x);
      T tanh_res = static_cast<T>(std::tanh(factor_a * (double_x + factor_b * double_x * double_x * double_x)));
      out[i] = x * (static_cast<T>(1.0) + tanh_res) / static_cast<T>(2.0);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Asin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<T>(asin(in[i]));
      }
    } else {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<T>(asin(static_cast<double>(in[i])));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ACos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atan(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAtan(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(atan(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Erf(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(erf(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Erfc(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(erfc(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexSin(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sin(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexSign(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (in[i] != static_cast<T>(0)) {
        out[i] = (in[i] / abs(in[i]));
      } else {
        out[i] = static_cast<T>(0);
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexSinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexCos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cos(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexACos(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acos(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexCosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Exp(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(exp(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexExp(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(exp(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Tan(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same<T, float>::value) {
        out[i] = static_cast<T>(tan(static_cast<double>(in[i])));
      } else {
        out[i] = static_cast<T>(tan(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Cosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(cosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Tanh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same<T, float>::value) {
        out[i] = static_cast<T>(tanh(static_cast<double>(in[i])));
      } else {
        out[i] = static_cast<T>(tanh(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAsinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Asinh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(asinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexAcosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Acosh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(acosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Atanh(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same<T, float>::value) {
        out[i] = static_cast<T>(atanh(static_cast<double>(in[i])));
      } else {
        out[i] = static_cast<T>(atanh(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Abs(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  if constexpr ((std::is_same_v<T, uint8_t>) || (std::is_same_v<T, uint16_t>) || (std::is_same_v<T, uint32_t>) ||
                (std::is_same_v<T, uint64_t>)) {
    MS_LOG(EXCEPTION) << "'Abs' cannot be instantiated.";
  } else {
    auto task = [&in, &out](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = abs(in[i]);
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  }
}

template <typename T>
void Log(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(log(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void ComplexLog(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(log(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Sqrt(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(sqrt(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Rsqrt(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(1) / sqrt(in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Softsign(ArithmeticSelfCpuKernelFunc *, const T *in, T *out, size_t size) {
  if constexpr ((std::is_same_v<T, uint8_t>) || (std::is_same_v<T, uint16_t>) || (std::is_same_v<T, uint32_t>) ||
                (std::is_same_v<T, uint64_t>)) {
    MS_LOG(EXCEPTION) << "'Softsign' cannot be instantiated.";
  } else {
    if constexpr (std::is_same_v<T, float>) {
      auto task = [&in, &out](size_t start, size_t end) {
        auto length = SizeToInt(end - start);
        (void)SoftsignFp32Opt(in + start, length, out + start);
      };
      constexpr float min_batch_size = 5000;
      ParallelLaunch(task, size, min_batch_size);
    } else {
      auto task = [&in, &out](size_t start, size_t end) {
        for (size_t i = start; i < end; i++) {
          out[i] = in[i] / (static_cast<T>(1.0) + std::abs(in[i]));
        }
      };
      constexpr float min_batch_size = 1024;
      ParallelLaunch(task, size, min_batch_size);
    }
  }
}

template <typename T>
void Relu(ArithmeticSelfCpuKernelFunc *content, const T *in, T *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = std::greater<T>()(in[i], 0) ? in[i] : 0;
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

void ArithmeticSelfCpuKernelFunc::InitFunc(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &) {
  kernel_name_ = base_operator->name();
  dtype_ = inputs.at(kIndex0)->GetDtype();
}

bool ArithmeticSelfCpuKernelFunc::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat16(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex64) {
    LaunchKernelComplex<std::complex<float>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeComplex128) {
    LaunchKernelComplex<std::complex<double>>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt8) {
    LaunchKernel<int8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt16) {
    LaunchKernel<int16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt8) {
    LaunchKernel<uint8_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt16) {
    LaunchKernel<uint16_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt32) {
    LaunchKernel<uint32_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeUInt64) {
    LaunchKernel<uint64_t>(inputs, outputs);
  } else if (dtype_ == kNumberTypeBool) {
    LaunchLogicalNot(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the type of 'x' must be float16, float32, float64, complex64, complex128, int8, int16, "
                         "int32, int64, uint8, uint16, uint32, uint64, or bool, but got "
                      << TypeIdLabel(dtype_);
  }
  return true;
}

void ArithmeticSelfCpuKernelFunc::LaunchLogicalNot(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &outputs) {
  auto *input = reinterpret_cast<bool *>(inputs[0]->addr);
  auto *output = reinterpret_cast<bool *>(outputs[0]->addr);
  size_t lens = outputs[0]->size / sizeof(bool);
  LogicalNot(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelFunc::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelFunc *, const T *, T *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T>},
                          {prim::kPrimSign->name(), Sign<T>},
                          {prim::kPrimNeg->name(), Neg<T>},
                          {prim::kPrimAtanh->name(), Atanh<T>},
                          {prim::kPrimAcosh->name(), Acosh<T>},
                          {prim::kPrimFloor->name(), Floor<T>},
                          {prim::kPrimSin->name(), Sin<T>},
                          {prim::kPrimGeLU->name(), Gelu<T>},
                          {prim::kPrimCos->name(), Cos<T>},
                          {prim::kPrimLog->name(), Log<T>},
                          {prim::kPrimTan->name(), Tan<T>},
                          {prim::kPrimAsin->name(), Asin<T>},
                          {prim::kPrimACos->name(), ACos<T>},
                          {prim::kPrimAtan->name(), Atan<T>},
                          {prim::kPrimSinh->name(), Sinh<T>},
                          {prim::kPrimCosh->name(), Cosh<T>},
                          {prim::kPrimTanh->name(), Tanh<T>},
                          {prim::kPrimAsinh->name(), Asinh<T>},
                          {prim::kPrimReciprocal->name(), Reciprocal<T>},
                          {prim::kPrimInv->name(), Inv<T>},
                          {prim::kPrimInvert->name(), Invert<T>},
                          {prim::kPrimRint->name(), Rint<T>},
                          {prim::kPrimRound->name(), Round<T>},
                          {prim::kPrimAbs->name(), Abs<T>},
                          {prim::kPrimSqrt->name(), Sqrt<T>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T>},
                          {prim::kPrimErf->name(), Erf<T>},
                          {prim::kPrimErfc->name(), Erfc<T>},
                          {prim::kPrimSoftsign->name(), Softsign<T>},
                          {prim::kPrimReLU->name(), Relu<T>},
                          {prim::kPrimExp->name(), Exp<T>}};

  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION)
      << "For 'ArithmeticSelf', only supports operators in "
      << Map2Str<std::unordered_map, std::function<void(ArithmeticSelfCpuKernelFunc *, const T *, T *, size_t)>>(
           arithmeticSelfFuncMap)
      << ", but got " << kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

template <typename T>
void ArithmeticSelfCpuKernelFunc::LaunchKernelComplex(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(T);
  static const std::unordered_map<std::string,
                                  std::function<void(ArithmeticSelfCpuKernelFunc *, const T *, T *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T>},
                          {prim::kPrimAcosh->name(), ComplexAcosh<T>},
                          {prim::kPrimAsinh->name(), ComplexAsinh<T>},
                          {prim::kPrimNeg->name(), Neg<T>},
                          {prim::kPrimSinh->name(), ComplexSinh<T>},
                          {prim::kPrimCosh->name(), ComplexCosh<T>},
                          {prim::kPrimSin->name(), ComplexSin<T>},
                          {prim::kPrimCos->name(), ComplexCos<T>},
                          {prim::kPrimACos->name(), ComplexACos<T>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T>},
                          {prim::kPrimReciprocal->name(), Reciprocal<T>},
                          {prim::kPrimSqrt->name(), Sqrt<T>},
                          {prim::kPrimTan->name(), Tan<T>},
                          {prim::kPrimAtan->name(), ComplexAtan<T>},
                          {prim::kPrimTanh->name(), Tanh<T>},
                          {prim::kPrimAtanh->name(), Atanh<T>},
                          {prim::kPrimInv->name(), Inv<T>},
                          {prim::kPrimSign->name(), ComplexSign<T>},
                          {prim::kPrimLog->name(), ComplexLog<T>},
                          {prim::kPrimExp->name(), ComplexExp<T>},
                          {prim::kPrimAsin->name(), Asin<T>}};
  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'ArithmeticSelf', it does not support " << kernel_name_ << " with complex as input. ";
  }
  func_pair->second(this, input, output, lens);
}

void ArithmeticSelfCpuKernelFunc::LaunchKernelFloat16(const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &outputs) {
  const auto *input = static_cast<float16 *>(inputs[0]->addr);
  auto *output = static_cast<float16 *>(outputs[0]->addr);
  const size_t lens = outputs[0]->size / sizeof(float16);
  static const std::unordered_map<
    std::string, std::function<void(ArithmeticSelfCpuKernelFunc *, const float16 *, float16 *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimNeg->name(), Neg<float16>},     {prim::kPrimAcosh->name(), Acosh<float16>},
                          {prim::kPrimSin->name(), Sin<float16>},     {prim::kPrimCos->name(), Cos<float16>},
                          {prim::kPrimAsin->name(), Asin<float16>},   {prim::kPrimACos->name(), ACos<float16>},
                          {prim::kPrimSinh->name(), Sinh<float16>},   {prim::kPrimCosh->name(), Cosh<float16>},
                          {prim::kPrimAsinh->name(), Asinh<float16>}, {prim::kPrimErfc->name(), Erfc<float16>},
                          {prim::kPrimRsqrt->name(), Rsqrt<float16>}, {prim::kPrimErf->name(), Erf<float16>},
                          {prim::kPrimSign->name(), Sign<float16>},   {prim::kPrimRint->name(), Rint<float16>},
                          {prim::kPrimAtan->name(), Atan<float16>}};
  const auto func_pair = arithmeticSelfFuncMap.find(kernel_name_);
  if (arithmeticSelfFuncMap.find(kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'ArithmeticSelf', it does not support " << kernel_name_ << " with float16 as input. ";
  }
  func_pair->second(this, input, output, lens);
}

// MKLDNN Sqrt
class SqrtMKLKernelFunc : public CpuKernelFunc, private EltWiseCpuKernelMod {
 public:
  SqrtMKLKernelFunc() : EltWiseCpuKernelMod(kSqrt) {}
  ~SqrtMKLKernelFunc() override = default;

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    if (!EltWiseCpuKernelMod::Init(base_operator, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "For 'Sqrt', init failed.";
    }
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    // The Resize of EltWiseCpuKernelMod must be called here.
    return EltWiseCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    return EltWiseCpuKernelMod::Launch(inputs, workspace, outputs);
  }
};

// MKLDNN Log
class LogMKLKernelFunc : public CpuKernelFunc, private EltWiseCpuKernelMod {
 public:
  LogMKLKernelFunc() : EltWiseCpuKernelMod(kLog) {}
  ~LogMKLKernelFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    if (!EltWiseCpuKernelMod::Init(base_operator, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "For 'Log', init failed.";
    }
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    // The Resize of EltWiseCpuKernelMod must be called here.
    return EltWiseCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    return EltWiseCpuKernelMod::Launch(inputs, workspace, outputs);
  }
};

// MKLDNN Exp
class ExpMKLKernelFunc : public CpuKernelFunc, private EltWiseCpuKernelMod {
 public:
  ExpMKLKernelFunc() : EltWiseCpuKernelMod(kExp) {}
  ~ExpMKLKernelFunc() override = default;
  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    if (!EltWiseCpuKernelMod::Init(base_operator, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "For 'Exp', init failed.";
    }
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    // The Resize of EltWiseCpuKernelMod must be called here.
    return EltWiseCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    return EltWiseCpuKernelMod::Launch(inputs, workspace, outputs);
  }
};

// MKLDNN Tanh
class TanhMKLKernelFunc : public CpuKernelFunc, private EltWiseCpuKernelMod {
 public:
  TanhMKLKernelFunc() : EltWiseCpuKernelMod(kTanh) {}
  ~TanhMKLKernelFunc() override = default;

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    if (!EltWiseCpuKernelMod::Init(base_operator, inputs, outputs)) {
      MS_LOG(EXCEPTION) << "For 'Tanh', init failed.";
    }
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    // The Resize of EltWiseCpuKernelMod must be called here.
    return EltWiseCpuKernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  }

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override {
    return EltWiseCpuKernelMod::Launch(inputs, workspace, outputs);
  }
};

std::shared_ptr<CpuKernelFunc> CreateArithSelfFunc() { return std::make_shared<ArithmeticSelfCpuKernelFunc>(); }
using ArithFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithFuncCreator>>> arith_kernel_attr_list_map = {
  {kRsqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSquare,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kNeg,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSign,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kFloor,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kRint,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kRound,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kReciprocal,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kInv,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kInvert,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), CreateArithSelfFunc}}},
  {kGeLU,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kLogicalNot, {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), CreateArithSelfFunc}}},
  {kAsin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kACos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kAtan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kCos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kTan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kSinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kCosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kTanh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<TanhMKLKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kAsinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kAcosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kAtanh,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kAbs,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kSqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<SqrtMKLKernelFunc>(); }},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc}}},
  {kLog,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<LogMKLKernelFunc>(); }}}},
  {kErf,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kErfc,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc}}},
  {kSoftsign,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc}}},
  {kReLU,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), CreateArithSelfFunc}}},
  {kExp,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), CreateArithSelfFunc},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     []() { return std::make_shared<ExpMKLKernelFunc>(); }}}}};
}  // namespace

bool ArithmeticSelfCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = arith_kernel_attr_list_map.find(kernel_name_);
  if (iter == arith_kernel_attr_list_map.end()) {
    MS_LOG(ERROR) << "For 'ArithmeticSelf', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ArithFuncCreator>>>(
                       arith_kernel_attr_list_map)
                  << ", but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'ArithmeticSelf', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = arith_kernel_attr_list_map[kernel_name_][index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int ArithmeticSelfCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs,
                                       const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Note: This is to call the Resize of SqrtMKLKernelFunc.
  if (int ret = func_obj_->Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

std::vector<KernelAttr> ArithmeticSelfCpuKernelMod::GetOpSupport() {
  auto iter = arith_kernel_attr_list_map.find(kernel_name_);
  if (iter == arith_kernel_attr_list_map.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic self cpu does not support " << kernel_name_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArithFuncCreator> &pair) { return pair.first; });

  return support_list;
}

bool IdentityCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != prim::kIdentity) {
    MS_LOG(ERROR) << "For 'Identity', the kernel name must be 'Identity', but got " << kernel_name_;
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int IdentityCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

template <typename T>
bool IdentityCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->addr);
  T *output = reinterpret_cast<T *>(outputs[0]->addr);
  size_t input_num = inputs[0]->size / sizeof(T);
  auto task = [input, output](size_t start, size_t end) {
    for (size_t i = 0; i < end; i++) {
      output[i] = input[i];
    }
  };
  ParallelLaunchAutoSearch(task, input_num, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, IdentityCpuKernelMod::IdentityFunc>> IdentityCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &IdentityCpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &IdentityCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &IdentityCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &IdentityCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &IdentityCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &IdentityCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &IdentityCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &IdentityCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &IdentityCpuKernelMod::LaunchKernel<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &IdentityCpuKernelMod::LaunchKernel<complex128>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &IdentityCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &IdentityCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &IdentityCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &IdentityCpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> IdentityCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IdentityFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Identity, IdentityCpuKernelMod);

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Rsqrt,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRsqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Square,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSquare); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Neg,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kNeg); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sign,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSign); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Floor,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kFloor); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Rint,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRint); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Round,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kRound); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Reciprocal,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kReciprocal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Inv,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kInv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Invert,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kInvert); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, GeLU,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kGeLU); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LogicalNot,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kLogicalNot); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Asin,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAsin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ACos,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kACos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Atan,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAtan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sin,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cos,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kCos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Tan,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kTan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Exp,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kExp); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sinh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Cosh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kCosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Tanh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kTanh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Asinh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAsinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Acosh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAcosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Atanh,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAtanh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Abs,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kAbs); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sqrt,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Log,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kLog); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Erf,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kErf); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Erfc,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kErfc); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Softsign,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kSoftsign); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReLU,
                                 []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(kReLU); });
}  // namespace kernel
}  // namespace mindspore

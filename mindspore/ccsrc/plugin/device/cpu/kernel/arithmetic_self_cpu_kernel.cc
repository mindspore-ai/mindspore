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
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include "ops/array_ops.h"
#include "ops/framework_ops.h"
#include "ops/lite_ops.h"
#include "ops/math_ops.h"
#include "ops/nn_ops.h"
#include "ops/nn_optimizer_ops.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"
#include "nnacl/fp32/exp_fp32.h"
#include "ops/op_utils.h"
#include "ops/auto_generate/gen_ops_name.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = complex64;
using complex128 = complex128;

constexpr float kMaxNegSerialSize = 5000.0f;
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 1;

constexpr auto kSquare = "Square";
constexpr auto kNeg = "Neg";
constexpr auto kSign = "Sign";
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
constexpr auto kSoftsign = "Softsign";
constexpr auto kReLU = "ReLU";
constexpr auto kReLU6 = "ReLU6";
constexpr auto kSoftplus = "Softplus";
constexpr auto kMish = "Mish";
constexpr auto kSigmoid = "Sigmoid";

template <typename T, typename S>
class ArithmeticSelfCpuKernelFunc : public CpuKernelFunc {
 public:
  ArithmeticSelfCpuKernelFunc() = default;
  ~ArithmeticSelfCpuKernelFunc() override = default;
  void InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                const std::vector<KernelTensor *> &outputs) override {
    kernel_name_ = primitive->name();
  }
  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
               const std::vector<KernelTensor *> &outputs) override {
    this->LaunchKernel(inputs, outputs);
    return true;
  }

 protected:
  std::string kernel_name_{kUnknown};
  virtual void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {}
};

template <typename T, typename S>
class ArithmeticSelfCpuKernelFuncComplex : public ArithmeticSelfCpuKernelFunc<T, S> {
 protected:
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
};

template <typename T, typename S>
class ArithmeticSelfCpuKernelFuncCommon : public ArithmeticSelfCpuKernelFunc<T, S> {
 protected:
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
};

template <typename T, typename S>
class ArithmeticSelfCpuKernelFuncBool : public ArithmeticSelfCpuKernelFunc<T, S> {
 protected:
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
};

template <typename T, typename S>
class ArithmeticSelfCpuKernelFuncFloat16 : public ArithmeticSelfCpuKernelFunc<T, S> {
 protected:
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
};

template <typename T, typename S>
void Square(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(in[i] * in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Sign(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    auto zero = static_cast<T>(0);
    for (size_t i = start; i < end; i++) {
      if (in[i] < zero) {
        out[i] = static_cast<S>(-1);
      } else if (in[i] > zero) {
        out[i] = static_cast<S>(1);
      } else {
        out[i] = zero;
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Neg(ArithmeticSelfCpuKernelFunc<T, S> *, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = -in[i];
    }
  };
  ParallelLaunch(task, size, kMaxNegSerialSize);
}

template <typename T, typename S>
void LogicalEqual(ArithmeticSelfCpuKernelFuncBool<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void LogicalNot(ArithmeticSelfCpuKernelFuncBool<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = !in[i];
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Floor(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(floor(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Rint(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  if constexpr ((std::is_same_v<T, float16>)) {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(rint(static_cast<float>(in[i])));
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(rint(in[i]));
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  }
}

template <typename T, typename S>
void Round(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(nearbyint(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Reciprocal(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr ((std::is_same_v<T, double>) || (std::is_same_v<T, float>) || (std::is_same_v<T, complex64>) ||
                    (std::is_same_v<T, complex128>)) {
        T one = static_cast<T>(1.0);
        out[i] = static_cast<S>(one / in[i]);
      } else {
        out[i] = static_cast<S>(1.0 / in[i]);
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Inv(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  Reciprocal<T, S>(content, in, out, size);
}

template <typename T, typename S>
void Invert(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
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

template <typename T, typename S>
void Gelu(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
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

template <typename T, typename S>
void Asin(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(asin(in[i]));
      }
    } else {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(asin(static_cast<double>(in[i])));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ACos(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(acos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Atan(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(atan(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexAtan(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(atan(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Sin(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(sin(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Cos(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(cos(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Erf(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float16>) {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(erf(static_cast<float>(in[i])));
      }
    } else {
      for (size_t i = start; i < end; i++) {
        out[i] = static_cast<S>(erf(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Erfc(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(erfc(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexSin(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(sin(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexSign(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (in[i] != static_cast<T>(0)) {
        out[i] = (in[i] / abs(in[i]));
      } else {
        out[i] = static_cast<S>(0);
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexSinh(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(sinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexCos(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(cos(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexACos(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(acos(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexCosh(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(cosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Exp(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      (void)::ExpFp32(in + start, out + start, end - start);
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(exp(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexExp(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(exp(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Tan(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same<T, float>::value) {
        out[i] = static_cast<S>(tan(static_cast<double>(in[i])));
      } else {
        out[i] = static_cast<S>(tan(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Sinh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(sinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Cosh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(cosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Tanh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr (std::is_same<T, float>::value) {
      (void)::Tanh(in + start, SizeToInt(end - start), out + start);
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(tanh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexAsinh(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(asinh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Asinh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(asinh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexAcosh(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(acosh(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Acosh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(acosh(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Atanh(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same<T, float>::value) {
        out[i] = static_cast<S>(atanh(static_cast<double>(in[i])));
      } else {
        out[i] = static_cast<S>(atanh(in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Abs(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  if constexpr ((std::is_same_v<T, uint8_t>) || (std::is_same_v<T, uint16_t>) || (std::is_same_v<T, uint32_t>) ||
                (std::is_same_v<T, uint64_t>)) {
    auto ret_code = memcpy_s(out, size * sizeof(T), in, size * sizeof(T));
    if (ret_code != EOK) {
      MS_LOG(EXCEPTION) << "For Abs, Failed to copy data, memcpy_s errorno: " << ret_code;
    }
  } else {
    auto task = [&in, &out](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = abs(in[i]);
      }
    };
    ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
  }
}

template <typename T, typename S>
void Log(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      auto ret = ::ElementLog(in + start, out + start, SizeToInt(end - start));
      if (ret == NNACL_ERRCODE_LOG_NEGATIVE_OR_ZERO) {
        for (size_t i = start; i < end; i++) {
          out[i] = static_cast<S>(log(static_cast<double>(in[i])));
        }
      }
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(log(static_cast<double>(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void ComplexLog(ArithmeticSelfCpuKernelFuncComplex<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(log(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Sqrt(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      auto ret = ::ElementSqrt(in + start, out + start, SizeToInt(end - start));
      if (ret == NNACL_ERRCODE_SQRT_NEGATIVE) {
        for (size_t i = start; i < end; i++) {
          out[i] = static_cast<S>(sqrt(in[i]));
        }
      }
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(sqrt(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Rsqrt(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [&in, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<S>(1) / sqrt(in[i]);
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Softsign(ArithmeticSelfCpuKernelFunc<T, S> *, const T *in, S *out, size_t size) {
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

template <typename T, typename S>
void Relu(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = std::greater<T>()(in[i], 0) ? in[i] : 0;
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Relu6(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      (void)::Fp32Relu6(in + start, SizeToInt(end - start), out + start);
      return;
    }
    constexpr T six = 6;
    constexpr T zero = 0;
    for (size_t i = start; i < end; i++) {
      if (std::less<T>()(in[i], zero)) {
        out[i] = zero;
      } else {
        out[i] = in[i] > six ? six : in[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Softplus(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      (void)::Softplus(in + start, SizeToInt(end - start), out + start);
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = std::log1p(std::exp(in[i]));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Mish(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    if constexpr (std::is_same_v<T, float>) {
      (void)::ElementMish(in + start, out + start, SizeToInt(end - start));
      return;
    }
    for (size_t i = start; i < end; i++) {
      out[i] = in[i] * std::tanh(std::log1p(std::exp(in[i])));
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T, typename S>
void Sigmoid(ArithmeticSelfCpuKernelFunc<T, S> *content, const T *in, S *out, size_t size) {
  auto task = [in, out](size_t start, size_t end) {
    if constexpr ((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>)) {
      constexpr T one_complex{1, 0};
      for (size_t i = start; i < end; i++) {
        out[i] = one_complex / (one_complex + std::exp(-in[i]));
      }
    } else if constexpr (std::is_same_v<T, float16>) {
      float16 one{1};
      for (size_t i = start; i < end; i++) {
        out[i] = one / (one + exp(-in[i]));
      }
    } else if constexpr (std::is_same_v<T, float>) {
      (void)::Sigmoid(in + start, SizeToInt(end - start), out + start);
    } else {
      constexpr T one = 1;
      for (size_t i = start; i < end; i++) {
        out[i] = one / (one + std::exp(-in[i]));
      }
    }
  };
  ParallelLaunchAutoSearch(task, size, content, &content->parallel_search_info_);
}

template <typename T>
void Identity(const T *in, T *out, size_t size) {
  (void)std::copy(in, in + size, out);
}

template <typename T>
bool IdentityCpuFunc(const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  T *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  T *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  size_t lens = outputs[0]->size() > 0 ? static_cast<size_t>(outputs[0]->size() / sizeof(T)) : 1;
  Identity<T>(input, output, lens);
  return true;
}

static std::vector<std::pair<KernelAttr, LaunchFunc>> identity_kernel_attr_lists = {
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), IdentityCpuFunc<uint64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), IdentityCpuFunc<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), IdentityCpuFunc<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), IdentityCpuFunc<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), IdentityCpuFunc<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), IdentityCpuFunc<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), IdentityCpuFunc<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), IdentityCpuFunc<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64), IdentityCpuFunc<complex64>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128), IdentityCpuFunc<complex128>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), IdentityCpuFunc<double>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), IdentityCpuFunc<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), IdentityCpuFunc<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), IdentityCpuFunc<bool>}};

template <typename T, typename S>
void ArithmeticSelfCpuKernelFuncBool<T, S>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *output = reinterpret_cast<S *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(S);
  if (this->kernel_name_ == kAbsOpName) {
    return LogicalEqual<T, S>(this, input, output, lens);
  } else {
    return LogicalNot<T, S>(this, input, output, lens);
  }
}

template <typename T, typename S>
void ArithmeticSelfCpuKernelFuncCommon<T, S>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *output = reinterpret_cast<S *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(S);
  static const std::unordered_map<
    std::string, std::function<void(ArithmeticSelfCpuKernelFuncCommon<T, S> *, const T *, S *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T, S>},
                          {prim::kPrimSign->name(), Sign<T, S>},
                          {prim::kPrimNeg->name(), Neg<T, S>},
                          {prim::kPrimAtanh->name(), Atanh<T, S>},
                          {prim::kPrimAcosh->name(), Acosh<T, S>},
                          {prim::kPrimFloor->name(), Floor<T, S>},
                          {prim::kPrimSin->name(), Sin<T, S>},
                          {prim::kPrimGeLU->name(), Gelu<T, S>},
                          {prim::kPrimCos->name(), Cos<T, S>},
                          {prim::kPrimLog->name(), Log<T, S>},
                          {prim::kPrimTan->name(), Tan<T, S>},
                          {prim::kPrimAsin->name(), Asin<T, S>},
                          {prim::kPrimACos->name(), ACos<T, S>},
                          {prim::kPrimAtan->name(), Atan<T, S>},
                          {prim::kPrimSinh->name(), Sinh<T, S>},
                          {prim::kPrimCosh->name(), Cosh<T, S>},
                          {prim::kPrimTanh->name(), Tanh<T, S>},
                          {prim::kPrimAsinh->name(), Asinh<T, S>},
                          {prim::kPrimReciprocal->name(), Reciprocal<T, S>},
                          {prim::kPrimInv->name(), Inv<T, S>},
                          {prim::kPrimInvert->name(), Invert<T, S>},
                          {prim::kPrimRint->name(), Rint<T, S>},
                          {prim::kPrimRound->name(), Round<T, S>},
                          {prim::kPrimAbs->name(), Abs<T, S>},
                          {prim::kPrimSqrt->name(), Sqrt<T, S>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T, S>},
                          {prim::kPrimErf->name(), Erf<T, S>},
                          {prim::kPrimErfc->name(), Erfc<T, S>},
                          {prim::kPrimSoftsign->name(), Softsign<T, S>},
                          {prim::kPrimReLU->name(), Relu<T, S>},
                          {prim::kPrimReLU6->name(), Relu6<T, S>},
                          {prim::kPrimSoftplus->name(), Softplus<T, S>},
                          {prim::kPrimMish->name(), Mish<T, S>},
                          {prim::kPrimSigmoid->name(), Sigmoid<T, S>},
                          {prim::kPrimExp->name(), Exp<T, S>}};

  const auto func_pair = arithmeticSelfFuncMap.find(this->kernel_name_);
  if (arithmeticSelfFuncMap.find(this->kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION)
      << "For 'ArithmeticSelf', only supports operators in "
      << Map2Str<std::unordered_map,
                 std::function<void(ArithmeticSelfCpuKernelFuncCommon<T, S> *, const T *, S *, size_t)>>(
           arithmeticSelfFuncMap)
      << ", but got " << this->kernel_name_;
  }
  func_pair->second(this, input, output, lens);
}

template <typename T, typename S>
void ArithmeticSelfCpuKernelFuncComplex<T, S>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *output = reinterpret_cast<S *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(S);
  static const std::unordered_map<
    std::string, std::function<void(ArithmeticSelfCpuKernelFuncComplex<T, S> *, const T *, S *, size_t)>>
    arithmeticSelfFuncMap{{prim::kPrimSquare->name(), Square<T, S>},
                          {prim::kPrimAcosh->name(), ComplexAcosh<T, S>},
                          {prim::kPrimAsinh->name(), ComplexAsinh<T, S>},
                          {prim::kPrimNeg->name(), Neg<T, S>},
                          {prim::kPrimSinh->name(), ComplexSinh<T, S>},
                          {prim::kPrimCosh->name(), ComplexCosh<T, S>},
                          {prim::kPrimSin->name(), ComplexSin<T, S>},
                          {prim::kPrimCos->name(), ComplexCos<T, S>},
                          {prim::kPrimACos->name(), ComplexACos<T, S>},
                          {prim::kPrimRsqrt->name(), Rsqrt<T, S>},
                          {prim::kPrimReciprocal->name(), Reciprocal<T, S>},
                          {prim::kPrimSqrt->name(), Sqrt<T, S>},
                          {prim::kPrimTan->name(), Tan<T, S>},
                          {prim::kPrimAtan->name(), ComplexAtan<T, S>},
                          {prim::kPrimTanh->name(), Tanh<T, S>},
                          {prim::kPrimAtanh->name(), Atanh<T, S>},
                          {prim::kPrimInv->name(), Inv<T, S>},
                          {prim::kPrimAbs->name(), Abs<T, S>},
                          {prim::kPrimSign->name(), ComplexSign<T, S>},
                          {prim::kPrimLog->name(), ComplexLog<T, S>},
                          {prim::kPrimExp->name(), ComplexExp<T, S>},
                          {prim::kPrimSigmoid->name(), Sigmoid<T, S>},
                          {prim::kPrimAsin->name(), Asin<T, S>}};
  const auto func_pair = arithmeticSelfFuncMap.find(this->kernel_name_);
  if (arithmeticSelfFuncMap.find(this->kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'ArithmeticSelf', it does not support " << this->kernel_name_
                      << " with complex as input. ";
  }
  func_pair->second(this, input, output, lens);
}

template <typename T, typename S>
void ArithmeticSelfCpuKernelFuncFloat16<T, S>::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  const auto *input = static_cast<T *>(inputs[0]->device_ptr());
  auto *output = static_cast<S *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(S);
  static const std::unordered_map<
    std::string, std::function<void(ArithmeticSelfCpuKernelFuncFloat16<T, S> *, const T *, S *, size_t)>>
    arithmeticSelfFuncMap{
      {prim::kPrimNeg->name(), Neg<T, S>},         {prim::kPrimAcosh->name(), Acosh<T, S>},
      {prim::kPrimSin->name(), Sin<T, S>},         {prim::kPrimCos->name(), Cos<T, S>},
      {prim::kPrimAsin->name(), Asin<T, S>},       {prim::kPrimACos->name(), ACos<T, S>},
      {prim::kPrimSinh->name(), Sinh<T, S>},       {prim::kPrimCosh->name(), Cosh<T, S>},
      {prim::kPrimAsinh->name(), Asinh<T, S>},     {prim::kPrimErfc->name(), Erfc<T, S>},
      {prim::kPrimRsqrt->name(), Rsqrt<T, S>},     {prim::kPrimErf->name(), Erf<T, S>},
      {prim::kPrimSign->name(), Sign<T, S>},       {prim::kPrimRint->name(), Rint<T, S>},
      {prim::kPrimAtan->name(), Atan<T, S>},       {prim::kPrimSqrt->name(), Sqrt<T, S>},
      {prim::kPrimSigmoid->name(), Sigmoid<T, S>},
    };
  const auto func_pair = arithmeticSelfFuncMap.find(this->kernel_name_);
  if (arithmeticSelfFuncMap.find(this->kernel_name_) == arithmeticSelfFuncMap.end()) {
    MS_LOG(EXCEPTION) << "For 'ArithmeticSelf', it does not support " << this->kernel_name_
                      << " with float16 as input. ";
  }
  func_pair->second(this, input, output, lens);
}

template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> CreateArithSelfFuncCommon() {
  return std::make_shared<ArithmeticSelfCpuKernelFuncCommon<T, S>>();
}
template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> CreateArithSelfFuncBool() {
  return std::make_shared<ArithmeticSelfCpuKernelFuncBool<T, S>>();
}
template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> CreateArithSelfFuncComplex() {
  return std::make_shared<ArithmeticSelfCpuKernelFuncComplex<T, S>>();
}
template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> CreateArithSelfFuncFloat16() {
  return std::make_shared<ArithmeticSelfCpuKernelFuncFloat16<T, S>>();
}
using ArithFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithFuncCreator>>> arith_kernel_attr_list_map = {
  {kRsqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kSquare,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CreateArithSelfFuncCommon<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &CreateArithSelfFuncCommon<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &CreateArithSelfFuncCommon<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &CreateArithSelfFuncCommon<uint64_t, uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kNeg,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &CreateArithSelfFuncCommon<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &CreateArithSelfFuncCommon<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &CreateArithSelfFuncCommon<uint64_t, uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CreateArithSelfFuncCommon<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kSign,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {ops::kNameFloor,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kRint,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kRound,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kReciprocal,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), &CreateArithSelfFuncBool<bool, bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<int64_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<int, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<int16_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<int8_t, float>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<int, float>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncBool<bool, float>}}},
  {kInv,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kInvert,
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CreateArithSelfFuncCommon<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &CreateArithSelfFuncCommon<uint16_t, uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &CreateArithSelfFuncCommon<uint32_t, uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &CreateArithSelfFuncCommon<uint64_t, uint64_t>}}},
  {kGeLU,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kLogicalNot,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), &CreateArithSelfFuncBool<bool, bool>}}},
  {kAsin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kACos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kAtan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kSin,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kCos,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kTan,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kSinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kCosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kTanh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kAsinh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kAcosh,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kAtanh,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kAbs,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), &CreateArithSelfFuncBool<bool, bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CreateArithSelfFuncCommon<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kSqrt,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>}}},
  {kLog,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}},
  {ops::kNameErf,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {ops::kNameErfc,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>}}},
  {kSoftsign,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}},
  {kReLU,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &CreateArithSelfFuncCommon<int64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), &CreateArithSelfFuncCommon<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &CreateArithSelfFuncCommon<int16_t, int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &CreateArithSelfFuncCommon<int8_t, int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &CreateArithSelfFuncCommon<uint8_t, uint8_t>}}},
  {kReLU6,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}},
  {kSoftplus,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}},
  {kMish,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}},
  {kSigmoid,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &CreateArithSelfFuncFloat16<float16, float16>}}},
  {kExp,
   {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &CreateArithSelfFuncCommon<double, double>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &CreateArithSelfFuncComplex<complex64, complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &CreateArithSelfFuncComplex<complex128, complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &CreateArithSelfFuncCommon<float, float>}}}};
}  // namespace

bool ArithmeticSelfCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
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
  func_obj_->InitFunc(primitive_, inputs, outputs);
  return true;
}

int ArithmeticSelfCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // Note: This is to call the Resize of SqrtMKLKernelFunc.
  if (int ret = func_obj_->Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
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

bool IdentityCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ != mindspore::kIdentityOpName) {
    MS_LOG(ERROR) << "For 'Identity', the kernel name must be 'Identity', but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = identity_kernel_attr_lists[index].second;
  return true;
}

int IdentityCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

std::vector<KernelAttr> IdentityCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> kernel_attr_list;
  (void)std::transform(identity_kernel_attr_lists.begin(), identity_kernel_attr_lists.end(),
                       std::back_inserter(kernel_attr_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return kernel_attr_list;
}

#define ARITHMETIC_SELF_CPU_REGISTER(FUNC, OP_NAME)          \
  MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FUNC, \
                                   []() { return std::make_shared<ArithmeticSelfCpuKernelMod>(OP_NAME); });

ARITHMETIC_SELF_CPU_REGISTER(Rsqrt, kRsqrt);
ARITHMETIC_SELF_CPU_REGISTER(Square, kSquare);
ARITHMETIC_SELF_CPU_REGISTER(Neg, kNeg);
ARITHMETIC_SELF_CPU_REGISTER(Sign, kSign);
ARITHMETIC_SELF_CPU_REGISTER(Floor, ops::kNameFloor);
ARITHMETIC_SELF_CPU_REGISTER(Rint, kRint);
ARITHMETIC_SELF_CPU_REGISTER(Round, kRound);
ARITHMETIC_SELF_CPU_REGISTER(Reciprocal, kReciprocal);
ARITHMETIC_SELF_CPU_REGISTER(Inv, kInv);
ARITHMETIC_SELF_CPU_REGISTER(Invert, kInvert);
ARITHMETIC_SELF_CPU_REGISTER(GeLU, kGeLU);
ARITHMETIC_SELF_CPU_REGISTER(LogicalNot, kLogicalNot);
ARITHMETIC_SELF_CPU_REGISTER(Asin, kAsin);
ARITHMETIC_SELF_CPU_REGISTER(ACos, kACos);
ARITHMETIC_SELF_CPU_REGISTER(Atan, kAtan);
ARITHMETIC_SELF_CPU_REGISTER(Sin, kSin);
ARITHMETIC_SELF_CPU_REGISTER(Cos, kCos);
ARITHMETIC_SELF_CPU_REGISTER(Tan, kTan);
ARITHMETIC_SELF_CPU_REGISTER(Exp, kExp);
ARITHMETIC_SELF_CPU_REGISTER(Sinh, kSinh);
ARITHMETIC_SELF_CPU_REGISTER(Cosh, kCosh);
ARITHMETIC_SELF_CPU_REGISTER(Tanh, kTanh);
ARITHMETIC_SELF_CPU_REGISTER(Asinh, kAsinh);
ARITHMETIC_SELF_CPU_REGISTER(Acosh, kAcosh);
ARITHMETIC_SELF_CPU_REGISTER(Atanh, kAtanh);
ARITHMETIC_SELF_CPU_REGISTER(Abs, kAbs);
ARITHMETIC_SELF_CPU_REGISTER(Sqrt, kSqrt);
ARITHMETIC_SELF_CPU_REGISTER(Log, kLog);
ARITHMETIC_SELF_CPU_REGISTER(Erf, ops::kNameErf);
ARITHMETIC_SELF_CPU_REGISTER(Erfc, ops::kNameErfc);
ARITHMETIC_SELF_CPU_REGISTER(Softsign, kSoftsign);
ARITHMETIC_SELF_CPU_REGISTER(ReLU, kReLU);
ARITHMETIC_SELF_CPU_REGISTER(ReLU6, kReLU6);
ARITHMETIC_SELF_CPU_REGISTER(Softplus, kSoftplus);
ARITHMETIC_SELF_CPU_REGISTER(Mish, kMish);
ARITHMETIC_SELF_CPU_REGISTER(Sigmoid, kSigmoid);

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Identity, IdentityCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore

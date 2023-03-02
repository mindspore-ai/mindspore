/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/math/unary_op_gpu_kernel.h"
#include <memory>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kAbs = "Abs";
constexpr auto kACos = "ACos";
constexpr auto kAcosh = "Acosh";
constexpr auto kAsin = "Asin";
constexpr auto kAsinh = "Asinh";
constexpr auto kAtan = "Atan";
constexpr auto kCos = "Cos";
constexpr auto kCosh = "Cosh";
constexpr auto kAtanh = "Atanh";
constexpr auto kErf = "Erf";
constexpr auto kErfc = "Erfc";
constexpr auto kExp = "Exp";
constexpr auto kExpm1 = "Expm1";
constexpr auto kFloor = "Floor";
constexpr auto kTrunc = "Trunc";
constexpr auto kCeil = "Ceil";
constexpr auto kLog = "Log";
constexpr auto kLog1p = "Log1p";
constexpr auto kNeg = "Neg";
constexpr auto kReciprocal = "Reciprocal";
constexpr auto kInv = "Inv";
constexpr auto kInvert = "Invert";
constexpr auto kRint = "Rint";
constexpr auto kRound = "Round";
constexpr auto kRsqrt = "Rsqrt";
constexpr auto kSign = "Sign";
constexpr auto kSin = "Sin";
constexpr auto kTan = "Tan";
constexpr auto kSinh = "Sinh";
constexpr auto kSqrt = "Sqrt";
constexpr auto kSquare = "Square";
}  // namespace

std::map<std::string, std::vector<std::pair<KernelAttr, UnaryOpGpuKernelMod::UnaryOpFunc>>>
  UnaryOpGpuKernelMod::kernel_attr_map_ = {
    {kExp,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kExpm1,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>}}},
    {kLog,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kLog1p,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kErf,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kErfc,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kNeg,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kReciprocal,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kInv,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kInvert,
     {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<uchar>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>}}},
    {kSquare,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kSqrt,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<uint16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<uint32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<uint64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kRsqrt,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kSin,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kSinh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kTan,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kAsin,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAsinh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kCos,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kCosh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kACos,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAcosh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAtan,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAtanh,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>}}},
    {kAbs,
     {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
       &UnaryOpGpuKernelMod::LaunchKernel<bool>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<unsigned char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
       &UnaryOpGpuKernelMod::LaunchKernel<int16_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>}}},
    {kFloor,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kTrunc,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<char>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
       &UnaryOpGpuKernelMod::LaunchKernel<uchar>}}},
    {kCeil,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kRint,
     {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kRound,
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}},
    {kSign,
     {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
       &UnaryOpGpuKernelMod::LaunchKernel<int32_t>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
       &UnaryOpGpuKernelMod::LaunchKernel<int64_t>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<float>>},
      {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
       &UnaryOpGpuKernelMod::LaunchKernel<utils::Complex<double>>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &UnaryOpGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &UnaryOpGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &UnaryOpGpuKernelMod::LaunchKernel<half>}}}};

bool UnaryOpGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'Unary op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, UnaryOpGpuKernelMod::UnaryOpFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return false;
  }
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;
  return true;
}

int UnaryOpGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  size_t input_element_num =
    std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_element_num == 0);
  if (is_null_input_) {
    return KRET_OK;
  }
  return KRET_OK;
}

std::vector<KernelAttr> UnaryOpGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR) << "For 'Unary op', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, UnaryOpGpuKernelMod::UnaryOpFunc>>>(
                       kernel_attr_map_)
                  << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UnaryOpFunc> &item) { return item.first; });
  return support_list;
}

template <typename T>
bool UnaryOpGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  std::map<std::string, std::function<void(const T *, T *, const size_t, cudaStream_t)>> func_map;

  const bool is_t_complex = (std::is_same_v<T, utils::Complex<float>>) || (std::is_same_v<T, utils::Complex<double>>);
  if constexpr (is_t_complex) {
    std::map<std::string, std::function<void(const T *, T *, const size_t, cudaStream_t)>> func_map_complex = {
      {kSqrt, SqrtOpt<T>}, {kTan, Tan<T>},     {kCosh, Cosh<T>},        {kAtanh, Atanh<T>},
      {kInv, InvOpt<T>},   {kLog, LogOpt<T>},  {kExp, ExpOpt<T>},       {kNeg, NegOpt<T>},
      {kSin, Sin<T>},      {kCos, Cos<T>},     {kACos, ACos<T>},        {kAcosh, Acosh<T>},
      {kAsin, Asin<T>},    {kAsinh, Asinh<T>}, {kSquare, SquareOpt<T>}, {kReciprocal, ReciprocalOpt<T>},
      {kRsqrt, Rsqrt<T>},  {kSign, Sign<T>},   {kAtan, Atan<T>},        {kSinh, Sinh<T>},
      {kExpm1, Expm1<T>},  {kLog1p, Log1p<T>}, {kAbs, Abs<T>}};
    copy(func_map_complex.begin(), func_map_complex.end(), inserter(func_map, func_map.begin()));
  } else {
    std::map<std::string, std::function<void(const T *, T *, const size_t, cudaStream_t)>> func_map_normal = {
      {kExp, ExpOpt<T>},  {kExpm1, Expm1<T>},   {kLog, LogOpt<T>},       {kLog1p, Log1p<T>},
      {kErf, Erf<T>},     {kErfc, Erfc<T>},     {kNeg, NegOpt<T>},       {kReciprocal, ReciprocalOpt<T>},
      {kInv, InvOpt<T>},  {kInvert, Invert<T>}, {kSquare, SquareOpt<T>}, {kSqrt, SqrtOpt<T>},
      {kRsqrt, Rsqrt<T>}, {kSin, Sin<T>},       {kCos, Cos<T>},          {kCosh, Cosh<T>},
      {kAsin, Asin<T>},   {kACos, ACos<T>},     {kAtan, Atan<T>},        {kAsinh, Asinh<T>},
      {kAcosh, Acosh<T>}, {kAbs, Abs<T>},       {kFloor, Floor<T>},      {kCeil, Ceil<T>},
      {kRint, Rint<T>},   {kRound, Round<T>},   {kSign, Sign<T>},        {kAtanh, Atanh<T>},
      {kTan, Tan<T>},     {kSinh, Sinh<T>},     {kTrunc, Trunc<T>}};
    copy(func_map_normal.begin(), func_map_normal.end(), inserter(func_map, func_map.begin()));
  }

  auto iter = func_map.find(kernel_name_);
  if (iter == func_map.end()) {
    MS_LOG(ERROR) << "For 'UnaryOp', only support these types: "
                  << kernel::Map2Str<std::map, std::function<void(const T *, T *, const size_t, cudaStream_t)>>(
                       func_map)
                  << " currently, but got " << kernel_name_;
    return false;
  }
  auto input_ptr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output_ptr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  iter->second(input_ptr, output_ptr, input_size_list_[0] / sizeof(T), reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Abs, []() { return std::make_shared<UnaryOpGpuKernelMod>(kAbs); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, ACos,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kACos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Acosh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAcosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Atanh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAtanh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Asin,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAsin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Asinh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAsinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Atan,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kAtan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Cos, []() { return std::make_shared<UnaryOpGpuKernelMod>(kCos); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Cosh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kCosh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Erf, []() { return std::make_shared<UnaryOpGpuKernelMod>(kErf); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Erfc,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kErfc); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Exp, []() { return std::make_shared<UnaryOpGpuKernelMod>(kExp); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Expm1,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kExpm1); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Floor,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kFloor); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Trunc,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kTrunc); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Ceil,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kCeil); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Log, []() { return std::make_shared<UnaryOpGpuKernelMod>(kLog); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Log1p,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kLog1p); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Neg, []() { return std::make_shared<UnaryOpGpuKernelMod>(kNeg); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Reciprocal,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kReciprocal); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Inv, []() { return std::make_shared<UnaryOpGpuKernelMod>(kInv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Invert,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kInvert); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Rint,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRint); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Round,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRound); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Rsqrt,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kRsqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sign,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSign); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sin, []() { return std::make_shared<UnaryOpGpuKernelMod>(kSin); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sinh,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSinh); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Tan, []() { return std::make_shared<UnaryOpGpuKernelMod>(kTan); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Sqrt,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSqrt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, Square,
                                 []() { return std::make_shared<UnaryOpGpuKernelMod>(kSquare); });
}  // namespace kernel
}  // namespace mindspore

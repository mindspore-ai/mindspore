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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_HELPER_H_
#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"

namespace mindspore {
namespace cukernel {
enum UnaryOptype {
  UNARY_OP_EXP = 0,
  UNARY_OP_EXPM1,
  UNARY_OP_LOG,
  UNARY_OP_LOG1P,
  UNARY_OP_ERF,
  UNARY_OP_ERFC,
  UNARY_OP_NEG,
  UNARY_OP_RECIPROCAL,
  UNARY_OP_INV,
  UNARY_OP_INVERT,
  UNARY_OP_SQUARE,
  UNARY_OP_SQRT,
  UNARY_OP_RSQRT,
  UNARY_OP_SIN,
  UNARY_OP_SINH,
  UNARY_OP_COS,
  UNARY_OP_TAN,
  UNARY_OP_COSH,
  UNARY_OP_ASIN,
  UNARY_OP_ACOS,
  UNARY_OP_ATAN,
  UNARY_OP_ASINH,
  UNARY_OP_ACOSH,
  UNARY_OP_ATANH,
  UNARY_OP_ABS,
  UNARY_OP_FLOOR,
  UNARY_OP_CEIL,
  UNARY_OP_RINT,
  UNARY_OP_ROUND,
  UNARY_OP_SIGN,
  UNARY_OP_REAL,
  UNARY_OP_IMAG,
  UNARY_OP_CONJ,
  UNARY_OP_TRUNC,
  UNARY_OP_INVALID_TYPE = 255
};

static const std::map<std::string, UnaryOptype> kUnaryOpTypeMap = {
  {"Exp", UNARY_OP_EXP},       {"Expm1", UNARY_OP_EXPM1},
  {"Log", UNARY_OP_LOG},       {"Log1p", UNARY_OP_LOG1P},
  {"Erf", UNARY_OP_ERF},       {"Erfc", UNARY_OP_ERFC},
  {"Neg", UNARY_OP_NEG},       {"Reciprocal", UNARY_OP_RECIPROCAL},
  {"Inv", UNARY_OP_INV},       {"Invert", UNARY_OP_INVERT},
  {"Square", UNARY_OP_SQUARE}, {"Sqrt", UNARY_OP_SQRT},
  {"Rsqrt", UNARY_OP_RSQRT},   {"Sin", UNARY_OP_SIN},
  {"Cos", UNARY_OP_COS},       {"Cosh", UNARY_OP_COSH},
  {"Asin", UNARY_OP_ASIN},     {"ACos", UNARY_OP_ACOS},
  {"Atan", UNARY_OP_ATAN},     {"Asinh", UNARY_OP_ASINH},
  {"Acosh", UNARY_OP_ACOSH},   {"Abs", UNARY_OP_ABS},
  {"Floor", UNARY_OP_FLOOR},   {"Ceil", UNARY_OP_CEIL},
  {"Rint", UNARY_OP_RINT},     {"Round", UNARY_OP_ROUND},
  {"Real", UNARY_OP_REAL},     {"Imag", UNARY_OP_IMAG},
  {"Sign", UNARY_OP_SIGN},     {"Conj", UNARY_OP_CONJ},
  {"Atanh", UNARY_OP_ATANH},   {"Tan", UNARY_OP_TAN},
  {"Sinh", UNARY_OP_SINH},     {"Trunc", UNARY_OP_TRUNC}};

template <typename T>
class UnaryHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit UnaryHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    unary_op_type_ = UNARY_OP_INVALID_TYPE;
  }
  virtual ~UnaryHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    auto iter = kUnaryOpTypeMap.find(kernel_name_);
    if (iter == kUnaryOpTypeMap.end()) {
      MS_LOG(ERROR) << "For 'UnaryOp', only support these types: "
                    << kernel::Map2Str<std::map, UnaryOptype>(kUnaryOpTypeMap) << " currently, but got "
                    << kernel_name_;
      return -1;
    }
    unary_op_type_ = iter->second;
    int flag = CalShapesSizeInBytes<T>(input_shapes, 1, kernel_name_, "input_shapes", &input_size_list_);
    output_size_list_ = input_size_list_;
    is_null_input_ = (flag == 1);
    if (flag != 0) {
      return flag;
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    static std::map<UnaryOptype, std::function<void(const T *, T *, const size_t, cudaStream_t)>> func_map = {
      {UNARY_OP_EXP, Exponential<T>}, {UNARY_OP_EXPM1, Expm1<T>},
      {UNARY_OP_LOG, Logarithm<T>},   {UNARY_OP_LOG1P, Log1p<T>},
      {UNARY_OP_ERF, Erf<T>},         {UNARY_OP_ERFC, Erfc<T>},
      {UNARY_OP_NEG, Negative<T>},    {UNARY_OP_RECIPROCAL, Reciprocal<T>},
      {UNARY_OP_INV, Inv<T>},         {UNARY_OP_INVERT, Invert<T>},
      {UNARY_OP_SQUARE, Square<T>},   {UNARY_OP_SQRT, Sqrt<T>},
      {UNARY_OP_RSQRT, Rsqrt<T>},     {UNARY_OP_SIN, Sin<T>},
      {UNARY_OP_COS, Cos<T>},         {UNARY_OP_COSH, Cosh<T>},
      {UNARY_OP_ASIN, Asin<T>},       {UNARY_OP_ACOS, ACos<T>},
      {UNARY_OP_ATAN, Atan<T>},       {UNARY_OP_ASINH, Asinh<T>},
      {UNARY_OP_ACOSH, Acosh<T>},     {UNARY_OP_ABS, Abs<T>},
      {UNARY_OP_FLOOR, Floor<T>},     {UNARY_OP_CEIL, Ceil<T>},
      {UNARY_OP_RINT, Rint<T>},       {UNARY_OP_ROUND, Round<T>},
      {UNARY_OP_SIGN, Sign<T>},       {UNARY_OP_ATANH, Atanh<T>},
      {UNARY_OP_TAN, Tan<T>},         {UNARY_OP_SINH, Sinh<T>},
      {UNARY_OP_TRUNC, Trunc<T>}};

    auto iter = func_map.find(unary_op_type_);
    if (iter != func_map.end()) {
      T *input_addr;
      T *output_addr;
      int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_addr);
      if (flag != 0) {
        return flag;
      }
      flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_addr);
      if (flag != 0) {
        return flag;
      }
      iter->second(input_addr, output_addr, input_size_list_[0] / sizeof(T),
                   reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      MS_LOG(ERROR) << "For 'UnaryOp', only support these types: "
                    << kernel::Map2Str<std::map, UnaryOptype>(kUnaryOpTypeMap) << " currently, but got "
                    << kernel_name_;
      return -1;
    }

    return 0;
  }

 private:
  UnaryOptype unary_op_type_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_HELPER_H_

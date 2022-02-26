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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <functional>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"

namespace mindspore {
namespace kernel {
enum UnaryOptype {
  UNARY_OP_EXP = 0,
  UNARY_OP_EXPM1,
  UNARY_OP_LOG,
  UNARY_OP_LOG1P,
  UNARY_OP_ERF,
  UNARY_OP_ERFC,
  UNARY_OP_NEG,
  UNARY_OP_RECIPROCAL,
  UNARY_OP_SQUARE,
  UNARY_OP_SQRT,
  UNARY_OP_RSQRT,
  UNARY_OP_SIN,
  UNARY_OP_COS,
  UNARY_OP_ASIN,
  UNARY_OP_ACOS,
  UNARY_OP_ATAN,
  UNARY_OP_ASINH,
  UNARY_OP_ACOSH,
  UNARY_OP_ABS,
  UNARY_OP_FLOOR,
  UNARY_OP_RINT,
  UNARY_OP_ROUND,
  UNARY_OP_SIGN,
  UNARY_OP_REAL,
  UNARY_OP_IMAG,
  UNARY_OP_CONJ,
  UNARY_OP_INVALID_TYPE = 255
};

static const std::map<std::string, UnaryOptype> kUnaryOpTypeMap = {
  {"Exp", UNARY_OP_EXP},       {"Expm1", UNARY_OP_EXPM1},
  {"Log", UNARY_OP_LOG},       {"Log1p", UNARY_OP_LOG1P},
  {"Erf", UNARY_OP_ERF},       {"Erfc", UNARY_OP_ERFC},
  {"Neg", UNARY_OP_NEG},       {"Reciprocal", UNARY_OP_RECIPROCAL},
  {"Square", UNARY_OP_SQUARE}, {"Sqrt", UNARY_OP_SQRT},
  {"Rsqrt", UNARY_OP_RSQRT},   {"Sin", UNARY_OP_SIN},
  {"Cos", UNARY_OP_COS},       {"Asin", UNARY_OP_ASIN},
  {"ACos", UNARY_OP_ACOS},     {"Atan", UNARY_OP_ATAN},
  {"Asinh", UNARY_OP_ASINH},   {"Acosh", UNARY_OP_ACOSH},
  {"Abs", UNARY_OP_ABS},       {"Floor", UNARY_OP_FLOOR},
  {"Rint", UNARY_OP_RINT},     {"Round", UNARY_OP_ROUND},
  {"Real", UNARY_OP_REAL},     {"Imag", UNARY_OP_IMAG},
  {"Sign", UNARY_OP_SIGN},     {"Conj", UNARY_OP_CONJ}};

template <typename T>
class UnaryOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  UnaryOpGpuKernelMod() { ResetResource(); }
  ~UnaryOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    static std::map<UnaryOptype, std::function<void(const T *, T *, const size_t, cudaStream_t)>> func_map = {
      {UNARY_OP_EXP, Exponential<T>}, {UNARY_OP_EXPM1, Expm1<T>},
      {UNARY_OP_LOG, Logarithm<T>},   {UNARY_OP_LOG1P, Log1p<T>},
      {UNARY_OP_ERF, Erf<T>},         {UNARY_OP_ERFC, Erfc<T>},
      {UNARY_OP_NEG, Negative<T>},    {UNARY_OP_RECIPROCAL, Reciprocal<T>},
      {UNARY_OP_SQUARE, Square<T>},   {UNARY_OP_SQRT, Sqrt<T>},
      {UNARY_OP_RSQRT, Rsqrt<T>},     {UNARY_OP_SIN, Sin<T>},
      {UNARY_OP_COS, Cos<T>},         {UNARY_OP_ASIN, Asin<T>},
      {UNARY_OP_ACOS, ACos<T>},       {UNARY_OP_ATAN, Atan<T>},
      {UNARY_OP_ASINH, Asinh<T>},     {UNARY_OP_ACOSH, Acosh<T>},
      {UNARY_OP_ABS, Abs<T>},         {UNARY_OP_FLOOR, Floor<T>},
      {UNARY_OP_RINT, Rint<T>},       {UNARY_OP_ROUND, Round<T>},
      {UNARY_OP_SIGN, Sign<T>}};

    auto iter = func_map.find(unary_op_type_);
    if (iter != func_map.end()) {
      T *input_addr = GetDeviceAddress<T>(inputs, 0);
      T *output_addr = GetDeviceAddress<T>(outputs, 0);
      iter->second(input_addr, output_addr, inputs[0]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: Exp, Expm1, Log, Log1p, Erf, Erfc,"
                        << " Neg, Reciprocal, Square, Sqrt, Rsqrt, Sin, Cos, Asin, ACos, Atan, Asinh, Acosh, Abs, "
                        << "Floor, Rint, Round, Real, Imag, Sign, Conj currently, but got " << unary_op_type_;
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kUnaryOpTypeMap.find(kernel_name);
    if (iter == kUnaryOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", only support these types: Exp, Expm1, Log, Log1p, Erf, Erfc,"
                        << " Neg, Reciprocal, Square, Sqrt, Rsqrt, Sin, Cos, Asin, ACos, Atan, Asinh, Acosh, Abs, "
                        << "Floor, Rint, Round, Real, Imag, Sign, Conj currently, but got " << kernel_name;
    }
    unary_op_type_ = iter->second;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
    }
    output_size_ = input_size_;
    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    unary_op_type_ = UNARY_OP_INVALID_TYPE;
    input_size_ = sizeof(T);
    output_size_ = sizeof(T);
    workspace_size_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  UnaryOptype unary_op_type_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_

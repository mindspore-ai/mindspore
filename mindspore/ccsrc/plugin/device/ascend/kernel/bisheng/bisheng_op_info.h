/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_OP_INFO_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_OP_INFO_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "include/api/format.h"
#include "include/api/data_type.h"
#include "kernel/oplib/opinfo.h"
#include "plugin/factory/ms_factory.h"

namespace cl::sycl::detail::half_impl {
class half;
}

namespace mindspore::kernel {
using half = cl::sycl::detail::half_impl::half;
class BiShengKernelMod;

#define REG(Clazz) const BishengOpInfoRegister<Clazz> Clazz::reg_ = BishengOpInfoRegister<Clazz>()

#define None_None \
  { "", "" }
#define None_Default \
  { "", "DefaultFormat" }
#define BOOL_None \
  { "bool", "" }
#define BOOL_Default \
  { "bool", "DefaultFormat" }
#define BOOL_5HD \
  { "bool", "NC1HWC0" }
#define BOOL_FracZ \
  { "bool", "FRACTAL_Z" }
#define BOOL_FracNZ \
  { "bool", "FRACTAL_NZ" }
#define BOOL_C1HWNCoC0 \
  { "bool", "C1HWNCoC0" }
#define BOOL_NCHW \
  { "bool", "NCHW" }
#define BOOL_NHWC \
  { "bool", "NHWC" }
#define BOOL_HWCN \
  { "bool", "HWCN" }
#define BOOL_NDHWC \
  { "bool", "NDHWC" }
#define BOOL_ChannelLast \
  { "bool", "ChannelLast" }

#define I8_None \
  { "int8", "" }
#define I8_Default \
  { "int8", "DefaultFormat" }
#define I8_5HD \
  { "int8", "NC1HWC0" }
#define I8_FracZ \
  { "int8", "FRACTAL_Z" }
#define I8_FracNZ \
  { "int8", "FRACTAL_NZ" }
#define I8_C1HWNCoC0 \
  { "int8", "C1HWNCoC0" }
#define I8_NCHW \
  { "int8", "NCHW" }
#define I8_NHWC \
  { "int8", "NHWC" }
#define I8_HWCN \
  { "int8", "HWCN" }
#define I8_NDHWC \
  { "int8", "NDHWC" }
#define I8_ChannelLast \
  { "int8", "ChannelLast" }
#define I8_NDC1HWC0 \
  { "int8", "NDC1HWC0" }

#define U8_None \
  { "uint8", "" }
#define U8_Default \
  { "uint8", "DefaultFormat" }
#define U8_5HD \
  { "uint8", "NC1HWC0" }
#define U8_FracZ \
  { "uint8", "FRACTAL_Z" }
#define U8_FracNZ \
  { "uint8", "FRACTAL_NZ" }
#define U8_C1HWNCoC0 \
  { "uint8", "C1HWNCoC0" }
#define U8_NCHW \
  { "uint8", "NCHW" }
#define U8_NHWC \
  { "uint8", "NHWC" }
#define U8_HWCN \
  { "uint8", "HWCN" }
#define U8_NDHWC \
  { "uint8", "NDHWC" }
#define U8_ChannelLast \
  { "uint8", "ChannelLast" }
#define U8_NDC1HWC0 \
  { "uint8", "NDC1HWC0" }

#define I16_None \
  { "int16", "" }
#define I16_Default \
  { "int16", "DefaultFormat" }
#define I16_5HD \
  { "int16", "NC1HWC0" }
#define I16_FracZ \
  { "int16", "FRACTAL_Z" }
#define I16_FracNZ \
  { "int16", "FRACTAL_NZ" }
#define I16_C1HWNCoC0 \
  { "int16", "C1HWNCoC0" }
#define I16_NCHW \
  { "int16", "NCHW" }
#define I16_NHWC \
  { "int16", "NHWC" }
#define I16_HWCN \
  { "int16", "HWCN" }
#define I16_NDHWC \
  { "int16", "NDHWC" }
#define I16_ChannelLast \
  { "int16", "ChannelLast" }

#define U16_None \
  { "uint16", "" }
#define U16_Default \
  { "uint16", "DefaultFormat" }
#define U16_5HD \
  { "uint16", "NC1HWC0" }
#define U16_FracZ \
  { "uint16", "FRACTAL_Z" }
#define U16_FracNZ \
  { "uint16", "FRACTAL_NZ" }
#define U16_C1HWNCoC0 \
  { "uint16", "C1HWNCoC0" }
#define U16_NCHW \
  { "uint16", "NCHW" }
#define U16_NHWC \
  { "uint16", "NHWC" }
#define U16_HWCN \
  { "uint16", "HWCN" }
#define U16_NDHWC \
  { "uint16", "NDHWC" }
#define U16_ChannelLast \
  { "uint16", "ChannelLast" }

#define I32_None \
  { "int32", "" }
#define I32_Default \
  { "int32", "DefaultFormat" }
#define I32_5HD \
  { "int32", "NC1HWC0" }
#define I32_FracZ \
  { "int32", "FRACTAL_Z" }
#define I32_FracNZ \
  { "int32", "FRACTAL_NZ" }
#define I32_C1HWNCoC0 \
  { "int32", "C1HWNCoC0" }
#define I32_NCHW \
  { "int32", "NCHW" }
#define I32_NHWC \
  { "int32", "NHWC" }
#define I32_HWCN \
  { "int32", "HWCN" }
#define I32_NDHWC \
  { "int32", "NDHWC" }
#define I32_NDC1HWC0 \
  { "int32", "NDC1HWC0" }
#define I32_NCDHW \
  { "int32", "NCDHW" }
#define I32_ChannelLast \
  { "int32", "ChannelLast" }

#define U32_None \
  { "uint32", "" }
#define U32_Default \
  { "uint32", "DefaultFormat" }
#define U32_5HD \
  { "uint32", "NC1HWC0" }
#define U32_FracZ \
  { "uint32", "FRACTAL_Z" }
#define U32_FracNZ \
  { "uint32", "FRACTAL_NZ" }
#define U32_C1HWNCoC0 \
  { "uint32", "C1HWNCoC0" }
#define U32_NCHW \
  { "uint32", "NCHW" }
#define U32_NHWC \
  { "uint32", "NHWC" }
#define U32_HWCN \
  { "uint32", "HWCN" }
#define U32_NDHWC \
  { "uint32", "NDHWC" }
#define U32_ChannelLast \
  { "uint32", "ChannelLast" }

#define I64_None \
  { "int64", "" }
#define I64_Default \
  { "int64", "DefaultFormat" }
#define I64_5HD \
  { "int64", "NC1HWC0" }
#define I64_FracZ \
  { "int64", "FRACTAL_Z" }
#define I64_FracNZ \
  { "int64", "FRACTAL_NZ" }
#define I64_C1HWNCoC0 \
  { "int64", "C1HWNCoC0" }
#define I64_NCHW \
  { "int64", "NCHW" }
#define I64_NHWC \
  { "int64", "NHWC" }
#define I64_HWCN \
  { "int64", "HWCN" }
#define I64_NDHWC \
  { "int64", "NDHWC" }
#define I64_ChannelLast \
  { "int64", "ChannelLast" }

#define U64_None \
  { "uint64", "" }
#define U64_Default \
  { "uint64", "DefaultFormat" }
#define U64_5HD \
  { "uint64", "NC1HWC0" }
#define U64_FracZ \
  { "uint64", "FRACTAL_Z" }
#define U64_FracNZ \
  { "uint64", "FRACTAL_NZ" }
#define U64_C1HWNCoC0 \
  { "uint64", "C1HWNCoC0" }
#define U64_NCHW \
  { "uint64", "NCHW" }
#define U64_NHWC \
  { "uint64", "NHWC" }
#define U64_HWCN \
  { "uint64", "HWCN" }
#define U64_NDHWC \
  { "uint64", "NDHWC" }
#define U64_ChannelLast \
  { "uint64", "ChannelLast" }

#define F16_None \
  { "float16", "" }
#define F16_Default \
  { "float16", "DefaultFormat" }
#define F16_5HD \
  { "float16", "NC1HWC0" }
#define F16_FracZ \
  { "float16", "FRACTAL_Z" }
#define F16_FracNZ \
  { "float16", "FRACTAL_NZ" }
#define F16_C1HWNCoC0 \
  { "float16", "C1HWNCoC0" }
#define F16_NCHW \
  { "float16", "NCHW" }
#define F16_NHWC \
  { "float16", "NHWC" }
#define F16_HWCN \
  { "float16", "HWCN" }
#define F16_NDHWC \
  { "float16", "NDHWC" }
#define F16_NCDHW \
  { "float16", "NCDHW" }
#define F16_DHWCN \
  { "float16", "DHWCN" }
#define F16_NDC1HWC0 \
  { "float16", "NDC1HWC0" }
#define F16_FRACTAL_Z_3D \
  { "float16", "FRACTAL_Z_3D" }
#define F16_FracZNLSTM \
  { "float16", "FRACTAL_ZN_LSTM" }
#define F16_FracZNRNN \
  { "float16", "FRACTAL_ZN_RNN" }
#define F16_ND_RNNBIAS \
  { "float16", "ND_RNN_BIAS" }
#define F16_ChannelLast \
  { "float16", "ChannelLast" }

#define F32_None \
  { "float32", "" }
#define F32_Default \
  { "float32", "DefaultFormat" }
#define F32_5HD \
  { "float32", "NC1HWC0" }
#define F32_FracZ \
  { "float32", "FRACTAL_Z" }
#define F32_FracNZ \
  { "float32", "FRACTAL_NZ" }
#define F32_C1HWNCoC0 \
  { "float32", "C1HWNCoC0" }
#define F32_NCHW \
  { "float32", "NCHW" }
#define F32_NHWC \
  { "float32", "NHWC" }
#define F32_HWCN \
  { "float32", "HWCN" }
#define F32_NDHWC \
  { "float32", "NDHWC" }
#define F32_NCDHW \
  { "float32", "NCDHW" }
#define F32_DHWCN \
  { "float32", "DHWCN" }
#define F32_NDC1HWC0 \
  { "float32", "NDC1HWC0" }
#define F32_FRACTAL_Z_3D \
  { "float32", "FRACTAL_Z_3D" }
#define F32_FracZNLSTM \
  { "float32", "FRACTAL_ZN_LSTM" }
#define F32_FracZNRNN \
  { "float32", "FRACTAL_ZN_RNN" }
#define F32_ND_RNNBIAS \
  { "float32", "ND_RNN_BIAS" }
#define F32_ChannelLast \
  { "float32", "ChannelLast" }

#define F64_None \
  { "float64", "" }
#define F64_Default \
  { "float64", "DefaultFormat" }
#define F64_5HD \
  { "float64", "NC1HWC0" }
#define F64_FracZ \
  { "float64", "FRACTAL_Z" }
#define F64_FracNZ \
  { "float64", "FRACTAL_NZ" }
#define F64_C1HWNCoC0 \
  { "float64", "C1HWNCoC0" }
#define F64_NCHW \
  { "float64", "NCHW" }
#define F64_NHWC \
  { "float64", "NHWC" }
#define F64_HWCN \
  { "float64", "HWCN" }
#define F64_NDHWC \
  { "float64", "NDHWC" }
#define F64_ChannelLast \
  { "float64", "ChannelLast" }

#define C64_Default \
  { "complex64", "DefaultFormat" }
#define C128_Default \
  { "complex128", "DefaultFormat" }

class BishengOpInfoRegisterHelper {
 public:
  BishengOpInfoRegisterHelper();

  void End();
  void OpName(const std::string &name);
  void Input(size_t index, const std::string &name, bool is_required);
  void Output(size_t index, const std::string &name, bool is_dynamic);
  void Attr(const std::string &name, const std::string &type, bool is_required);
  KernelAttr DataTypeFormat(const std::vector<std::pair<std::string, std::string>> &args);

 protected:
  std::shared_ptr<OpInfo> op_info_;
  std::map<size_t, std::shared_ptr<OpIOInfo>> inputs_;
  std::map<size_t, std::shared_ptr<OpIOInfo>> outputs_;
};

template <typename T>
class BishengOpInfoRegister : public BishengOpInfoRegisterHelper {
 public:
  BishengOpInfoRegister() : BishengOpInfoRegisterHelper(), func_list_(T::func_list_) {}
  const BishengOpInfoRegister<T> &End() {
    BishengOpInfoRegisterHelper::End();
    Factory<BiShengKernelMod>::Instance().Register(op_info_->op_name(),
                                                   std::move([]() { return std::make_shared<T>(); }));
    return *this;
  }
  BishengOpInfoRegister<T> &OpName(const std::string &name) {
    BishengOpInfoRegisterHelper::OpName(name);
    return *this;
  }
  BishengOpInfoRegister<T> &Input(size_t index, const std::string &name, bool is_required = true) {
    BishengOpInfoRegisterHelper::Input(index, name, is_required);
    return *this;
  }
  BishengOpInfoRegister<T> &Output(size_t index, const std::string &name, bool is_dynamic = false) {
    BishengOpInfoRegisterHelper::Output(index, name, is_dynamic);
    return *this;
  }
  BishengOpInfoRegister<T> &Attr(const std::string &name, const std::string &type, bool is_required = true) {
    BishengOpInfoRegisterHelper::Attr(name, type, is_required);
    return *this;
  }
  BishengOpInfoRegister<T> &DataTypeFormat(const std::vector<std::pair<std::string, std::string>> &args,
                                           typename T::Func &&func) {
    auto attr = BishengOpInfoRegisterHelper::DataTypeFormat(args);
    func_list_.emplace_back(attr, std::move(func));
    return *this;
  }

 private:
  typename T::FuncList &func_list_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_ASCEND_BISHENG_OP_INFO_H

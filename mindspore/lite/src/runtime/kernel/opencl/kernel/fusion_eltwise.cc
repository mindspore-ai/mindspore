/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/opencl/kernel/fusion_eltwise.h"
#include <algorithm>
#include "src/runtime/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/scale.h"
#include "src/common/prim_inner.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

static std::set<EltwiseOperator> SupportedOperators = {
  // Arithmetic Primitive
  Operator_Mul,
  Operator_Add,
  Operator_Sub,
  Operator_Div,
  // ArithmeticSelf Primitive
  Operator_Neg,
  // Other Primitive
  Operator_Scale,
  // Activation
  Operator_Act_NO_ACTIVATION,
  Operator_Act_RELU,
  Operator_Act_SIGMOID,
  Operator_Act_RELU6,
  Operator_Act_RELU1,
  Operator_Act_TANH,
};

std::pair<bool, FusionEltwiseParameter *> CheckSupportOrCreateParam(
  LiteKernel *node, bool create_param = false,
  const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {}) {
  MS_ASSERT(node);
  PrimitiveType node_type = node->Type();
  auto operator_ = static_cast<const EltwiseOperator>(node_type);
  auto *op_parameter = reinterpret_cast<OpenCLKernel *>(node)->GetParameter();
  bool support = false;
  FusionEltwiseParameter *param = nullptr;

  if (node_type == PrimitiveType_FusionEltwise) {
    support = true;
    if (create_param) {
      auto *eltwise = reinterpret_cast<FusionEltwiseOpenCLKernel *>(node);
      param = reinterpret_cast<FusionEltwiseParameter *>(eltwise->GetParameter());
      eltwise->ClearParameter();
    }
  } else if (IsArithmetic(node_type) || node_type == schema::PrimitiveType_ScaleFusion) {
    auto *arith_param = reinterpret_cast<ArithmeticParameter *>(op_parameter);
    auto *scale_param = reinterpret_cast<ScaleParameter *>(op_parameter);
    auto act_type = static_cast<ActivationType>(
      node_type == schema::PrimitiveType_ScaleFusion ? scale_param->activation_type_ : arith_param->activation_type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    support = SupportedOperators.count(operator_) && SupportedOperators.count(act_operator);
    if (node_type == schema::PrimitiveType_ScaleFusion) {
      support = support && node->in_tensors().size() == 3 && scale_param->axis_ == -1;
    } else {
      support = support && (node->in_tensors().size() == 2);
    }
    if (create_param) {
      param = new (std::nothrow) FusionEltwiseParameter(operator_, node->name(), node->in_tensors(), replace_map);
      MS_ASSERT(param);
      if (act_operator != Operator_Act_NO_ACTIVATION) {
        std::string act_name = schema::EnumNameActivationType(act_type);
        auto *fake_tensor = reinterpret_cast<lite::Tensor *>(param);
        param =
          new (std::nothrow) FusionEltwiseParameter(act_operator, act_name, {fake_tensor}, {{fake_tensor, param}});
        MS_ASSERT(param);
      }
    }
  } else if (IsArithmeticSelf(node_type)) {
    support = node->in_tensors().size() == 1 && SupportedOperators.count(operator_);
    if (create_param) {
      param = new (std::nothrow) FusionEltwiseParameter(operator_, node->name(), node->in_tensors(), replace_map);
      MS_ASSERT(param);
    }
  } else if (node_type == schema::PrimitiveType_Activation) {
    auto act_type = static_cast<ActivationType>(reinterpret_cast<ActivationParameter *>(op_parameter)->type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    support = node->in_tensors().size() == 1 && SupportedOperators.count(act_operator);
    if (create_param) {
      param = new (std::nothrow) FusionEltwiseParameter(act_operator, node->name(), node->in_tensors(), replace_map);
      MS_ASSERT(param);
    }
  }
  return {support, param};
}

bool IsOperatorSupported(LiteKernel *node) { return CheckSupportOrCreateParam(node).first; }

FusionEltwiseParameter *CreateFusionEltwiseParameter(
  LiteKernel *node, const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map) {
  return CheckSupportOrCreateParam(node, true, replace_map).second;
}

bool IsEltwiseAndOperatorSupported(LiteKernel *node) {
  MS_ASSERT(node);
  if (!IsOperatorSupported(node)) {
    return false;
  }
  if (node->out_tensors().size() != 1) {
    return false;
  }
  auto *output_tensor = node->out_tensors().front();
  MS_ASSERT(output_tensor);
  auto output_info = GpuTensorInfo(output_tensor);
  auto output_shape = output_tensor->shape();
  for (auto *in_tensor : node->in_tensors()) {
    MS_ASSERT(in_tensor);
    auto shape = in_tensor->shape();
    bool is_scalar = shape.empty() || (shape.size() == 1 && shape.front() == 1);
    bool is_vector = shape.size() == 1 && shape.front() == output_info.C;
    bool _111C = shape.size() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == 1 && shape[3] == output_info.C;
    bool same_with_out = shape == output_shape;
    if (!(is_scalar || is_vector || _111C || same_with_out)) {
      return false;
    }
    if (in_tensor->data_type() != kNumberTypeFloat16 && in_tensor->data_type() != kNumberTypeFloat32) {
      return false;
    }
  }
  if (output_tensor->data_type() != kNumberTypeFloat16 && output_tensor->data_type() != kNumberTypeFloat32) {
    return false;
  }
  return true;
}

int FusionEltwiseOpenCLKernel::Prepare() {
  std::string source = Codegen();
  std::string program_name = "FusionEltwise\n" + source;
  std::string kernel_name = "FusionEltwise";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  InitWeights();
  SetGlobalLocal();
  SetConstArgs();
  return RET_OK;
}

template <typename DstT, typename SrcT>
void CopyNumber(void *dst, void *src, size_t n) {
  MS_ASSERT(dst);
  MS_ASSERT(src);
  if (sizeof(DstT) == sizeof(SrcT)) {
    memcpy(dst, src, n * sizeof(DstT));
  } else {
    auto *dst_ = static_cast<DstT *>(dst);
    auto *src_ = static_cast<SrcT *>(src);
    for (int i = 0; i < n; ++i) {
      dst_[i] = static_cast<DstT>(src_[i]);
    }
  }
}

int FusionEltwiseOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  bool use_fp16 = ocl_runtime_->GetFp16Enable();
  for (auto *tensor : in_tensors_) {
    MS_ASSERT(tensor);
    if (tensor->IsConst()) {
      if (IsScalar(tensor->shape())) {
        float value = (tensor->data_type() == kNumberTypeFloat16) ? *(reinterpret_cast<float16_t *>(tensor->data_c()))
                                                                  : *(reinterpret_cast<float32_t *>(tensor->data_c()));
        scalar_weights_.push_back(value);
      } else {
        auto tensor_info = GpuTensorInfo(tensor);
        size_t num = tensor_info.ElementsNum;
        size_t size = tensor_info.Image2DSize;
        void *buffer = allocator->Malloc(size);
        allocator->MapBuffer(buffer, CL_MAP_WRITE, nullptr, true);
        memset(buffer, 0x00, size);
        if (tensor->data_type() == kNumberTypeFloat16) {
          if (use_fp16) {
            CopyNumber<float16_t, float16_t>(buffer, tensor->data_c(), num);
          } else {
            CopyNumber<float32_t, float16_t>(buffer, tensor->data_c(), num);
          }
        } else {
          if (use_fp16) {
            CopyNumber<float16_t, float32_t>(buffer, tensor->data_c(), num);
          } else {
            CopyNumber<float32_t, float32_t>(buffer, tensor->data_c(), num);
          }
        }
        allocator->UnmapBuffer(buffer);
        buffer_weights_.push_back(buffer);
      }
    }
  }
  return RET_OK;
}

void FusionEltwiseOpenCLKernel::SetGlobalLocal() {
  auto output = GpuTensorInfo(out_tensors_.front());
  global_size_ = {output.N * output.H, output.W, output.Slice};
  local_size_ = {};
  AlignGlobalLocal(global_size_, local_size_);
}

void FusionEltwiseOpenCLKernel::SetConstArgs() {
  auto output = GpuTensorInfo(out_tensors_.front());
  cl_int4 output_shape = {static_cast<cl_int>(output.N), static_cast<cl_int>(output.H), static_cast<cl_int>(output.W),
                          static_cast<cl_int>(output.C)};
  int arg_idx = 0;
  int scalar_idx = 0;
  int buffer_idx = 0;
  for (auto *in_tensor : in_tensors_) {
    MS_ASSERT(in_tensor);
    if (in_tensor->IsConst()) {
      if (IsScalar(in_tensor->shape())) {
        if (ocl_runtime_->GetFp16Enable()) {
          auto value = static_cast<float16_t>(scalar_weights_[scalar_idx++]);
          ocl_runtime_->SetKernelArg(kernel_, arg_idx, *(reinterpret_cast<cl_half *>(&value)));
        } else {
          ocl_runtime_->SetKernelArg(kernel_, arg_idx, scalar_weights_[scalar_idx++]);
        }
      } else {
        ocl_runtime_->SetKernelArg(kernel_, arg_idx, buffer_weights_[buffer_idx++], lite::opencl::MemType::BUF);
      }
    }
    arg_idx++;  // for act input
  }
  arg_idx++;  // for output
  ocl_runtime_->SetKernelArg(kernel_, arg_idx, output_shape);
}

int FusionEltwiseOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  for (auto *in_tensor : in_tensors_) {
    if (!in_tensor->IsConst()) {
      ocl_runtime_->SetKernelArg(kernel_, arg_idx, in_tensor->data_c());
    }
    arg_idx++;
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx, out_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

std::string FusionEltwiseOpenCLKernel::Codegen() {
  std::stringstream code;
  code << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
          "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
          "__kernel void FusionEltwise(";

  for (int i = 0; i < in_tensors_.size(); ++i) {
    MS_ASSERT(in_tensors_[i]);
    if (in_tensors_[i]->IsConst()) {
      if (IsScalar(in_tensors_[i]->shape())) {
        code << "FLT in" << i << ", ";
      } else {
        code << "__global FLT4 *input" << i << ", ";
      }
    } else {
      code << "__read_only image2d_t input" << i << ", ";
    }
  }

  code << "__write_only image2d_t output, int4 output_shape) {\n"
          "  int N = output_shape.x, H = output_shape.y, W = output_shape.z, C = output_shape.w;\n"
          "  int SLICES = (C + 3) / 4;\n"
          "  int nh = get_global_id(0);\n"
          "  int w = get_global_id(1);\n"
          "  int slice = get_global_id(2);\n"
          "  int n = nh / H;\n"
          "  int h = nh % H;\n"
          "  if (n >= N || h >= H || w >= W || slice >= SLICES) {\n"
          "    return;\n"
          "  }\n";

  auto output = GpuTensorInfo(out_tensors_.front());
  for (int i = 0; i < in_tensors_.size(); ++i) {
    auto *tensor = in_tensors_[i];
    MS_ASSERT(tensor);
    auto shape = in_tensors_[i]->shape();
    bool is_scalar = IsScalar(shape);
    bool is_vector = shape.size() == 1 && shape.front() == output.C;
    bool _111C = shape.size() == 4 && shape[0] == 1 && shape[1] == 1 && shape[2] == 1 && shape[3] == output.C;
    if (tensor->IsConst()) {
      if (!is_scalar) {
        code << "  FLT4 in" << i << " = input" << i << "[";
        if (is_vector || _111C) {
          code << "slice";
        } else {
          code << "(nh * W + w) * SLICES + slice";
        }
        code << "];\n";
      }
    } else {
      code << "  FLT4 in" << i << " = READ_IMAGE(input" << i << ", smp_zero, (int2)(";
      if (is_scalar) {
        code << "0, 0";
      } else if (is_vector || _111C) {
        code << "slice, 0";
      } else {
        code << "w * SLICES + slice, nh";
      }
      code << "));\n";
    }
  }
  code << "\n";
  MS_LOG(DEBUG) << "\n" << reinterpret_cast<FusionEltwiseParameter *>(op_parameter_)->name_ << ":";
  code << CodegenCore(reinterpret_cast<FusionEltwiseParameter *>(op_parameter_));
  code << "\n  WRITE_IMAGE(output, (int2)(w * SLICES + slice, nh), out);\n"
          "}\n\n";
  return code.str();
}

std::string FusionEltwiseOpenCLKernel::CodegenCore(FusionEltwiseParameter *param, const std::string &out_name,
                                                   int degree) {
  std::stringstream code;
  std::string log_prefix(degree * 2, ' ');
  std::string cl_prefix((degree + 1) * 2, ' ');

  std::vector<std::string> input_names;
  MS_ASSERT(param);
  for (const auto &input : param->inputs_) {
    if (input.is_leaf_) {
      input_names.push_back("in" + std::to_string(GetTensorIdx(reinterpret_cast<lite::Tensor *>(input.value_))));
      MS_LOG(DEBUG) << log_prefix << degree << " Tensor=" << input.value_;
    } else {
      std::string var = GetFormatVarName(input.name_);
      input_names.push_back(var);
      MS_LOG(DEBUG) << log_prefix << degree << " Parameter(degree=" << degree << ")";
      code << CodegenCore(input.value_, var, degree + 1);
    }
  }
  const std::string &var0 = input_names.at(0);

  static std::map<EltwiseOperator, char> simple_symbols = {
    {Operator_Add, '+'},
    {Operator_Sub, '-'},
    {Operator_Mul, '*'},
    {Operator_Div, '/'},
  };
  if (simple_symbols.count(param->operator_)) {
    const std::string &var1 = input_names.at(1);
    code << cl_prefix << "FLT4 " << out_name << " = " << var0 << " " << simple_symbols[param->operator_] << " " << var1
         << ";\n";
  } else if (param->operator_ == Operator_Neg) {
    code << cl_prefix << "FLT4 " << out_name << " = -" << var0 << ";\n";
  } else if (param->operator_ == Operator_Scale) {
    const std::string &var1 = input_names.at(1);
    const std::string &var2 = input_names.at(2);
    code << cl_prefix << "FLT4 " << out_name << " = " << var0 << " * " << var1 << " + " << var2 << ";\n";
  } else {
    if (param->operator_ == Operator_Act_NO_ACTIVATION) {
      code << cl_prefix << "FLT4 " << out_name << " = " << var0 << ";\n";
    } else if (param->operator_ == Operator_Act_RELU) {
      code << cl_prefix << "FLT4 " << out_name << " =  max(" << var0 << ", (FLT4)(0.0f));\n";
    } else if (param->operator_ == Operator_Act_SIGMOID) {
      code << cl_prefix << "FLT4 " << out_name << " =  (FLT4)(1.f) / ((FLT4)(1.f) + exp(-" << var0 << "));\n";
    } else if (param->operator_ == Operator_Act_RELU6) {
      code << cl_prefix << "FLT4 " << out_name << " =  clamp(" << var0 << ", (FLT4)(0.0f), (FLT4)(6.0f));\n";
    } else if (param->operator_ == Operator_Act_LEAKY_RELU) {
    } else if (param->operator_ == Operator_Act_RELU1) {
      code << cl_prefix << "FLT4 " << out_name << " =  clamp(" << var0 << ", (FLT4)(0.0f), (FLT4)(1.0f));\n";
    } else if (param->operator_ == Operator_Act_TANH) {
      std::string exp0 = GetFormatVarName();
      std::string exp1 = GetFormatVarName();
      code << cl_prefix << "FLT4 " << exp0 << " =  exp(" + var0 + ");\n";
      code << cl_prefix << "FLT4 " << exp1 << " =  exp(-" + var0 + ");\n";
      code << cl_prefix << "FLT4 " << out_name << " =  (" << exp0 << " - " << exp1 << ") / (" << exp0 << " + " << exp1
           << ");\n";
    }
  }

  return code.str();
}

std::string FusionEltwiseOpenCLKernel::GetFormatVarName(std::string name) {
  if (var_names_.count(name)) {
    return simplify_var_name_ ? var_names_[name] : name;
  } else {
    if (name.empty()) {
      name = "_var_" + std::to_string(var_names_.size());
    } else {
      char c = name.front();
      if (c != '_' && !std::isalpha(c)) {
        name = '_' + name;
      }
      std::replace_if(
        name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    }
    auto new_name = "tmp" + std::to_string(var_names_.size());
    var_names_.emplace(name, new_name);
    return simplify_var_name_ ? new_name : name;
  }
}

int FusionEltwiseOpenCLKernel::GetTensorIdx(lite::Tensor *in_tensor) {
  MS_ASSERT(in_tensor);
  auto pos = std::find(in_tensors_.begin(), in_tensors_.end(), in_tensor);
  if (pos != in_tensors_.end()) {
    return pos - in_tensors_.begin();
  } else {
    for (const auto &in_kernel : in_kernels_) {
      MS_ASSERT(in_kernel);
      MS_ASSERT(in_kernel->in_tensors().size());
      MS_ASSERT(in_kernel->out_tensors().size());
      if (static_cast<int>(in_kernel->Type()) == lite::PRIM_TO_FORMAT) {
        if (in_tensor == in_kernel->in_tensors().front()) {
          return std::find(in_tensors_.begin(), in_tensors_.end(), in_kernel->out_tensors().front()) -
                 in_tensors_.begin();
        }
      }
    }
    return 0;
  }
}

}  // namespace mindspore::kernel

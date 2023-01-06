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
#include "src/litert/kernel/opencl/kernel/fusion_eltwise.h"
#include <algorithm>
#include "src/litert/kernel/opencl/utils.h"
#include "include/errorcode.h"
#include "nnacl/arithmetic.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/scale.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
constexpr int LOG_PREFIX_SCALE = 2;
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

bool CheckSupport(KernelExec *node) {
  MS_ASSERT(node);
  PrimitiveType node_type = node->type();
  auto operator_ = static_cast<const EltwiseOperator>(node_type);
  auto *op_parameter = reinterpret_cast<OpenCLKernel *>(node->kernel())->GetParameter();

  if (node_type == PrimitiveType_FusionEltwise) {
    return true;
  }
  if (IsArithmetic(node_type) || node_type == schema::PrimitiveType_ScaleFusion) {
    auto *arith_param = reinterpret_cast<ArithmeticParameter *>(op_parameter);
    auto *scale_param = reinterpret_cast<ScaleParameter *>(op_parameter);
    auto act_type = static_cast<ActivationType>(
      node_type == schema::PrimitiveType_ScaleFusion ? scale_param->activation_type_ : arith_param->activation_type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    auto support = SupportedOperators.count(operator_) && SupportedOperators.count(act_operator);
    if (node_type == schema::PrimitiveType_ScaleFusion) {
      return support && node->in_tensors().size() == INPUT_TENSOR_SIZE_3 && scale_param->axis_ == -1;
    } else {
      return support && (node->in_tensors().size() == INPUT_TENSOR_SIZE_2);
    }
  }
  if (IsArithmeticSelf(node_type)) {
    return node->in_tensors().size() == INPUT_TENSOR_SIZE_1 && SupportedOperators.count(operator_);
  }
  if (node_type == schema::PrimitiveType_Activation) {
    auto act_type = static_cast<ActivationType>(reinterpret_cast<ActivationParameter *>(op_parameter)->type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    return node->in_tensors().size() == 1 && SupportedOperators.count(act_operator);
  }
  return false;
}

FusionEltwiseParameter *CreateParam(KernelExec *node,
                                    const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map = {}) {
  MS_ASSERT(node);
  PrimitiveType node_type = node->type();
  auto operator_ = static_cast<const EltwiseOperator>(node_type);
  auto *op_parameter = reinterpret_cast<OpenCLKernel *>(node->kernel())->GetParameter();
  FusionEltwiseParameter *param = nullptr;

  if (node_type == PrimitiveType_FusionEltwise) {
    auto *eltwise = reinterpret_cast<FusionEltwiseOpenCLKernel *>(node->kernel());
    param = reinterpret_cast<FusionEltwiseParameter *>(eltwise->GetParameter());
    eltwise->ClearParameter();
  }
  if (IsArithmetic(node_type) || node_type == schema::PrimitiveType_ScaleFusion) {
    auto *arith_param = reinterpret_cast<ArithmeticParameter *>(op_parameter);
    auto *scale_param = reinterpret_cast<ScaleParameter *>(op_parameter);
    auto act_type = static_cast<ActivationType>(
      node_type == schema::PrimitiveType_ScaleFusion ? scale_param->activation_type_ : arith_param->activation_type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    param = new (std::nothrow) FusionEltwiseParameter(operator_, node->name(), node->in_tensors(), replace_map);
    if (param == nullptr) {
      MS_LOG(ERROR) << "FusionEltwiseParameter is nullptr.";
      return nullptr;
    }
    if (act_operator != Operator_Act_NO_ACTIVATION) {
      std::string act_name = schema::EnumNameActivationType(act_type);
      auto *fake_tensor = reinterpret_cast<lite::Tensor *>(param);
      param = new (std::nothrow) FusionEltwiseParameter(act_operator, act_name, {fake_tensor}, {{fake_tensor, param}});
    }
  }
  if (IsArithmeticSelf(node_type)) {
    param = new (std::nothrow) FusionEltwiseParameter(operator_, node->name(), node->in_tensors(), replace_map);
  }
  if (node_type == schema::PrimitiveType_Activation) {
    auto act_type = static_cast<ActivationType>(reinterpret_cast<ActivationParameter *>(op_parameter)->type_);
    EltwiseOperator act_operator = Activation2Operator(act_type);
    param = new (std::nothrow) FusionEltwiseParameter(act_operator, node->name(), node->in_tensors(), replace_map);
  }
  if (param == nullptr) {
    MS_LOG(ERROR) << "Parameter is nullptr.";
    return nullptr;
  }

  param->op_parameter_.is_zero_shape_ = false;
  return param;
}

FusionEltwiseParameter *CreateFusionEltwiseParameter(
  KernelExec *node, const std::map<lite::Tensor *, FusionEltwiseParameter *> &replace_map) {
  return CreateParam(node, replace_map);
}

bool CheckDateTypeSupport(lite::Tensor *tensor) {
  if (tensor->data_type() != kNumberTypeFloat16 && tensor->data_type() != kNumberTypeFloat32) {
    return false;
  }
  return true;
}

bool IsEltwiseAndOperatorSupported(KernelExec *node) {
  MS_ASSERT(node);
  if (!CheckSupport(node)) {
    return false;
  }
  if (node->out_tensors().size() != 1) {
    return false;
  }
  auto *output_tensor = node->out_tensors()[0];
  MS_ASSERT(output_tensor);
  auto output_info = GpuTensorInfo::CreateGpuTensorInfo(output_tensor);
  if (output_info == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  auto output_shape = output_tensor->shape();
  for (auto *in_tensor : node->in_tensors()) {
    MS_ASSERT(in_tensor);
    auto shape = in_tensor->shape();
    bool is_scalar = shape.empty() || (shape.size() == DIMENSION_1D && shape.front() == 1);
    bool is_vector = shape.size() == DIMENSION_1D && shape.front() == static_cast<int>(output_info->C);
    bool _111C = shape.size() == DIMENSION_4D && shape[kNHWC_N] == 1 && shape[kNHWC_H] == 1 && shape[kNHWC_W] == 1 &&
                 shape[kNHWC_C] == static_cast<int>(output_info->C);
    bool same_with_out = shape == output_shape;
    if (!(is_scalar || is_vector || _111C || same_with_out)) {
      return false;
    }
    if (!CheckDateTypeSupport(in_tensor)) {
      return false;
    }
  }
  return CheckDateTypeSupport(output_tensor);
}

int FusionEltwiseOpenCLKernel::Prepare() {
  std::string source = Codegen();
  if (source.empty()) {
    MS_LOG(ERROR) << "Codegen source failed.";
    return RET_ERROR;
  }
  const std::string program_name = "FusionEltwise\n" + source;
  const std::string kernel_name = "FusionEltwise";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  std::vector<std::string> build_options_ext;
  if (ocl_runtime_->GetFp16Enable()) {
    build_options_ext = {" -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh "};
  } else {
    build_options_ext = {" -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef "};
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  ret = InitWeights();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitWeights failed.";
    return ret;
  }
  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
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
    for (size_t i = 0; i < n; ++i) {
      dst_[i] = static_cast<DstT>(src_[i]);
    }
  }
}

#ifdef ENABLE_FP16
int FusionEltwiseOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  bool use_fp16 = ocl_runtime_->GetFp16Enable();
  for (auto *tensor : in_tensors_) {
    MS_ASSERT(tensor);
    if (tensor->IsConst()) {
      if (IsScalar(tensor->shape())) {
        float value = (tensor->data_type() == kNumberTypeFloat16) ? *(reinterpret_cast<float16_t *>(tensor->data()))
                                                                  : *(reinterpret_cast<float32_t *>(tensor->data()));
        scalar_weights_.push_back(value);
      } else {
        auto tensor_info = GpuTensorInfo::CreateGpuTensorInfo(tensor);
        if (tensor_info == nullptr) {
          MS_LOG(ERROR) << "Create gpu tensor info failed.";
          return RET_ERROR;
        }
        size_t num = tensor_info->ElementsNum;
        size_t size = tensor_info->Image2DSize;
        void *buffer = allocator->Malloc(size, lite::opencl::MemType::BUF);
        if (buffer == nullptr) {
          MS_LOG(ERROR) << "Malloc failed.";
          return RET_ERROR;
        }
        if (allocator->MapBuffer(buffer, CL_MAP_WRITE, nullptr, true) == nullptr) {
          MS_LOG(ERROR) << "Map Buffer failed.";
          return RET_ERROR;
        }
        memset(buffer, 0x00, size);
        if (tensor->data_type() == kNumberTypeFloat16) {
          if (use_fp16) {
            CopyNumber<float16_t, float16_t>(buffer, tensor->data(), num);
          } else {
            CopyNumber<float32_t, float16_t>(buffer, tensor->data(), num);
          }
        } else {
          if (use_fp16) {
            CopyNumber<float16_t, float32_t>(buffer, tensor->data(), num);
          } else {
            CopyNumber<float32_t, float32_t>(buffer, tensor->data(), num);
          }
        }
        if (allocator->UnmapBuffer(buffer) != RET_OK) {
          MS_LOG(ERROR) << "UnmapBuffer failed.";
          return RET_ERROR;
        }
        buffer_weights_.push_back(buffer);
      }
    }
  }
  return RET_OK;
}
#else
int FusionEltwiseOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  for (auto *tensor : in_tensors_) {
    MS_ASSERT(tensor);
    if (tensor->IsConst()) {
      if (IsScalar(tensor->shape())) {
        float value = *reinterpret_cast<float *>(tensor->data());
        scalar_weights_.push_back(value);
      } else {
        auto tensor_info = GpuTensorInfo::CreateGpuTensorInfo(tensor);
        if (tensor_info == nullptr) {
          MS_LOG(ERROR) << "Create gpu tensor info failed.";
          return RET_ERROR;
        }
        size_t num = tensor_info->ElementsNum;
        size_t size = tensor_info->Image2DSize;
        void *buffer_weight = allocator->Malloc(size, lite::opencl::MemType::BUF);
        if (buffer_weight == nullptr) {
          MS_LOG(ERROR) << "Malloc failed.";
          return RET_ERROR;
        }
        if (allocator->MapBuffer(buffer_weight, CL_MAP_WRITE, nullptr, true) == nullptr) {
          MS_LOG(ERROR) << "Map Buffer failed.";
          return RET_ERROR;
        }
        memset(buffer_weight, 0x00, size);
        CopyNumber<float, float>(buffer_weight, tensor->data(), num);
        if (allocator->UnmapBuffer(buffer_weight) != RET_OK) {
          MS_LOG(ERROR) << "UnmapBuffer failed.";
          return RET_ERROR;
        }
        buffer_weights_.push_back(buffer_weight);
      }
    }
  }
  return RET_OK;
}
#endif

int FusionEltwiseOpenCLKernel::SetGlobalLocal() {
  auto output = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_.front());
  if (output == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  global_size_ = {output->N * output->D * output->H, output->W, output->Slice};
  local_size_ = {};
  AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int FusionEltwiseOpenCLKernel::SetConstArgs() {
  auto output = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_.front());
  if (output == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  cl_int4 output_shape = {static_cast<cl_int>(output->N), static_cast<cl_int>(output->D * output->H),
                          static_cast<cl_int>(output->W), static_cast<cl_int>(output->C)};
  int arg_idx = 0;
  int scalar_idx = 0;
  int buffer_idx = 0;
  for (auto *in_tensor : in_tensors_) {
    MS_ASSERT(in_tensor);
    if (in_tensor->IsConst()) {
      if (IsScalar(in_tensor->shape())) {
#ifdef ENABLE_FP16
        if (ocl_runtime_->GetFp16Enable()) {
          auto value = static_cast<float16_t>(scalar_weights_[scalar_idx++]);
          if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, *(reinterpret_cast<cl_half *>(&value))) != CL_SUCCESS) {
            MS_LOG(ERROR) << "SetKernelArg failed.";
            return RET_ERROR;
          }
        } else {
          if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, scalar_weights_[scalar_idx++]) != CL_SUCCESS) {
            MS_LOG(ERROR) << "SetKernelArg failed.";
            return RET_ERROR;
          }
        }
#else
        if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, scalar_weights_[scalar_idx++]) != CL_SUCCESS) {
          MS_LOG(ERROR) << "SetKernelArg failed.";
          return RET_ERROR;
        }
#endif
      } else {
        if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, buffer_weights_[buffer_idx++], true) != CL_SUCCESS) {
          MS_LOG(ERROR) << "SetKernelArg failed.";
          return RET_ERROR;
        }
      }
    }
    arg_idx++;  // for act input
  }
  arg_idx++;  // for output
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, output_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int FusionEltwiseOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_idx = 0;
  for (auto *in_tensor : in_tensors_) {
    if (!in_tensor->IsConst()) {
      if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, in_tensor->data()) != CL_SUCCESS) {
        MS_LOG(ERROR) << "SetKernelArg failed.";
        return RET_ERROR;
      }
    }
    arg_idx++;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::string FusionEltwiseOpenCLKernel::Codegen() {
  std::stringstream code;
  code << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
          "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
          "__kernel void FusionEltwise(";

  for (size_t i = 0; i < in_tensors_.size(); ++i) {
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

  auto output = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_.front());
  if (output == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return "";
  }
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    auto *tensor = in_tensors_[i];
    MS_ASSERT(tensor);
    auto shape = in_tensors_[i]->shape();
    bool is_scalar = IsScalar(shape);
    bool is_vector = shape.size() == DIMENSION_1D && shape.front() == static_cast<int>(output->C);
    bool _111C = shape.size() == DIMENSION_4D && shape[kNHWC_N] == 1 && shape[kNHWC_H] == 1 && shape[kNHWC_W] == 1 &&
                 shape[kNHWC_C] == static_cast<int>(output->C);
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
  std::string log_prefix(degree * LOG_PREFIX_SCALE, ' ');
  std::string cl_prefix((degree + 1) * LOG_PREFIX_SCALE, ' ');

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
    const std::string &var2 = input_names.at(2);  // 2 : second input
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
      code << cl_prefix << var0 + " = clamp(" + var0 + ", (FLT)(-10.0f), (FLT)(10.0f));\n";
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
    auto new_name = "tmp_" + name + "_" + std::to_string(var_names_.size());
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
    auto in_kernels = *this->in_kernels_;
    for (const auto &in_kernel : in_kernels) {
      MS_ASSERT(in_kernel);
      MS_ASSERT(in_kernel->in_tensors().size());
      MS_ASSERT(in_kernel->out_tensors().size());
      int type = in_kernel->op_parameter() == nullptr ? in_kernel->type() : in_kernel->op_parameter()->type_;
      if (type == PrimType::PrimType_Inner_ToFormat) {
        if (in_tensor == in_kernel->in_tensors().front()) {
          return std::find(in_tensors_.begin(), in_tensors_.end(), in_kernel->out_tensors().front()) -
                 in_tensors_.begin();
        }
      }
    }
    MS_LOG(ERROR) << "FusionEltwise can't find index ";
  }
  return 0;
}
}  // namespace mindspore::kernel

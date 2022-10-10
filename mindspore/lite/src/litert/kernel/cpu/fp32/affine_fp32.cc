/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/affine_fp32.h"
#include <algorithm>
#include <set>
#include "src/litert/kernel/cpu/fp32/matmul_fp32.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/activation_fp32.h"
#include "nnacl/fp32/splice_fp32.h"
#include "src/common/utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Affine;
namespace mindspore::kernel {
int AffineFp32CPUKernel::DoActivation(lite::Tensor *tensor) {
  auto data = static_cast<float *>(tensor->MutableData());
  int length = tensor->ElementsNum();
  switch (schema::ActivationType(affine_parameter_->activation_type_)) {
    case schema::ActivationType_RELU:
      return Fp32Relu(data, length, data);
    case schema::ActivationType_RELU6:
      return Fp32Relu6(data, length, data);
    case schema::ActivationType_SIGMOID:
      return Sigmoid(data, length, data);
    case schema::ActivationType_TANH:
      return Tanh(data, length, data);
    case schema::ActivationType_SWISH:
      return Swish(data, length, data);
    case schema::ActivationType_HSWISH:
      return HSwish(data, length, data);
    case schema::ActivationType_HSIGMOID:
      return HSigmoid(data, length, data);
    case schema::ActivationType_SOFTPLUS:
      return Softplus(data, length, data);
    default:
      MS_LOG(ERROR) << "Activation type error";
      return RET_ERROR;
  }
}

bool AffineFp32CPUKernel::CheckAffineValid() {
  // input_data , weight, (bias), tensorArrayRead, bias maybe not exist
  if (in_tensors_.size() < kAffineMinInputNum) {
    return false;
  }
  if (out_tensors_.size() != kAffineMaxOutputNum) {
    return false;
  }
  return true;
}

int AffineFp32CPUKernel::CheckActivationValid() {
  const std::set<schema::ActivationType> valid_activation_types = {
    schema::ActivationType_RELU,     schema::ActivationType_RELU6,    schema::ActivationType_SIGMOID,
    schema::ActivationType_TANH,     schema::ActivationType_HSWISH,   schema::ActivationType_SWISH,
    schema::ActivationType_HSIGMOID, schema::ActivationType_SOFTPLUS,
  };

  if (valid_activation_types.find(schema::ActivationType(affine_parameter_->activation_type_)) ==
      valid_activation_types.end()) {
    MS_LOG(ERROR) << "Activation fp32 not support type: " << affine_parameter_->activation_type_;
    return RET_PARAM_INVALID;
  }
  return RET_OK;
}

AffineFp32CPUKernel::~AffineFp32CPUKernel() {
  if (full_mult_kernel_ == nullptr) {
    delete full_mult_kernel_;
    full_mult_kernel_ = nullptr;
  }

  if (increment_mult_kernel_ == nullptr) {
    delete increment_mult_kernel_;
    increment_mult_kernel_ = nullptr;
  }

  if (full_input_ == nullptr) {
    delete full_input_;
    full_input_ = nullptr;
  }

  if (increment_input_ == nullptr) {
    delete increment_input_;
    increment_input_ = nullptr;
  }

  if (increment_output_ == nullptr) {
    delete increment_output_;
    increment_output_ = nullptr;
  }

  if (previous_output_ == nullptr) {
    delete previous_output_;
    previous_output_ = nullptr;
  }
}

int AffineFp32CPUKernel::FullRunInit() {
  // Construct Splice Param
  src_to_dst_row_offset_ =
    *std::min_element(affine_parameter_->context_, affine_parameter_->context_ + affine_parameter_->context_size_);
  std::vector<int> src_shape = in_tensors_.at(kInputIndex)->shape();
  std::vector<int> dst_shape = full_input_->shape();
  if (src_shape.size() != dst_shape.size() || src_shape.size() != kInputSize2 || dst_shape.size() != kInputSize2) {
    MS_LOG(ERROR) << "splice kernel src_shape size not equal to dst_shape size";
    return RET_ERROR;
  }
  // src and dst shape: {batch, row, col}
  splice_src_row_ = src_shape.at(kInputRow);
  splice_src_col_ = src_shape.at(kInputCol);
  splice_dst_row_ = dst_shape.at(kInputRow);
  splice_dst_col_ = dst_shape.at(kInputCol);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(splice_src_col_, affine_parameter_->context_size_, RET_ERROR);
  if (splice_src_col_ * affine_parameter_->context_size_ != splice_dst_col_) {
    MS_LOG(ERROR) << "splice kernel src_col not match dst_col";
    return RET_ERROR;
  }
  for (int r = 0; r < splice_dst_row_; ++r) {
    for (int off = 0; off < affine_parameter_->context_size_; ++off) {
      int r_off = r - src_to_dst_row_offset_ + affine_parameter_->context_[off];
      if (r_off < 0) {
        MS_LOG(ERROR) << "splice row index out of range";
        return RET_ERROR;
      }
      if (r_off >= splice_src_row_) {
        MS_LOG(ERROR) << "splice row index out of range";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int AffineFp32CPUKernel::IncrementInit() {
  auto out_tensor = out_tensors_.at(kOutputIndex);
  auto out_shape = out_tensor->shape();
  matmul_col_ = out_shape.at(kInputCol);
  matmul_row_ = out_shape.at(kInputRow);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul_row_, matmul_col_, RET_ERROR);
  if (out_tensor->Size() != matmul_row_ * matmul_col_ * sizeof(float)) {
    MS_LOG(ERROR) << "size mismatch!";
    MS_LOG(ERROR) << "out_tensor->Size() = " << out_tensor->Size();
    MS_LOG(ERROR) << "matmul_row * matmul_col * sizeof(float) = " << matmul_row_ * matmul_col_ * sizeof(float);
    return RET_PARAM_INVALID;
  }
  previous_output_ =
    reinterpret_cast<float *>(this->ms_context_->allocator->Malloc(matmul_row_ * matmul_col_ * sizeof(float)));
  return RET_OK;
}

int AffineFp32CPUKernel::Prepare() {
  // Update shape info of input and output
  if (!CheckAffineValid()) {
    MS_LOG(ERROR) << "Affine Parameter not vailed";
    return RET_PARAM_INVALID;
  }
  if (affine_parameter_->activation_type_ != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    auto ret = CheckActivationValid();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "CheckActivationValid failed";
      return ret;
    }
  }
  auto ret = ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ReSize failed";
    return ret;
  }
  ret = FullRunInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FullRunInit failed";
    return ret;
  }
  ret = IncrementInit();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "IncrementInit failed";
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return RET_OK;
}

int AffineFp32CPUKernel::ReSize() {
  if (full_mult_kernel_ == nullptr) {
    // need to select actual execute kernel here
    full_mult_kernel_ = FullMatmulKernelCreate();
    if (full_mult_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for full_mult_kernel_, got a nullptr.";
      return RET_ERROR;
    }
    full_mult_kernel_->set_name(this->name_);
  }
  auto ret = full_mult_kernel_->ReSize();
  if (ret != RET_OK) {
    return ret;
  }

  if (increment_mult_kernel_ == nullptr) {
    // need to select actual execute kernel here
    increment_mult_kernel_ = IncrementMatmulKernelCreate();
    if (increment_mult_kernel_ == nullptr) {
      MS_LOG(ERROR) << "Selecting execute kernel failed for increment_mult_kernel_, got a nullptr.";
      return RET_ERROR;
    }
    increment_mult_kernel_->set_name(this->name_);
  }
  ret = increment_mult_kernel_->ReSize();
  if (ret != RET_OK) {
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *AffineFp32CPUKernel::FullMatmulKernelCreate() {
  auto input_shape = in_tensors_.front()->shape();
  int out_dim = affine_parameter_->output_dim_;
  int context_min = affine_parameter_->context_[0];
  int context_max = affine_parameter_->context_[affine_parameter_->context_size_ - 1];
  std::vector<int> splice_output_shape = {1, input_shape.at(1) - (context_max - context_min), out_dim};

  full_input_ = new (std::nothrow) lite::Tensor(kNumberTypeFloat32, splice_output_shape);
  MS_CHECK_TRUE_MSG(full_input_ != nullptr, nullptr, "Create a new-tensor failed.");

  if (in_tensors_.size() < kAffineMinInputNum) {
    MS_LOG(ERROR) << "wrong affine input size";
    return nullptr;
  }

  std::vector<lite::Tensor *> input_tensors;
  // For affine op, the possible inputs are:
  // { input, weight, bias, tensor_array_read }
  // { input, weight, tensor_array_read }
  if (in_tensors_.size() == kAffineMaxInputNum) {
    input_tensors = {full_input_, in_tensors_.at(kWeightIndex), in_tensors_.at(kBiasIndex)};
  } else {
    input_tensors = {full_input_, in_tensors_.at(kWeightIndex)};
  }

  OpParameter *params = MatmulParameterCreate();
  if (params == nullptr) {
    MS_LOG(ERROR) << "MatmulParameterCreate failed.";
    return nullptr;
  }

  kernel::LiteKernel *kernel = new (std::nothrow) kernel::MatmulCPUKernel(
    params, input_tensors, out_tensors_, static_cast<const lite::InnerContext *>(this->ms_context_));

  if (kernel != nullptr) {
    auto ret = kernel->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "matmul kernel init failed.";
      delete kernel;
      return nullptr;
    }
  }
  return kernel;
}

kernel::LiteKernel *AffineFp32CPUKernel::IncrementMatmulKernelCreate() {
  auto input_shape = in_tensors_.front()->shape();
  int src_col = input_shape.at(input_shape.size() - 1);
  int context_dims = affine_parameter_->context_size_;
  int affine_splice_output_col = affine_parameter_->output_dim_;

  MS_CHECK_INT_MUL_NOT_OVERFLOW(context_dims, src_col, nullptr);
  if (context_dims * src_col != affine_splice_output_col) {
    MS_LOG(ERROR) << "context_dims * src_col_ != affine_splice_output_col: " << context_dims << " * " << src_col
                  << " != " << affine_splice_output_col;
    return nullptr;
  }

  increment_input_ = new (std::nothrow) lite::Tensor(kNumberTypeFloat32, {1, 1, affine_splice_output_col});
  MS_CHECK_TRUE_MSG(increment_input_ != nullptr, nullptr, "Create a new-tensor failed.");

  // matmul_output == 1 * matmul_col
  int matmul_col = out_tensors_.front()->shape().back();
  increment_output_ = new (std::nothrow) lite::Tensor(kNumberTypeFloat32, {1, 1, matmul_col});
  MS_CHECK_TRUE_MSG(increment_output_ != nullptr, nullptr, "Create a new-tensor failed.");
  increment_output_->MallocData();

  if (in_tensors_.size() < kAffineMinInputNum) {
    MS_LOG(ERROR) << "wrong affine input size";
    return nullptr;
  }

  std::vector<lite::Tensor *> input_tensors;
  // For affine op, the possible inputs are:
  // { input, weight, bias, tensor_array_read }
  // { input, weight, tensor_array_read }
  if (in_tensors_.size() == kAffineMaxInputNum) {
    input_tensors = {increment_input_, in_tensors_.at(kWeightIndex), in_tensors_.at(kBiasIndex)};
  } else {
    input_tensors = {increment_input_, in_tensors_.at(kWeightIndex)};
  }

  OpParameter *params = MatmulParameterCreate();
  if (params == nullptr) {
    MS_LOG(ERROR) << "MatmulParameterCreate failed.";
    return nullptr;
  }

  kernel::LiteKernel *kernel = new (std::nothrow) kernel::MatmulCPUKernel(
    params, input_tensors, {increment_output_}, static_cast<const lite::InnerContext *>(this->ms_context_));

  if (kernel != nullptr) {
    auto ret = kernel->Prepare();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "matmul kernel init failed.";
      delete kernel;
      return nullptr;
    }
  }

  return kernel;
}

OpParameter *AffineFp32CPUKernel::MatmulParameterCreate() {
  auto origin_matmul = affine_parameter_->matmul_parameter_;
  auto *matmul_param = reinterpret_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (matmul_param == nullptr) {
    MS_LOG(ERROR) << "malloc MatMulParameter failed.";
    return nullptr;
  }
  memset(matmul_param, 0, sizeof(MatMulParameter));
  matmul_param->op_parameter_.type_ = origin_matmul->op_parameter_.type_;
  matmul_param->b_transpose_ = origin_matmul->b_transpose_;
  matmul_param->a_transpose_ = origin_matmul->a_transpose_;
  matmul_param->has_bias_ = origin_matmul->has_bias_;
  matmul_param->act_type_ = origin_matmul->act_type_;
  matmul_param->op_parameter_.thread_num_ = this->context()->thread_num_;
  return reinterpret_cast<OpParameter *>(matmul_param);
}

int AffineFp32CPUKernel::Run() {
  if (full_run_) {
    return FullMatmulRun();
  } else {
    return IncrementMatmulRun();
  }
}

int AffineFp32CPUKernel::IncrementSplice() {
  auto input_data = static_cast<const float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto output_data = static_cast<float *>(increment_input_->MutableData());
  int forward_offset = splice_dst_row_ - 1 - src_to_dst_row_offset_;
  // splice last context input to outputs
  for (int i = 0; i < affine_parameter_->context_size_; ++i) {
    auto forward_row = forward_offset + affine_parameter_->context_[i];
    auto src_offset_ptr = input_data + forward_row * splice_src_col_;
    auto splice_offset_ptr = output_data + i * splice_src_col_;
    memcpy(splice_offset_ptr, src_offset_ptr, splice_src_col_ * sizeof(float));
  }
  return RET_OK;
}

int AffineFp32CPUKernel::IncrementMatmulRun() {
  auto ret = IncrementSplice();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "IncrementSplice failed.";
    return ret;
  }

  if (increment_mult_kernel_ == nullptr) {
    MS_LOG(ERROR) << "increment_mult_kernel_ is null, can't call increment_mult_kernel_->Run().";
    return RET_NULL_PTR;
  }
  ret = increment_mult_kernel_->Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "increment_mult_kernel_->Run() failed";
    return ret;
  }

  if (increment_output_->data() == nullptr) {
    MS_LOG(ERROR) << "increment_output_ data is null.";
    return RET_NULL_PTR;
  }

  if (affine_parameter_->activation_type_ != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    ret = DoActivation(increment_output_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DoActivation() failed";
      return ret;
    }
  }

  auto out_tensor = out_tensors_.at(kOutputIndex);
  auto matmul_output = static_cast<float *>(increment_output_->MutableData());
  auto output_data = static_cast<float *>(out_tensor->MutableData());

  memcpy(output_data, previous_output_ + matmul_col_, (matmul_row_ - 1) * matmul_col_ * sizeof(float));
  memcpy(output_data + (matmul_row_ - 1) * matmul_col_, matmul_output, matmul_col_ * sizeof(float));
  memcpy(previous_output_, output_data, matmul_row_ * matmul_col_ * sizeof(float));
  return RET_OK;
}

int AffineFp32CPUKernel::FullSpliceRun() {
  auto input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->MutableData());
  auto output_data = reinterpret_cast<float *>(full_input_->MutableData());
  // Splice Run
  if (input_data == nullptr || output_data == nullptr) {
    MS_LOG(ERROR) << "splice kernel input or output data is nullptr";
    return RET_ERROR;
  }
  for (int r = 0; r < splice_dst_row_; ++r) {
    for (int off = 0; off < affine_parameter_->context_size_; ++off) {
      int r_off = r - src_to_dst_row_offset_ + affine_parameter_->context_[off];
      const float *tmp_src_data = input_data + static_cast<int64_t>(r_off) * splice_src_col_;
      float *tmp_dst_data = output_data + r * splice_dst_col_;
      memcpy(tmp_dst_data + off * splice_src_col_, tmp_src_data, splice_src_col_ * sizeof(float));
    }
  }

  return RET_OK;
}

int AffineFp32CPUKernel::FullMatmulRun() {
  // Run Splice
  auto ret = FullSpliceRun();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "run FullSplice failed";
    return ret;
  }
  // Run Matmul
  if (full_mult_kernel_ == nullptr) {
    MS_LOG(ERROR) << "full_mult_kernel_ is null, can't call full_mult_kernel_->Run().";
    return RET_NULL_PTR;
  }
  ret = full_mult_kernel_->Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "full_mult_kernel_->Run() failed";
    return ret;
  }

  if (affine_parameter_->activation_type_ != schema::ActivationType::ActivationType_NO_ACTIVATION) {
    ret = DoActivation(out_tensors_.at(kOutputIndex));
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DoActivation() failed";
      return ret;
    }
  }
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto output_data = static_cast<float *>(output_tensor->MutableData());
  memcpy(previous_output_, output_data, matmul_row_ * matmul_col_ * sizeof(float));
  full_run_ = false;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Affine, LiteKernelCreator<AffineFp32CPUKernel>)
}  // namespace mindspore::kernel

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

#include "coder/opcoders/nnacl/fp32/matmul_fp32_base_coder.h"
#include <string>
#include <vector>
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "wrapper/fp32/matmul_fp32_wrapper.h"
#include "coder/opcoders/nnacl/dequant/de_quant.h"

using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::lite::micro::nnacl {

int MatMulFP32BaseCoder::ReSize() {
  ResizeParameter();
  thread_count_ = MSMIN(thread_num_, UP_DIV(params_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_align_, col_tile_), thread_count_);
  // can not call Malloc in DoCode,so move this runtime init to final resize
  if (!params_->a_const_) {
    MS_CHECK_RET_CODE(InitBufferA(), "InitBufferA failed");
  }
  if (!params_->b_const_) {
    MS_CHECK_RET_CODE(InitBufferB(), "InitBufferB failed");
  }
  return RET_OK;
}

int MatMulFP32BaseCoder::InitBiasData() {
  if (input_tensors_.size() == 3) {
    int max_bias_data = UP_ROUND(bias_tensor_->ElementsNum(), C16NUM);
    bias_pack_ptr_size_ = static_cast<size_t>(max_bias_data * sizeof(float));
    bias_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(bias_ptr_);
  }
  return RET_OK;
}

void MatMulFP32BaseCoder::InitParameter() {
  row_tile_ = C12NUM;
  if (target_ == kARM32A) {
    col_tile_ = C4NUM;
  } else {
    col_tile_ = C8NUM;
  }
}

void MatMulFP32BaseCoder::ResizeParameter() {
  if (params_->row_ == 1) {
    vec_matmul_ = true;
  }
  params_->row_align_ = vec_matmul_ ? 1 : UP_ROUND(params_->row_, row_tile_);
  params_->col_align_ = vec_matmul_ ? params_->col_ : UP_ROUND(params_->col_, col_tile_);
}

int MatMulFP32BaseCoder::InitBufferA() {
  if (a_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  a_pack_ptr_size_ = static_cast<size_t>(params_->batch * params_->row_align_ * params_->deep_ * sizeof(float));
  if (params_->a_const_) {
    a_pack_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  } else {
    a_pack_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, a_pack_ptr_size_, kWorkspace));
  }
  MS_CHECK_PTR(a_pack_ptr_);
  return RET_OK;
}

int MatMulFP32BaseCoder::InitBufferB() {
  if (b_pack_ptr_ != nullptr) {
    return RET_OK;
  }
  b_pack_ptr_size_ = static_cast<size_t>(params_->batch * params_->col_align_ * params_->deep_ * sizeof(float));
  if (params_->b_const_) {
    b_pack_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, kOnlineSize, kOnlinePackWeight));
  } else {
    b_pack_ptr_ = reinterpret_cast<float *>(allocator_->Malloc(kNumberTypeFloat32, b_pack_ptr_size_, kWorkspace));
  }
  MS_CHECK_PTR(b_pack_ptr_);
  return RET_OK;
}

int MatMulFP32BaseCoder::InitMatrixA(const float *src_ptr) {
  ::InitMatrixA(src_ptr, a_pack_ptr_, params_, vec_matmul_);
  return RET_OK;
}

int MatMulFP32BaseCoder::InitMatrixB(const float *src_ptr) {
  ::InitMatrixB(src_ptr, b_pack_ptr_, params_, vec_matmul_);
  return RET_OK;
}

int MatMulFP32BaseCoder::Init() {
  thread_count_ = thread_num_;
  ResizeParameter();
  MS_CHECK_RET_CODE(InitBiasData(), "InitBiasData failed");
  if (params_->a_const_) {
    MS_CHECK_RET_CODE(InitBufferA(), "InitBufferA failed");
  }
  if (params_->b_const_) {
    MS_CHECK_RET_CODE(InitBufferB(), "InitBufferB failed");
  }
  return RET_OK;
}

int MatMulFP32BaseCoder::Prepare(CoderContext *const context) { return RET_OK; }

int MatMulFP32BaseCoder::DoCode(CoderContext *const context) {
  // generate code .h .c
  std::vector<std::string> asm_files;
  if (target_ == kARM32A) {
    asm_files = {"MatmulFp32.S", "MatmulFp32Opt.S", "MatmulFp32Opt12x4.S"};
  } else if (target_ == kARM64) {
    asm_files = {"MatmulFp32.S", "MatmulFp32Opt.S", "MatVecMulFp32.S"};
  }
  std::vector<std::string> h_files = {"nnacl/fp32/matmul_fp32.h", "wrapper/fp32/matmul_fp32_wrapper.h"};
  std::vector<std::string> c_files = {"matmul_fp32.c", "matmul_fp32_wrapper.c"};
  if (de_quant_flag_) {
    h_files.emplace_back("wrapper/fp32/dequant_int8_to_fp32_wrapper.h");
    c_files.emplace_back("dequant_int8_to_fp32_wrapper.c");
  }
  Collect(context, h_files, c_files, asm_files);
  NNaclFp32Serializer code;
  NNaclFp32Serializer init_code;
  code.CodeStruct("mat_mul_parameter", *params_);
  init_code.CodeStruct("mat_mul_parameter", *params_);
  // do bias packing to init
  if (bias_ptr_) {
    init_code.CodeMallocExpression(bias_ptr_, bias_pack_ptr_size_);
    init_code.CodeFunction("memcpy", bias_ptr_, bias_tensor_, bias_pack_ptr_size_);
  }

  // Get Tensor Pointer
  std::string a_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string b_str = allocator_->GetRuntimeAddr(filter_tensor_);
  std::string c_str = allocator_->GetRuntimeAddr(output_tensor_);
  std::string a_pack_str = allocator_->GetRuntimeAddr(a_pack_ptr_);
  std::string b_pack_str = allocator_->GetRuntimeAddr(b_pack_ptr_);

  // do const value packing to init
  if (!params_->a_const_) {
    code.CodeFunction("InitMatrixA", input_tensor_, a_pack_ptr_, "&mat_mul_parameter", vec_matmul_);
    init_code.CodeMallocExpression(b_pack_ptr_, b_pack_ptr_size_);
    std::string b_src_str = b_str;
    if (de_quant_flag_) {
      // reuse to b_pack_str
      b_src_str = Dequant::GetInstance()->de_quant_buffer_str();
      std::string de_quant_function = Dequant::GetInstance()->GetMicroDeQuantFunction(filter_tensor_, b_str);
      init_code << de_quant_function;
    }
    // b_pack_str has been memset, no need to memset
    init_code.CodeFunction("InitMatrixB", b_src_str, b_pack_ptr_, "&mat_mul_parameter", vec_matmul_);
  }
  if (!params_->b_const_) {
    init_code.CodeMallocExpression(a_pack_str, a_pack_ptr_size_);
    std::string a_src_str = a_str;
    if (de_quant_flag_) {
      // reuse to a_pack_str
      a_src_str = Dequant::GetInstance()->de_quant_buffer_str();
      std::string de_quant_function = Dequant::GetInstance()->GetMicroDeQuantFunction(input_tensor_, a_str);
      init_code << de_quant_function;
    }
    // a_pack_str has been memset, no need to memset
    init_code.CodeFunction("InitMatrixA", a_src_str, a_pack_ptr_, "&mat_mul_parameter", vec_matmul_);
    code.CodeFunction("InitMatrixB", filter_tensor_, b_pack_ptr_, "&mat_mul_parameter", vec_matmul_);
  }

  int current_stride_oc = thread_stride_ * col_tile_;
  int current_rest_oc = params_->col_ - kDefaultTaskId * thread_stride_ * col_tile_;
  int cur_oc = MSMIN(current_stride_oc, current_rest_oc);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  code << "for (int i = 0; i < " << params_->batch << "; ++i) {\n";
  if (vec_matmul_) {
    code << "\t\tfloat *batch_a_ptr = " << a_pack_str << " + i * " << params_->deep_ << ";\n";
    code << "\t\tfloat *batch_b_ptr = " << b_pack_str << " + i * " << params_->deep_ * params_->col_ << ";\n";
    code << "\t\tfloat *batch_c_ptr = " << c_str << " + i * " << params_->row_ * params_->col_ << ";\n";
  } else {
    code << "\t\tfloat *batch_a_ptr = " << a_pack_str << " + i * " << params_->row_align_ * params_->deep_ << ";\n";
    code << "\t\tfloat *batch_b_ptr = " << b_pack_str << " + i * " << params_->deep_ * params_->col_align_ << ";\n";
    code << "\t\tfloat *batch_c_ptr = " << c_str << " + i * " << params_->row_ * params_->col_ << ";\n";
  }

  if (vec_matmul_) {
    code.CodeFunction("MatVecMulFp32", "batch_a_ptr", "batch_b_ptr", "batch_c_ptr", bias_ptr_, params_->act_type_,
                      params_->deep_, cur_oc);
  } else {
    code.CodeFunction("MatMulOpt", "batch_a_ptr", "batch_b_ptr", "batch_c_ptr", bias_ptr_, params_->act_type_,
                      params_->deep_, params_->row_, cur_oc, params_->col_, "OutType_Nhwc");
  }
  code << "\t\t}\n";

  context->AppendCode(code.str());
  context->AppendInitCode(init_code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl

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

#include "coder/opcoders/nnacl/int8/matmul_int8_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
namespace mindspore::lite::micro::nnacl {

int MatMulInt8Coder::ReSize(CoderContext *const context) {
  int batch = 1;
  std::vector<int> x_shape = input_tensor_->shape();
  std::vector<int> o_shape = output_tensor_->shape();
  if (x_shape.size() <= 2 || o_shape.size() <= 2) {
    MS_LOG(ERROR) << "x_shape.size() or o_shape.size() is less than two";
    return RET_ERROR;
  }
  for (size_t i = 0; i < x_shape.size() - 2; ++i) {
    batch *= x_shape.at(i);
  }
  params_->batch = batch;
  params_->row_ = o_shape.at(o_shape.size() - 2);
  params_->col_ = o_shape.at(o_shape.size() - 1);
  params_->deep_ = params_->a_transpose_ ? x_shape.at(x_shape.size() - 2) : x_shape.at(x_shape.size() - 1);
  params_->row_4_ = UP_ROUND(params_->row_, C4NUM);
  params_->col_4_ = UP_ROUND(params_->col_, C4NUM);
  params_->deep_16_ = UP_ROUND(params_->deep_, C16NUM);

  a_pack_ptr_size_ = static_cast<size_t>(params_->row_4_ * params_->deep_16_ * sizeof(int8_t));
  a_pack_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, a_pack_ptr_size_, kOfflinePackWeight));
  MS_CHECK_PTR(a_pack_ptr_);
  input_sums_size_ = static_cast<size_t>(params_->row_4_ * sizeof(int));
  input_sums_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, input_sums_size_, kOfflinePackWeight));
  MS_CHECK_PTR(input_sums_);
  b_pack_batch_ptr_size_ = static_cast<size_t>(params_->batch * params_->col_4_ * params_->deep_16_ * sizeof(int8_t));
  if (params_->b_const_) {
    b_pack_batch_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(b_pack_batch_ptr_);
    weight_bias_sums_batch_ =
      reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, kOnlineSize, kOnlinePackWeight));
  } else {
    b_pack_batch_ptr_ =
      reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, b_pack_batch_ptr_size_, kOfflinePackWeight));
    MS_CHECK_PTR(b_pack_batch_ptr_);
    weight_bias_sums_batch_ =
      reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, weight_bias_sums_batch_size_, kOfflinePackWeight));
  }
  MS_CHECK_PTR(weight_bias_sums_batch_);
  if (input_tensors_.size() == 3) {
    bias_prt_size_ = static_cast<size_t>(params_->col_4_ * sizeof(int));
    bias_ptr_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(bias_ptr_);
  } else {
    bias_ptr_ = nullptr;
  }
  thread_count_ = MSMIN(thread_num_, UP_DIV(params_->col_4_, C4NUM));
  thread_stride_ = UP_DIV(UP_DIV(params_->col_4_, C4NUM), thread_count_);

  std::vector<QuantArg> params = input_tensor_->quant_params();
  MS_CHECK_TRUE(!params.empty(), "params is empty");
  quant_params_.input.zp_ = params.front().zeroPoint;
  quant_params_.input.scale_ = static_cast<float>(params.front().scale);

  params = filter_tensor_->quant_params();
  MS_CHECK_TRUE(!params.empty(), "params is empty");
  quant_params_.weight.zp_ = params.front().zeroPoint;
  quant_params_.weight.scale_ = static_cast<float>(params.front().scale);

  params = output_tensor_->quant_params();
  MS_CHECK_TRUE(!params.empty(), "params is empty");
  quant_params_.output.zp_ = params.front().zeroPoint;
  quant_params_.output.scale_ = static_cast<float>(params.front().scale);
  double real_multiplier = quant_params_.input.scale_ * quant_params_.weight.scale_ / quant_params_.output.scale_;
  QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_params_.quant_multiplier, &quant_params_.left_shift,
                                            &quant_params_.right_shift);
  if (params_->b_const_) {
    NNaclInt8Serializer init_code;
    if (bias_ptr_) {
      init_code.CodeMallocExpression(bias_ptr_, bias_prt_size_);
      init_code.CodeFunction("memset", bias_ptr_, 0, bias_prt_size_);
      init_code.CodeFunction("memcpy", bias_ptr_, bias_tensor_->data_c(), bias_prt_size_);
    }
    init_code.CodeMallocExpression(weight_bias_sums_batch_, weight_bias_sums_batch_size_);
    init_code.CodeFunction("memset", weight_bias_sums_batch_, 0, weight_bias_sums_batch_size_);
    init_code.CodeMallocExpression(b_pack_batch_ptr_, b_pack_batch_ptr_size_);
    init_code.CodeFunction("memset", b_pack_batch_ptr_, 0, b_pack_batch_ptr_size_);

    init_code << "int tmp_weight_zp = " << quant_params_.weight.zp_ << ";\n";
    init_code.CodeFunction("InitIn8MatrixB", filter_tensor_->data_c(), weight_bias_sums_batch_, b_pack_batch_ptr_,
                           params_->batch, params_->deep_, params_->col_, params_->col_4_, params_->deep_16_,
                           quant_params_.input.zp_, "&tmp_weight_zp", bias_ptr_, params_->b_transpose_);
    context->AppendInitCode(init_code.str());
  }
  return RET_OK;
}

int MatMulInt8Coder::Init() {
  params_ = reinterpret_cast<MatMulParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  }
  params_->b_const_ = (filter_tensor_->data_c() != nullptr);
  return RET_OK;
}

int MatMulInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Init(), "MatMulInt8Coder Init failed");
  MS_CHECK_RET_CODE(ReSize(context), "MatMulInt8Coder ReSize failed");
  return RET_OK;
}

int MatMulInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/common_func.h", "nnacl/int8/common_func_int8.h", "nnacl/int8/matmul_int8.h"},
          {"common_func.c", "common_func_int8.c", "matmul_int8.c"});

  std::string a_ptr_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string c_ptr_str = allocator_->GetRuntimeAddr(output_tensor_);
  int a_stride = params_->row_ * params_->deep_;
  int c_stride = params_->row_ * params_->col_;

  NNaclInt8Serializer code;
  code.precision(kPrecision);
  int task_id = 0;
  int cur_oc = MSMIN(thread_stride_, UP_DIV(params_->col_4_, C4NUM) - task_id * thread_stride_);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  code << "int tmp_weight_zp = " << quant_params_.weight.zp_ << ";\n";
  if (!params_->b_const_) {
    code.CodeFunction("InitIn8MatrixB", filter_tensor_->data_c(), weight_bias_sums_batch_, b_pack_batch_ptr_,
                      params_->batch, params_->deep_, params_->col_, params_->col_4_, params_->deep_16_,
                      quant_params_.input.zp_, "&tmp_weight_zp", bias_ptr_, params_->b_transpose_);
  }
  std::string b_batch_str = allocator_->GetRuntimeAddr(b_pack_batch_ptr_);
  std::string weight_bias_sums_batch_str = allocator_->GetRuntimeAddr(weight_bias_sums_batch_);
  code.CodeFunction("memset", input_sums_, 0, input_sums_size_);
  code.CodeFunction("memset", a_pack_ptr_, 0, a_pack_ptr_size_);
  code << "for (int i = 0; i < " << params_->batch << "; ++i) {\n";
  code << "    int8_t* cur_a_ptr = " << a_ptr_str << " + i * " << a_stride << ";\n";
  if (params_->a_transpose_) {
    code.CodeFunction("RowMajor2Col16x4MajorInt8", "cur_a_ptr", params_->deep_, params_->row_, a_pack_ptr_);
    code.CodeFunction("CalcInputSums", "cur_a_ptr", params_->deep_, quant_params_.weight.zp_, input_sums_, ColMajor);
  } else {
    code.CodeFunction("RowMajor2Row16x4MajorInt8", "cur_a_ptr", a_pack_ptr_, params_->row_, params_->deep_);
    code.CodeFunction("CalcInputSums", "cur_a_ptr", params_->row_, params_->deep_, quant_params_.weight.zp_,
                      input_sums_, RowMajor);
  }
  code << "    b_pack_ptr_ = " << b_batch_str << " + i * " << params_->col_4_ * params_->deep_16_ << ";\n";
  code << "    weight_bias_sums_ = " << weight_bias_sums_batch_str << " + i * " << params_->col_4_ << ";\n";
  code << "    c_ptr_ = " << c_ptr_str << " + i * " << c_stride << ";\n";
  int cur_oc_res = MSMIN(thread_stride_ * C4NUM, params_->col_ - task_id * thread_stride_ * C4NUM);

  code << "  int8_t* cur_b = b_pack_ptr_ + " << task_id * thread_stride_ * C4NUM * params_->deep_16_ << ";\n";
  code << "  int32_t* cur_bias = weight_bias_sums_ + " << task_id * thread_stride_ * C4NUM << ";\n";
  code << "  int8_t *cur_c = c_ptr_ + " << task_id * thread_stride_ * C4NUM << ";\n";
  code << "  static const int left_shift = " << quant_params_.left_shift << ";\n";
  code << "  static const int right_shift = " << quant_params_.right_shift << ";\n";
  code << "  static const int quant_multiplier = " << quant_params_.quant_multiplier << ";\n";
  if (target_ == kARM64) {
    code.CodeFunction("MatmulInt8Neon64", "cur_a_ptr", "cur_b", "cur_c", params_->row_4_, cur_oc * C4NUM,
                      params_->deep_16_, input_sums_, "cur_bias", INT8_MIN, INT8_MAX, quant_params_.output.zp_,
                      "&quant_multiplier", "&left_shift", "&right_shift", params_->row_, cur_oc_res, params_->col_,
                      false);
  } else {
    code.CodeFunction("MatMulInt8_16x4_r", "cur_a_ptr", "cur_b", "cur_c", params_->row_, cur_oc_res, params_->deep_16_,
                      params_->col_, input_sums_, "cur_bias", "&left_shift", "&right_shift", "&quant_multiplier",
                      quant_params_.output.zp_, INT8_MIN, INT8_MAX, false);
  }
  code << "}\n";
  MS_LOG(DEBUG) << "FullConnectionInt8Coder has been called";
  context->AppendCode(code.str());

  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl

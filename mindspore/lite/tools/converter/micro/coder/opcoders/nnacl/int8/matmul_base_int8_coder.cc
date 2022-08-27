/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/matmul_base_int8_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/wrapper/int8/matmul_int8_wrapper.h"
#include "src/litert/kernel/cpu/fp32/matmul_fp32_base.h"

namespace mindspore::lite::micro::nnacl {
int MatMulBaseInt8Coder::ReSize(CoderContext *const context) {
  auto ret = mindspore::kernel::MatmulFp32BaseCPUKernel::InitBroadcastParams(
    input_tensors_[kInputIndex]->shape(), input_tensors_[kWeightIndex]->shape(), param_, &a_offset_, &b_offset_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitBroadcastParams failed.";
    return RET_ERROR;
  }
  ResizeParameter();
  if (InitTmpBuffer() != RET_OK) {
    FreeQuantParam();
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulBaseInt8Coder::InitTmpBuffer() {
  a_pack_ptr_size_ = param_->row_align_ * param_->deep_16_ * sizeof(int8_t);
  pack_a_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, a_pack_ptr_size_, kWorkspace));
  MS_CHECK_PTR(pack_a_ptr_);
  b_pack_ptr_size_ = param_->batch * param_->col_align_ * param_->deep_16_ * sizeof(int8_t);
  if (param_->b_const_) {
    if (target_ == kCortex_M) {
      pack_b_ptr_ =
        reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, b_pack_ptr_size_, kOfflinePackWeight));
    } else {
      pack_b_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
    }
  } else {
    pack_b_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, b_pack_ptr_size_, kWorkspace));
  }
  MS_CHECK_PTR(pack_b_ptr_);
  input_sums_size_ = static_cast<size_t>(param_->row_align_ * sizeof(int));
  input_sums_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt32, input_sums_size_, kWorkspace));
  MS_CHECK_PTR(input_sums_);
  weight_bias_sums_size_ = static_cast<size_t>(param_->batch * param_->col_align_ * sizeof(int));
  if (param_->b_const_) {
    if (target_ == kCortex_M) {
      weight_bias_sums_ =
        reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt32, weight_bias_sums_size_, kOfflinePackWeight));
    } else {
      weight_bias_sums_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt32, kOnlineSize, kOnlinePackWeight));
    }
  } else {
    weight_bias_sums_ =
      reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt32, weight_bias_sums_size_, kWorkspace));
  }
  MS_CHECK_PTR(weight_bias_sums_);
  if (param_->b_const_ && target_ == kCortex_M) {
    // For MCU init filter offline
    memset(weight_bias_sums_, 0, weight_bias_sums_size_);
    if (bias_tensor_ != nullptr) {
      InitInt8MatrixB(reinterpret_cast<int8_t *>(filter_tensor_->data()), weight_bias_sums_, pack_b_ptr_, param_->batch,
                      param_->deep_, param_->col_, param_->col_align_, param_->deep_16_, quant_.input_.zp_,
                      quant_.filter_zp_, reinterpret_cast<int *>(bias_tensor_->data()), param_->b_transpose_,
                      filter_per_channel_);
      MemoryAllocator::GetInstance()->FreeTensor(bias_tensor_);
      bias_tensor_ = nullptr;
    } else {
      InitInt8MatrixB(reinterpret_cast<int8_t *>(filter_tensor_->data()), weight_bias_sums_, pack_b_ptr_, param_->batch,
                      param_->deep_, param_->col_, param_->col_align_, param_->deep_16_, quant_.input_.zp_,
                      quant_.filter_zp_, nullptr, param_->b_transpose_, filter_per_channel_);
    }

    MemoryAllocator::GetInstance()->FreeTensor(filter_tensor_);
    filter_tensor_ = nullptr;
  }
  return RET_OK;
}

MatMulBaseInt8Coder::~MatMulBaseInt8Coder() { FreeQuantParam(); }

void MatMulBaseInt8Coder::ResizeParameter() {
  param_->row_align_ = UP_ROUND(param_->row_, row_tile_);
  param_->col_align_ = UP_ROUND(param_->col_, col_tile_);
  param_->deep_16_ = UP_ROUND(param_->deep_, C16NUM);

  thread_count_ = MSMIN(kDefaultThreadNum, UP_DIV(param_->col_align_, col_tile_));
  thread_stride_ = UP_DIV(UP_DIV(param_->col_align_, col_tile_), thread_count_);
}

void MatMulBaseInt8Coder::FreeQuantParam() {
  if (quant_.filter_scale_ != nullptr) {
    free(quant_.filter_scale_);
    quant_.filter_scale_ = nullptr;
  }
  if (quant_.filter_zp_ != nullptr) {
    free(quant_.filter_zp_);
    quant_.filter_zp_ = nullptr;
  }
  if (quant_.left_shift_ != nullptr) {
    free(quant_.left_shift_);
    quant_.left_shift_ = nullptr;
  }
  if (quant_.right_shift_ != nullptr) {
    free(quant_.right_shift_);
    quant_.right_shift_ = nullptr;
  }
  if (quant_.quant_multiplier_ != nullptr) {
    free(quant_.quant_multiplier_);
    quant_.quant_multiplier_ = nullptr;
  }
}

int MatMulBaseInt8Coder::MallocQuantParam() {
  std::vector<LiteQuantParam> weight_quant_params = filter_tensor_->quant_params();
  int col = filter_tensor_->shape().front();
  filter_per_channel_ = (weight_quant_params.size() > 1);
  weight_quant_num_ = filter_per_channel_ ? col : 1;
  quant_.filter_scale_ = reinterpret_cast<float *>(malloc(weight_quant_num_ * sizeof(float)));
  MS_CHECK_PTR(quant_.filter_scale_);
  quant_.filter_zp_ = reinterpret_cast<int32_t *>(malloc(weight_quant_num_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.filter_zp_);
  quant_.left_shift_ = reinterpret_cast<int32_t *>(malloc(weight_quant_num_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.left_shift_);
  quant_.right_shift_ = reinterpret_cast<int32_t *>(malloc(weight_quant_num_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.right_shift_);
  quant_.quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(weight_quant_num_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.quant_multiplier_);
  return RET_OK;
}

int MatMulBaseInt8Coder::InitQuantParam() {
  std::vector<LiteQuantParam> in_quant_params = input_tensor_->quant_params();
  MS_CHECK_TRUE(!in_quant_params.empty(), "in_quant_params is empty");
  quant_.input_.zp_ = in_quant_params.front().zeroPoint;
  quant_.input_.scale_ = static_cast<float>(in_quant_params.front().scale);

  std::vector<LiteQuantParam> out_quant_params = output_tensor_->quant_params();
  MS_CHECK_TRUE(!out_quant_params.empty(), "out_quant_params is empty");
  quant_.output_.zp_ = out_quant_params.front().zeroPoint;
  quant_.output_.scale_ = static_cast<float>(out_quant_params.front().scale);
  std::vector<LiteQuantParam> weight_quant_params = filter_tensor_->quant_params();
  for (int i = 0; i < weight_quant_num_; i++) {
    quant_.filter_zp_[i] = weight_quant_params[i].zeroPoint;
    quant_.filter_scale_[i] = static_cast<float>(weight_quant_params[i].scale);
  }

  for (int i = 0; i < weight_quant_num_; ++i) {
    const auto in_scale = static_cast<double>(quant_.input_.scale_ * quant_.filter_scale_[i]);
    double real_multiplier = in_scale / static_cast<double>(quant_.output_.scale_);
    QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_.quant_multiplier_[i], &quant_.left_shift_[i],
                                              &quant_.right_shift_[i]);
  }

  CalculateActivationRangeQuantized(param_->act_type_ == ActType_Relu, param_->act_type_ == ActType_Relu6,
                                    quant_.output_.zp_, quant_.output_.scale_, &quant_.out_act_min_,
                                    &quant_.out_act_max_);
  return RET_OK;
}

void MatMulBaseInt8Coder::InitParameter() {
  param_->a_const_ = (input_tensor_ != nullptr);
  param_->b_const_ = (filter_tensor_ != nullptr);
  row_tile_ = C4NUM;
  if (target_ == kARM32) {
    col_tile_ = C2NUM;
  } else {
    col_tile_ = C4NUM;
  }
}

int MatMulBaseInt8Coder::InitBias() {
  if (bias_tensor_ != nullptr) {
    int max_bias_data_elements = UP_ROUND(bias_tensor_->ElementsNum(), C4NUM);
    // to pack to init
    bias_ptr_size_ = static_cast<size_t>(max_bias_data_elements * sizeof(int));
  }
  return RET_OK;
}

int MatMulBaseInt8Coder::Init() {
  if (MallocQuantParam() != RET_OK) {
    FreeQuantParam();
    return RET_ERROR;
  }
  MS_CHECK_RET_CODE(InitQuantParam(), "matmul int8 init quant_param failed");
  if (InitBias() != RET_OK) {
    FreeQuantParam();
    return RET_ERROR;
  }
  return RET_OK;
}

int MatMulBaseInt8Coder::Prepare(CoderContext *const context) { return RET_OK; }

void MatMulBaseInt8Coder::DoBatchCode(NNaclInt8Serializer *code_ptr) {
  NNaclInt8Serializer &code = *code_ptr;
  std::string value_str_end = ";\n";
  std::string a_ptr_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string c_ptr_str = allocator_->GetRuntimeAddr(output_tensor_);
  std::string pack_b_ptr_str = allocator_->GetRuntimeAddr(pack_b_ptr_);
  std::string weight_bias_sums_str = allocator_->GetRuntimeAddr(weight_bias_sums_);
  code.precision(kPrecision);
  std::string tmp_weight_zp =
    "int32_t tmp_weight_zp = " + (filter_per_channel_ ? std::to_string(1) : std::to_string(quant_.filter_zp_[0]));
  code << tmp_weight_zp << value_str_end;
  for (int i = 0; i < param_->batch; i++) {
    code << "\n  {\n";
    std::string current_src_a = a_ptr_str + "+" + std::to_string(a_offset_[i] * param_->row_ * param_->deep_);
    if (param_->a_transpose_) {
      code.CodeFunction("RowMajor2Col16x4MajorInt8", current_src_a, pack_a_ptr_, param_->deep_, param_->row_);
      code.CodeFunction("CalcInputSums", current_src_a, param_->row_, param_->deep_, "tmp_weight_zp", input_sums_,
                        ColMajor);
    } else {
      code.CodeFunction("RowMajor2Row16x4MajorInt8", current_src_a, pack_a_ptr_, param_->row_, param_->deep_);
      code.CodeFunction("CalcInputSums", current_src_a, param_->row_, param_->deep_, "tmp_weight_zp", input_sums_,
                        RowMajor);
    }
    std::string batch_b_ptr_str =
      pack_b_ptr_str + "+" + std::to_string(b_offset_[i] * param_->col_align_ * param_->deep_16_);
    std::string batch_c_ptr_str = c_ptr_str + "+" + std::to_string(i * param_->row_ * param_->col_);
    std::string batch_weight_bias_sums_str =
      weight_bias_sums_str + "+" + std::to_string(b_offset_[i] * param_->col_align_);
    int stride = thread_stride_ * col_tile_;
    int cur_stride = kDefaultTaskId * stride;
    int res_stride = param_->col_ - cur_stride;
    int cur_oc = MSMIN(stride, res_stride);
    if (cur_oc <= 0) {
      code << "\n  }\n";
      return;
    }
    code.CodeStruct("matmul_quant_parameter", quant_, weight_quant_num_);
    std::string cur_left = "int32_t *cur_left = matmul_quant_parameter.left_shift_";
    std::string cur_right = "int32_t *cur_right = matmul_quant_parameter.right_shift_";
    std::string cur_mul = "int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ ";
    std::string cur_zp = "int32_t *cur_zp = matmul_quant_parameter.filter_zp_ ";
    if (filter_per_channel_) {
      code << cur_left << " + " << cur_stride << value_str_end;
      code << cur_right << " + " << cur_stride << value_str_end;
      code << cur_mul << " + " << cur_stride << value_str_end;
      code << cur_zp << " + " << cur_stride << value_str_end;
    } else {
      code << cur_left << value_str_end;
      code << cur_right << value_str_end;
      code << cur_mul << value_str_end;
      code << cur_zp << value_str_end;
    }
    std::string batch_b_ptr_str_final = batch_b_ptr_str + " + " + std::to_string(cur_stride * param_->deep_16_);
    std::string batch_c_ptr_final = batch_c_ptr_str + "+" + std::to_string(cur_stride);
    std::string weight_bias_sums_str_final = batch_weight_bias_sums_str + "+" + std::to_string(cur_stride);
    code.CodeFunction("MatmulInt8Opt", pack_a_ptr_, batch_b_ptr_str_final, batch_c_ptr_final, param_->row_, cur_oc,
                      param_->deep_16_, input_sums_, weight_bias_sums_str_final, quant_.out_act_min_,
                      quant_.out_act_max_, quant_.output_.zp_, "cur_mul", "cur_left", "cur_right", param_->col_,
                      filter_per_channel_, "cur_zp");
    code << "\n  }\n";
  }
}

int MatMulBaseInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/common_func.h",
            "nnacl/int8/common_func_int8.h",
            "nnacl/int8/matmul_int8.h",
            "nnacl/int8/fixed_point.h",
            "nnacl/int8/relux_int8.h",
            "wrapper/int8/matmul_int8_wrapper.h",
          },
          {
            "common_func.c",
            "common_func_int8.c",
            "matmul_int8.c",
            "fixed_point.c",
            "relux_int8.c",
            "matmul_int8_wrapper.c",
          });
  NNaclInt8Serializer init_code, code;
  size_t w_buf_size = 0;
  auto filter_tensor_name = MemoryAllocator::GetInstance()->GetRuntimeAddr(filter_tensor_);
  std::string bias_ptr_str = MemoryAllocator::GetInstance()->GetRuntimeAddr(bias_tensor_);
  bias_ptr_str = (bias_ptr_str == "") ? "NULL" : bias_ptr_str;
  if (target_ != kCortex_M) {
    if (param_->b_const_) {
      init_code.CodeBufferOffsetExpression(weight_bias_sums_, context->weight_name(), context->weight_offset_name(),
                                           context->weight_size_name(), weight_bias_sums_size_);
      init_code.CodeBufferOffsetExpression(pack_b_ptr_, context->weight_name(), context->weight_offset_name(),
                                           context->weight_size_name(), b_pack_ptr_size_);
      w_buf_size += weight_bias_sums_size_ + b_pack_ptr_size_;
      init_code.CodeArray("init_filter_zp", quant_.filter_zp_, weight_quant_num_, false);
      init_code.CodeFunction("InitInt8MatrixB", filter_tensor_name, weight_bias_sums_, pack_b_ptr_, param_->batch,
                             param_->deep_, param_->col_, param_->col_align_, param_->deep_16_, quant_.input_.zp_,
                             "init_filter_zp", bias_ptr_str, param_->b_transpose_, filter_per_channel_);
    } else {
      code.CodeArray("init_filter_zp", quant_.filter_zp_, weight_quant_num_, false);
      code.CodeFunction("InitInt8MatrixB", filter_tensor_name, weight_bias_sums_, pack_b_ptr_, param_->batch,
                        param_->deep_, param_->col_, param_->col_align_, param_->deep_16_, quant_.input_.zp_,
                        "init_filter_zp", bias_ptr_str, param_->b_transpose_, filter_per_channel_);
    }
  }
  DoBatchCode(&code);

  MS_LOG(DEBUG) << "FullConnectionInt8Coder has been called";
  context->AppendInitCode(init_code.str());
  context->AppendCode(code.str());
  context->AppendInitWeightSizeCode(w_buf_size);
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl

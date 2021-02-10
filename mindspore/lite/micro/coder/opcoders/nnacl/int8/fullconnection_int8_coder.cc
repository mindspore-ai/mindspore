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

#include "coder/opcoders/nnacl/int8/fullconnection_int8_coder.h"
#include "nnacl/int8/matmul_int8.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::lite::micro::nnacl {

FullConnectionInt8Coder ::~FullConnectionInt8Coder() { FreeQuantParam(); }

int FullConnectionInt8Coder::MallocQuantParam() {
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  std::vector<QuantArg> weight_quant_params = filter_tensor_->quant_params();
  MS_CHECK_TRUE(!filter_tensor_->shape().empty(), "filter tensor shape is empty");
  int col = filter_tensor_->shape().front();
  filter_per_channel_ = (weight_quant_params.size() > 1);
  init_size_ = filter_per_channel_ ? col : 1;
  quant_.filter_scale_ = reinterpret_cast<float *>(malloc(init_size_ * sizeof(float)));
  MS_CHECK_PTR(quant_.filter_scale_);
  quant_.filter_zp_ = reinterpret_cast<int32_t *>(malloc(init_size_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.filter_zp_);
  quant_.left_shift_ = reinterpret_cast<int32_t *>(malloc(init_size_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.left_shift_);
  quant_.right_shift_ = reinterpret_cast<int32_t *>(malloc(init_size_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.right_shift_);
  quant_.quant_multiplier_ = reinterpret_cast<int32_t *>(malloc(init_size_ * sizeof(int32_t)));
  MS_CHECK_PTR(quant_.quant_multiplier_);
  return RET_OK;
}

void FullConnectionInt8Coder::FreeQuantParam() {
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

void FullConnectionInt8Coder::InitParam() {
  int row = 1;
  int out_put_tensor_size = static_cast<int>(output_tensor_->shape().size());
  for (int i = 0; i < out_put_tensor_size - 1; ++i) {
    row *= (output_tensor_->shape()).at(i);
  }
  fc_param_->row_ = row;
  fc_param_->col_ = output_tensor_->shape().back();
  fc_param_->deep_ = filter_tensor_->shape().at(1);
  fc_param_->row_4_ = UP_ROUND(fc_param_->row_, C4NUM);
  fc_param_->col_4_ = UP_ROUND(fc_param_->col_, C4NUM);
  fc_param_->col_8_ = UP_ROUND(fc_param_->col_, C8NUM);
  fc_param_->deep_16_ = UP_ROUND(fc_param_->deep_, C16NUM);
  thread_count_ = MSMIN(thread_num_, UP_DIV(fc_param_->col_4_, C4NUM));
  thread_stride_ = UP_DIV(UP_DIV(fc_param_->col_4_, C4NUM), thread_count_);
}

int FullConnectionInt8Coder::ReSize(CoderContext *const context) {
  InitParam();
  pack_a_ptr_size_ = static_cast<size_t>(fc_param_->row_4_ * fc_param_->deep_16_ * sizeof(int8_t));
  pack_a_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, pack_a_ptr_size_, kOfflinePackWeight));
  MS_CHECK_PTR(pack_a_ptr_);
  pack_b_ptr_size_ = static_cast<size_t>(fc_param_->col_4_ * fc_param_->deep_16_ * sizeof(int8_t));
  weight_bias_sums_size_ = static_cast<size_t>(fc_param_->col_4_ * sizeof(int));
  if (fc_param_->b_const_) {
    pack_b_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(pack_b_ptr_);
    weight_bias_sums_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, kOnlineSize, kOnlinePackWeight));
  } else {
    pack_b_ptr_ = reinterpret_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, pack_b_ptr_size_, kOfflinePackWeight));
    MS_CHECK_PTR(pack_b_ptr_);
    weight_bias_sums_ =
      reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, weight_bias_sums_size_, kOfflinePackWeight));
  }
  MS_CHECK_PTR(weight_bias_sums_);
  input_sums_size_ = static_cast<size_t>(fc_param_->row_4_ * sizeof(int));
  input_sums_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, input_sums_size_, kOfflinePackWeight));
  MS_CHECK_PTR(input_sums_);
  if (input_tensors_.size() == kInputSize2) {
    bias_ptr_size_ = static_cast<size_t>(fc_param_->col_4_ * sizeof(int));
    bias_ptr_ = reinterpret_cast<int *>(allocator_->Malloc(kNumberTypeInt, kOnlineSize, kOnlinePackWeight));
    MS_CHECK_PTR(bias_ptr_);
  } else {
    bias_ptr_ = nullptr;
  }
  NNaclInt8Serializer init_code;
  if (input_tensors_.size() == kInputSize2) {
    init_code.CodeMallocExpression(bias_ptr_, bias_ptr_size_);
    init_code.CodeFunction("memset", bias_ptr_, 0, bias_ptr_size_);
    init_code.CodeFunction("memcpy", bias_ptr_, bias_tensor_, bias_ptr_);
  }
  if (fc_param_->b_const_) {
    init_code.CodeMallocExpression(pack_b_ptr_, pack_b_ptr_size_);
    init_code.CodeMallocExpression(weight_bias_sums_, weight_bias_sums_size_);
    init_code.CodeFunction("RowMajor2Row16x4MajorInt8", filter_tensor_, pack_b_ptr_, fc_param_->col_, fc_param_->deep_);
    init_code.CodeFunction("CalcWeightBiasSums", filter_tensor_, fc_param_->deep_, fc_param_->col_, quant_.input_.zp_,
                           quant_.filter_zp_, bias_ptr_, weight_bias_sums_, ColMajor, filter_per_channel_);
  }
  context->AppendInitCode(init_code.str());
  return RET_OK;
}

int FullConnectionInt8Coder::Init() {
  fc_param_ = reinterpret_cast<MatMulParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data_c());
  }
  fc_param_->a_const_ = (input_tensor_->data_c() != nullptr);
  fc_param_->b_const_ = (filter_tensor_->data_c() != nullptr);
  int ret = MallocQuantParam();
  if (ret != RET_OK) {
    FreeQuantParam();
    return ret;
  }
  std::vector<QuantArg> in_quant_params = input_tensor_->quant_params();
  MS_CHECK_TRUE(!in_quant_params.empty(), "in_quant_params empty is empty");
  quant_.input_.zp_ = in_quant_params.front().zeroPoint;
  quant_.input_.scale_ = static_cast<float>(in_quant_params.front().scale);
  std::vector<QuantArg> out_quant_params = output_tensor_->quant_params();
  MS_CHECK_TRUE(!out_quant_params.empty(), "out_quant_params empty is empty");
  quant_.output_.zp_ = out_quant_params.front().zeroPoint;
  quant_.output_.scale_ = static_cast<float>(out_quant_params.front().scale);

  int weight_quant_num = filter_per_channel_ ? static_cast<int>(filter_tensor_->shape().front()) : 1;
  std::vector<QuantArg> weight_quant_params = filter_tensor_->quant_params();
  MS_CHECK_TRUE(!weight_quant_params.empty(), "weight_quant_params empty is empty");
  for (int i = 0; i < weight_quant_num; i++) {
    quant_.filter_zp_[i] = weight_quant_params[i].zeroPoint;
    quant_.filter_scale_[i] = static_cast<float>(weight_quant_params[i].scale);
  }

  for (int i = 0; i < weight_quant_num; ++i) {
    auto in_scale = static_cast<double>(quant_.input_.scale_ * quant_.filter_scale_[i]);
    double real_multiplier = in_scale / static_cast<double>(quant_.output_.scale_);
    QuantizeRoundParameterWithDoublePrecision(real_multiplier, &quant_.quant_multiplier_[i], &quant_.left_shift_[i],
                                              &quant_.right_shift_[i]);
  }
  CalculateActivationRangeQuantized(fc_param_->act_type_ == ActType_Relu, fc_param_->act_type_ == ActType_Relu6,
                                    quant_.output_.zp_, quant_.output_.scale_, &quant_.out_act_min_,
                                    &quant_.out_act_max_);
  return RET_OK;
}

int FullConnectionInt8Coder::Prepare(CoderContext *const context) {
  // only support one thread currently
  thread_count_ = thread_num_;
  MS_CHECK_RET_CODE(Init(), "FullConnectionInt8Coder init failed");
  return ReSize(context);
}

int FullConnectionInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/common_func.h", "nnacl/int8/common_func_int8.h", "nnacl/int8/matmul_int8.h"},
          {"common_func.c", "common_func_int8.c", "matmul_int8.c"});

  NNaclInt8Serializer code;
  code.precision(kPrecision);
  code.CodeFunction("memset", input_sums_, 0, input_sums_size_);
  code.CodeFunction("memset", pack_a_ptr_, 0, pack_a_ptr_size_);
  code.CodeFunction("RowMajor2Row16x4MajorInt8", input_tensor_, pack_a_ptr_, fc_param_->row_, fc_param_->deep_);
  int32_t tmp_weight_zp = filter_per_channel_ ? 1 : quant_.filter_zp_[0];
  code.CodeFunction("CalcInputSums", input_tensor_, fc_param_->row_, fc_param_->deep_, tmp_weight_zp, input_sums_,
                    RowMajor);

  if (!fc_param_->b_const_) {
    code.CodeFunction("memset", pack_b_ptr_, 0, pack_b_ptr_size_);
    code.CodeFunction("memset", weight_bias_sums_, 0, weight_bias_sums_size_);
    code.CodeFunction("RowMajor2Row16x4MajorInt8", filter_tensor_, pack_b_ptr_, fc_param_->col_, fc_param_->deep_);
    code.CodeFunction("CalcWeightBiasSums", filter_tensor_, fc_param_->deep_, fc_param_->col_, quant_.input_.zp_,
                      quant_.filter_zp_, bias_ptr_, weight_bias_sums_, ColMajor, filter_per_channel_);
  }
  int stride = thread_stride_ * C4NUM;
  int res_stride = fc_param_->col_;
  int cur_oc = MSMIN(stride, res_stride);
  if (cur_oc <= 0) {
    return RET_OK;
  }
  int32_t *cur_left = quant_.left_shift_;
  int32_t *cur_right = quant_.right_shift_;
  int32_t *cur_mul = quant_.quant_multiplier_;
  int32_t *cur_zp = quant_.filter_zp_;

  code.CodeArray("cur_left_shift", cur_left, init_size_, true);
  code.CodeArray("cur_right_shift", cur_right, init_size_, true);
  code.CodeArray("cur_multiplier", cur_mul, init_size_, true);
  code.CodeArray("cur_filter_zp", cur_zp, init_size_, true);

  code.CodeFunction("MatmulInt8Opt", pack_a_ptr_, pack_b_ptr_, output_tensor_->data_c(), fc_param_->row_, cur_oc,
                    fc_param_->deep_16_, input_sums_, weight_bias_sums_, quant_.out_act_min_, quant_.out_act_max_,
                    quant_.output_.zp_, "&cur_multiplier", "&cur_left_shift", "&cur_right_shift", fc_param_->col_,
                    filter_per_channel_, "&cur_filter_zp");
  MS_LOG(DEBUG) << "FullConnectionInt8Coder has been called";
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_FullConnection,
                   CPUOpCoderCreator<FullConnectionInt8Coder>)

}  // namespace mindspore::lite::micro::nnacl

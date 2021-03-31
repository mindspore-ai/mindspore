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
#include "coder/opcoders/nnacl/int8/reduce_int8_coder.h"
#include <vector>
#include <string>
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

using mindspore::schema::PrimitiveType_ReduceFusion;
namespace mindspore::lite::micro::nnacl {
int ReduceInt8Coder::CalculateQuantArgs() {
  QuantArg input_quant = input_tensor_->quant_params().at(0);
  QuantArg output_quant = output_tensor_->quant_params().at(0);
  quant_arg_.in_scale_ = input_quant.scale;
  quant_arg_.in_zp_ = input_quant.zeroPoint;
  quant_arg_.out_scale_ = output_quant.scale;
  quant_arg_.out_zp_ = output_quant.zeroPoint;
  const double input_output_multiplier = quant_arg_.in_scale_ / quant_arg_.out_scale_;
  int shift;
  QuantizeMultiplierSmallerThanOne(input_output_multiplier, &quant_arg_.in_out_multiplier_, &shift);
  quant_arg_.in_out_left_shift_ = shift < 0 ? -shift : 0;
  quant_arg_.in_out_right_shift_ = shift > 0 ? shift : 0;
  MS_CHECK_TRUE(num_axes_ < MAX_SHAPE_SIZE, "the number of axes should be less the max num");
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    for (int i = 0; i < num_axes_; ++i) {
      auto axis = axes_[i];
      std::vector<int> in_shape = input_tensor_->shape();
      if (static_cast<int>(in_shape.size()) - 1 < axis) {
        MS_LOG(ERROR) << "input tensor shape is invalid";
        return RET_ERROR;
      }
      double reciprocal = 1.0 / in_shape.at(axis);
      auto *qm = new (std::nothrow) QuantMulArg;
      MS_CHECK_PTR(qm);
      QuantizeMultiplierSmallerThanOne(reciprocal, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      mean_multipliers_.push_back(qm);
    }
  }

  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceProd)) {
    for (int i = 0; i < num_axes_; ++i) {
      int axis = axes_[i];
      std::vector<int> in_shape = input_tensors_.at(kInputIndex)->shape();
      if (static_cast<int>(in_shape.size()) - 1 < axis) {
        MS_LOG(ERROR) << "input tensor shape is invalid";
        return RET_ERROR;
      }
      int axis_size = in_shape.at(axis);
      double prod_multiplier = std::pow(quant_arg_.in_scale_, axis_size - 1);
      auto *qm = new (std::nothrow) QuantMulArg;
      MS_CHECK_PTR(qm);
      QuantizeMultiplierSmallerThanOne(prod_multiplier, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      mean_multipliers_.push_back(qm);
    }
  }

  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSumSquare)) {
    for (int i = 0; i < num_axes_ - 1; ++i) {
      auto *qm = new (std::nothrow) QuantMulArg;
      MS_CHECK_PTR(qm);
      double sum_square_multiplier = quant_arg_.in_scale_;
      QuantizeMultiplierSmallerThanOne(sum_square_multiplier, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      sum_square_multipliers_.push_back(qm);
    }
    // for last num_axes
    auto *qm = new (std::nothrow) QuantMulArg;
    MS_CHECK_PTR(qm);
    double sum_square_multiplier = quant_arg_.in_scale_ * (quant_arg_.in_scale_ / quant_arg_.out_scale_);
    QuantizeMultiplierSmallerThanOne(sum_square_multiplier, &qm->multiplier_, &shift);
    qm->left_shift_ = shift < 0 ? -shift : 0;
    qm->right_shift_ = shift > 0 ? shift : 0;
    sum_square_multipliers_.push_back(qm);
  }

  return RET_OK;
}

int ReduceInt8Coder::MallocTmpBuffer() {
  data_buffers_.clear();
  if (num_axes_ != static_cast<int>(buffer_sizes_.size())) {
    MS_LOG(ERROR) << "num_axes_ size is invalid";
    return RET_ERROR;
  }
  for (auto buffer_size : buffer_sizes_) {
    auto *buffer =
      static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, buffer_size * sizeof(int32_t), kWorkspace));
    MS_CHECK_PTR(buffer);
    data_buffers_.emplace_back(buffer);
  }
  return RET_OK;
}

void ReduceInt8Coder::GetQuantArgs(size_t index) {
  if (index > static_cast<size_t>(num_axes_)) {
    MS_LOG(ERROR) << "index is invalid, beyond num_axes_";
    return;
  }
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    quant_arg_.mean_multiplier_ = mean_multipliers_.at(index)->multiplier_;
    quant_arg_.mean_left_shift_ = mean_multipliers_.at(index)->left_shift_;
    quant_arg_.mean_right_shift_ = mean_multipliers_.at(index)->right_shift_;
  }
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceProd)) {
    quant_arg_.prod_multiplier_ = prod_multipliers_.at(index)->multiplier_;
    quant_arg_.prod_left_shift_ = prod_multipliers_.at(index)->left_shift_;
    quant_arg_.prod_right_shift_ = prod_multipliers_.at(index)->right_shift_;
  }

  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSumSquare)) {
    quant_arg_.sum_square_multiplier_ = sum_square_multipliers_.at(index)->multiplier_;
    quant_arg_.sum_square_left_shift_ = sum_square_multipliers_.at(index)->left_shift_;
    quant_arg_.sum_square_right_shift_ = sum_square_multipliers_.at(index)->right_shift_;
  }
}

int ReduceInt8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(ReduceBaseCoder::Init(), "Init failed");
  std::vector<int> in_shape = input_tensor_->shape();
  if (!in_shape.empty()) {
    this->valid_shape_ = true;
    MS_CHECK_RET_CODE(CalculateQuantArgs(), "CalculateQuantArgs failed");
  } else {
    this->valid_shape_ = false;
  }
  switch (mode_) {
    case static_cast<int>(schema::ReduceMode_ReduceMean): {
      reducer_ = "ReduceMeanInt8";
      last_reducer_ = "ReduceMeanLastAxis";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceSum): {
      reducer_ = "ReduceSumInt8";
      last_reducer_ = "ReduceSumLastAxis";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMax): {
      reducer_ = "ReduceMaxInt8";
      last_reducer_ = "ReduceMaxLastAxis";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceMin): {
      reducer_ = "ReduceMinInt8";
      last_reducer_ = "ReduceMinLastAxis";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceProd): {
      reducer_ = "ReduceProdInt8";
      last_reducer_ = "ReduceProdLastAxis";
      break;
    }
    case static_cast<int>(schema::ReduceMode_ReduceSumSquare): {
      reducer_ = "ReduceSumSquareInt8";
      last_reducer_ = "ReduceSumSquareLastAxis";
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce mode" << mode_;
      return RET_ERROR;
  }
  MS_CHECK_RET_CODE(ReduceBaseCoder::ReSize(), "ReSize failed");
  if (!this->valid_shape_) {
    MS_CHECK_RET_CODE(CalculateQuantArgs(), "CalculateQuantArgs failed");
  }
  MS_CHECK_RET_CODE(MallocTmpBuffer(), "MallocTmpBuffer failed");
  begin_src_data_ = static_cast<int32_t *>(
    allocator_->Malloc(kNumberTypeInt32, sizeof(int32_t) * input_tensor_->ElementsNum(), kWorkspace));
  MS_CHECK_PTR(begin_src_data_);
  return RET_OK;
}

int ReduceInt8Coder::DoCode(CoderContext *const context) {
  MS_LOG(DEBUG) << "*****Reduce code start*****";
  NNaclInt8Serializer code;
  Collect(context, {"nnacl/int8/reduce_int8.h"}, {"reduce_int8.c", "fixed_point.c"});
  std::string src_addr = allocator_->GetRuntimeAddr(input_tensor_);
  std::string dst_addr;
  std::string begin_src_data_src = allocator_->GetRuntimeAddr(begin_src_data_);

  code << "int *begin_data = (int *)(" << begin_src_data_src << ");\n";
  code << "int8_t *ori_data = (int8_t *)(" << src_addr << ");\n";
  code << "for (int i = 0; i < " << input_tensor_->ElementsNum() << "; ++i) {\n"
       << "    begin_data[i] = (int)ori_data[i];\n"
       << "  }\n";
  for (int i = 0; i < num_axes_; ++i) {
    GetQuantArgs(i);
    std::string quant_arg_i = "quant_arg_" + std::to_string(i);
    std::string ptr_quan_arg_i = "&" + quant_arg_i;
    code.CodeStruct(quant_arg_i, quant_arg_);
    if (i != num_axes_ - 1) {
      is_last_axis = false;
      dst_addr = allocator_->GetRuntimeAddr(data_buffers_.at(i));
    } else {
      is_last_axis = true;
      dst_addr = allocator_->GetRuntimeAddr(output_tensor_);
    }
    outer_size_ = outer_sizes_.at(i);
    inner_size_ = inner_sizes_.at(i);
    axis_size_ = axis_sizes_.at(i);
    if (!is_last_axis) {
      code.CodeFunction(reducer_, outer_size_, inner_size_, axis_size_, begin_src_data_src, dst_addr, ptr_quan_arg_i,
                        kDefaultTaskId, thread_num_);
    } else {
      code.CodeFunction(last_reducer_, outer_size_, inner_size_, axis_size_, begin_src_data_src, dst_addr,
                        ptr_quan_arg_i, kDefaultTaskId, thread_num_);
    }
    begin_src_data_src = dst_addr;
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_ReduceFusion, CPUOpCoderCreator<ReduceInt8Coder>)

}  // namespace mindspore::lite::micro::nnacl

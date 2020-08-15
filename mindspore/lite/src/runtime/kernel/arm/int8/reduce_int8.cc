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

#include <algorithm>
#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "nnacl/quantization/quantize.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/int8/reduce_int8.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Reduce;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {
int ReduceInt8CPUKernel::Init() {
  auto ret = ReduceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    return ret;
  }
  ret = CalculateQuantArgs();
  if (ret != RET_OK) {
    return ret;
  }

  switch (mode_) {
    case static_cast<int>(ReduceMode_ReduceMean): {
      reducer_ = ReduceMeanInt8;
      last_reducer_ = ReduceMeanLastAxis;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceSum): {
      reducer_ = ReduceSumInt8;
      last_reducer_ = ReduceSumLastAxis;
      break;
    }

    case static_cast<int>(ReduceMode_ReduceMax): {
      reducer_ = ReduceMaxInt8;
      last_reducer_ = ReduceMaxLastAxis;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceMin): {
      reducer_ = ReduceMinInt8;
      last_reducer_ = ReduceMinLastAxis;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceProd): {
      reducer_ = ReduceProdInt8;
      last_reducer_ = ReduceProdLastAxis;
      break;
    }
    case static_cast<int>(ReduceMode_ReduceSumSquare): {
      // In multi-axes reduce cases, sum square output different output for different reduce order
      // e.g. axes [2, 3] is different from axes [3, 2].
      reducer_ = ReduceSumSquareInt8;
      last_reducer_ = ReduceSumSquareLastAxis;
      break;
    }
    default:
      MS_LOG(ERROR) << "Reduce unsupported reduce mode: " << mode_;
      return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ReduceInt8CPUKernel::CalculateQuantArgs() {
  lite::tensor::Tensor *input = in_tensors_.at(0);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_arg_.in_scale_ = input->GetQuantParams().front().scale;
  quant_arg_.in_zp_ = input->GetQuantParams().front().zeroPoint;
  quant_arg_.out_scale_ = output->GetQuantParams().front().scale;
  quant_arg_.out_zp_ = output->GetQuantParams().front().zeroPoint;

  // (quant_out - out_zp) * out_scale = (quant_in - in_zp) * in_scale
  const double input_output_multiplier = quant_arg_.in_scale_ / quant_arg_.out_scale_;
  int shift;
  QuantizeMultiplierSmallerThanOne(input_output_multiplier, &quant_arg_.in_out_multiplier_, &shift);
  quant_arg_.in_out_left_shift_ = shift < 0 ? -shift : 0;
  quant_arg_.in_out_right_shift_ = shift > 0 ? shift : 0;

  // (quant_out - zp_out)*scale_out = sum((quant_in -zp)*scale_in) * (1/num) for each axis in axes
  // quant_out = sum(quant_in-zp) * (scale_in/scale_out) * (1/num)
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    for (auto i = 0; i < num_axes_; i++) {
      auto axis = axes_[i];
      double reciprocal = 1.0 / in_tensors_.at(0)->shape()[axis];
      QuantMulArg *qm = new (std::nothrow) QuantMulArg;
      if (qm == nullptr) {
        MS_LOG(ERROR) << "Reduce new QuantMulArg failed.";
        return RET_NULL_PTR;
      }
      QuantizeMultiplierSmallerThanOne(reciprocal, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      mean_multipliers_.push_back(qm);
    }
  }

  // (quant_out - zp) * scale_out = prod(quant_in - zp) * scale_in^num
  // quant_out = prod(quant_in-zp) * (scale_in^num/scale_out) + zp_out
  // scale_in^num-1 * scale_in/scale_out
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceProd)) {
    for (auto i = 0; i < num_axes_; i++) {
      int axis_size = in_tensors_.at(0)->shape()[axes_[i]];
      QuantMulArg *qm = new (std::nothrow) QuantMulArg;
      if (qm == nullptr) {
        MS_LOG(ERROR) << "ReduceProd new QuantMulArg failed.";
        return RET_NULL_PTR;
      }
      double prod_multiplier = pow(quant_arg_.in_scale_, axis_size - 1);
      QuantizeMultiplierSmallerThanOne(prod_multiplier, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      prod_multipliers_.push_back(qm);
    }
  }

  // (quant_out - zp) * scale_out = sum((quant_in - zp)^2 * scale_in^2)
  // quant_out = sum((quant_in - zp)^2) * scale_in^2 / scale_out + zp_out
  // scale_in * scale_in/scale_out
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSumSquare)) {
    for (auto i = 0; i < num_axes_ - 1; i++) {
      QuantMulArg *qm = new (std::nothrow) QuantMulArg;
      if (qm == nullptr) {
        MS_LOG(ERROR) << "ReduceProd new QuantMultiplier failed.";
        return RET_NULL_PTR;
      }
      double sumsquare_multiplier = quant_arg_.in_scale_;
      QuantizeMultiplierSmallerThanOne(sumsquare_multiplier, &qm->multiplier_, &shift);
      qm->left_shift_ = shift < 0 ? -shift : 0;
      qm->right_shift_ = shift > 0 ? shift : 0;
      sum_square_multipliers_.push_back(qm);
    }

    QuantMulArg *qm = new (std::nothrow) QuantMulArg;
    if (qm == nullptr) {
      MS_LOG(ERROR) << "ReduceProd new QuantMultiplier failed.";
      return RET_NULL_PTR;
    }
    double sumsquare_multiplier = quant_arg_.in_scale_ * quant_arg_.in_scale_ / quant_arg_.out_scale_;
    QuantizeMultiplierSmallerThanOne(sumsquare_multiplier, &qm->multiplier_, &shift);
    qm->left_shift_ = shift < 0 ? -shift : 0;
    qm->right_shift_ = shift > 0 ? shift : 0;
    sum_square_multipliers_.push_back(qm);
  }
  return RET_OK;
}

int ReduceInt8CPUKernel::MallocTmpBuffer() {
  auto input_shape = in_tensors_.at(0)->shape();
  for (auto i = 0; i < num_axes_ - 1; i++) {
    int axis = axes_[i];
    size_t size = 1;
    for (auto j = 0; j < input_shape.size(); j++) {
      if (static_cast<size_t>(axis) != j) {
        size *= input_shape[j];
      }
    }
    int32_t *buffer = reinterpret_cast<int32_t *>(malloc(size * sizeof(int32_t)));
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed.";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
    input_shape[axis] = 1;
  }

  auto input = in_tensors_.at(0);
  begin_src_data_ = reinterpret_cast<int32_t *>(malloc(sizeof(int32_t) * input->ElementsNum()));
  if (begin_src_data_ == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_data = reinterpret_cast<int8_t *>(input->Data());
  for (auto i = 0; i < input->ElementsNum(); i++) {
    begin_src_data_[i] = static_cast<int32_t>(input_data[i]);
  }
  return RET_OK;
}

int ReduceInt8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto reduce = reinterpret_cast<ReduceInt8CPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceInt8CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  is_last_axis_ = false;
  tmp_shape_ = in_tensors_.at(0)->shape();
  src_data_ = begin_src_data_;

  for (int i = 0; i < data_buffers_.size(); ++i) {
    if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
      quant_arg_.mean_multiplier_ = mean_multipliers_[i]->multiplier_;
      quant_arg_.mean_left_shift_ = mean_multipliers_[i]->left_shift_;
      quant_arg_.mean_right_shift_ = mean_multipliers_[i]->right_shift_;
    }

    if (mode_ == static_cast<int>(schema::ReduceMode_ReduceProd)) {
      quant_arg_.prod_multiplier_ = prod_multipliers_[i]->multiplier_;
      quant_arg_.prod_left_shift_ = prod_multipliers_[i]->left_shift_;
      quant_arg_.prod_right_shift_ = prod_multipliers_[i]->right_shift_;
    }
    if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSumSquare)) {
      quant_arg_.sum_square_multiplier_ = sum_square_multipliers_[i]->multiplier_;
      quant_arg_.sum_square_left_shift_ = sum_square_multipliers_[i]->left_shift_;
      quant_arg_.sum_square_right_shift_ = sum_square_multipliers_[i]->right_shift_;
    }
    dst_data_ = data_buffers_[i];
    int axis = axes_[i];
    outer_size_ = 1;
    for (int j = 0; j < axis; j++) {
      outer_size_ *= tmp_shape_[j];
    }
    inner_size_ = 1;
    for (int k = axis + 1; k < static_cast<int>(tmp_shape_.size()); k++) {
      inner_size_ *= tmp_shape_[k];
    }
    axis_size_ = tmp_shape_[axis];
    auto error_code = LiteBackendParallelLaunch(ReduceInt8Impl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
    tmp_shape_[axis] = 1;
    src_data_ = dst_data_;
  }

  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    quant_arg_.mean_multiplier_ = mean_multipliers_.back()->multiplier_;
    quant_arg_.mean_left_shift_ = mean_multipliers_.back()->left_shift_;
    quant_arg_.mean_right_shift_ = mean_multipliers_.back()->right_shift_;
  }
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceProd)) {
    quant_arg_.prod_multiplier_ = prod_multipliers_.back()->multiplier_;
    quant_arg_.prod_left_shift_ = prod_multipliers_.back()->left_shift_;
    quant_arg_.prod_right_shift_ = prod_multipliers_.back()->right_shift_;
  }
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceSumSquare)) {
    quant_arg_.sum_square_multiplier_ = sum_square_multipliers_.back()->multiplier_;
    quant_arg_.sum_square_left_shift_ = sum_square_multipliers_.back()->left_shift_;
    quant_arg_.sum_square_right_shift_ = sum_square_multipliers_.back()->right_shift_;
  }
  int last_reduce_axis = axes_[num_axes_ - 1];
  outer_size_ = 1;
  for (int i = 0; i < last_reduce_axis; i++) {
    outer_size_ *= tmp_shape_[i];
  }
  inner_size_ = 1;
  for (int i = last_reduce_axis + 1; i < static_cast<int>(tmp_shape_.size()); i++) {
    inner_size_ *= tmp_shape_[i];
  }
  axis_size_ = tmp_shape_[last_reduce_axis];
  last_dst_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());
  is_last_axis_ = true;
  auto error_code = LiteBackendParallelLaunch(ReduceInt8Impl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }

  if (begin_src_data_ != nullptr) {
    free(begin_src_data_);
    begin_src_data_ = nullptr;
  }

  return RET_OK;
}

int ReduceInt8CPUKernel::CallReduceUnit(int task_id) {
  int ret;
  if (!is_last_axis_) {
    ret =
      reducer_(outer_size_, inner_size_, axis_size_, src_data_, dst_data_, &quant_arg_, task_id, context_->thread_num_);
  } else {
    ret = last_reducer_(outer_size_, inner_size_, axis_size_, src_data_, last_dst_data_, &quant_arg_, task_id,
                        context_->thread_num_);
  }
  return ret;
}
}  // namespace mindspore::kernel

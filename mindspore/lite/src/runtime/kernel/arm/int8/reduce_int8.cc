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

#include "src/runtime/kernel/arm/int8/reduce_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "nnacl/pack.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

using mindspore::kernel::KERNEL_ARCH::kCPU;

namespace mindspore::kernel {
void ReduceInt8CPUKernel::OneAxis() {
  auto axis_info = axes_[0];
  if (axis_info == 0) {
    pattern_ = kernel::N;
  } else if (axis_info == 1) {
    pattern_ = kernel::H;
  } else if (axis_info == 2) {
    pattern_ = kernel::W;
  } else {
    pattern_ = kernel::C;
  }
}

void ReduceInt8CPUKernel::TwoAxes() {
  auto axis_info1 = axes_[0];
  auto axis_info2 = axes_[1];
  auto axis_sum = axis_info1 + axis_info2;
  if (axis_sum == 1) {
    pattern_ = kernel::NH;
  } else if (axis_sum == 2) {
    pattern_ = kernel::NW;
  } else if (axis_sum == 3) {
    if (axis_info1 == 0) {
      pattern_ = kernel::NC;
    } else {
      pattern_ = kernel::HW;
    }
  } else if (axis_sum == 4) {
    pattern_ = kernel::HC;
  } else {
    MS_ASSERT(axis_sum == 5);
    pattern_ = kernel::WC;
  }
}

void ReduceInt8CPUKernel::ThreeAxes() {
  auto axis_info1 = axes_[0];
  auto axis_info2 = axes_[1];
  auto axis_info3 = axes_[2];
  auto axis_sum = axis_info1 + axis_info2 + axis_info3;
  if (axis_sum == 3) {
    pattern_ = kernel::NHW;
  } else if (axis_sum == 4) {
    pattern_ = kernel::NHC;
  } else if (axis_sum == 5) {
    pattern_ = kernel::NWC;
  } else {
    MS_ASSERT(axis_sum == 6);
    pattern_ = kernel::HWC;
  }
}

void ReduceInt8CPUKernel::Match4DReducePattern() {
  if (num_axes_ == 1) {
    OneAxis();
  } else if (num_axes_ == 2) {
    TwoAxes();
  } else if (num_axes_ == 3) {
    ThreeAxes();
  } else {
    MS_ASSERT(num_axes_ == 4);
    pattern_ = kernel::NHWC;
  }
}

int ReduceInt8CPUKernel::Init() {
  auto ret = ReduceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  Match4DReducePattern();
  if (!this->in_tensors_[0]->shape().empty()) {
    this->valid_shape_ = true;
    ret = CalculateQuantArgs();
    if (ret != RET_OK) {
      return ret;
    }
  } else {
    this->valid_shape_ = false;
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

void ReduceInt8CPUKernel::ReduceMean4DCalQuantParam() {
  int reduce_num = 1;
  auto in_shape = in_tensors_.front()->shape();
  switch (pattern_) {
    case N:
      reduce_num = in_shape[0];
      break;
    case H:
      reduce_num = in_shape[1];
      break;
    case W:
      reduce_num = in_shape[2];
      break;
    case C:
      reduce_num = in_shape[3];
      break;
    case NH:
      reduce_num = in_shape[0] * in_shape[1];
      break;
    case NW:
      reduce_num = in_shape[0] * in_shape[2];
      break;
    case NC:
      reduce_num = in_shape[0] * in_shape[3];
      break;
    case HW:
      reduce_num = in_shape[1] * in_shape[2];
      break;
    case HC:
      reduce_num = in_shape[1] * in_shape[3];
      break;
    case WC:
      reduce_num = in_shape[2] * in_shape[3];
      break;
    case NHW:
      reduce_num = in_shape[0] * in_shape[1] * in_shape[2];
      break;
    case NHC:
      reduce_num = in_shape[0] * in_shape[1] * in_shape[3];
      break;
    case NWC:
      reduce_num = in_shape[0] * in_shape[2] * in_shape[3];
      break;
    case HWC:
      reduce_num = in_shape[1] * in_shape[2] * in_shape[3];
      break;
    case NHWC:
      reduce_num = in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3];
      break;
  }
  bias_ = quant_arg_.out_zp_ - quant_arg_.in_zp_ * quant_arg_.in_scale_ / quant_arg_.out_scale_;
  int shift;
  double reciprocal = quant_arg_.in_scale_ / (quant_arg_.out_scale_ * reduce_num);
  QuantizeMultiplierSmallerThanOne(reciprocal, &reduce_mean_quant_param_.multiplier_, &shift);
  reduce_mean_quant_param_.left_shift_ = shift < 0 ? -shift : 0;
  reduce_mean_quant_param_.right_shift_ = shift > 0 ? shift : 0;
}

int ReduceInt8CPUKernel::CalculateQuantArgs() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_arg_.in_scale_ = input->quant_params().front().scale;
  quant_arg_.in_zp_ = input->quant_params().front().zeroPoint;
  quant_arg_.out_scale_ = output->quant_params().front().scale;
  quant_arg_.out_zp_ = output->quant_params().front().zeroPoint;

  // (quant_out - out_zp) * out_scale = (quant_in - in_zp) * in_scale
  const double input_output_multiplier = quant_arg_.in_scale_ / quant_arg_.out_scale_;
  int shift;
  QuantizeMultiplierSmallerThanOne(input_output_multiplier, &quant_arg_.in_out_multiplier_, &shift);
  quant_arg_.in_out_left_shift_ = shift < 0 ? -shift : 0;
  quant_arg_.in_out_right_shift_ = shift > 0 ? shift : 0;

  // (quant_out - zp_out)*scale_out = sum((quant_in -zp)*scale_in) * (1/num) for each axis in axes
  // quant_out = sum(quant_in-zp) * (scale_in/scale_out) * (1/num)
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean)) {
    if (input->shape().size() == 4 && pattern_ == kernel::HW) {
      // special case, can use pattern
      ReduceMean4DCalQuantParam();
      pattern_impl_ = true;
    } else {
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
    return CalculateQuantArgsReduceSumSquare();
  }
  return RET_OK;
}

int ReduceInt8CPUKernel::CalculateQuantArgsReduceSumSquare() {
  int shift;
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
  return RET_OK;
}

int ReduceInt8CPUKernel::MallocTmpBuffer() {
  data_buffers_.clear();
  MS_ASSERT(static_cast<int>(buffer_sizes_.size()) == num_axes_ - 1);
  // malloc num_axes_-1 buffers, since reduce on last axis will generate result to out_tensor, no need for buffer.
  for (auto buffer_size : buffer_sizes_) {
    int32_t *buffer = reinterpret_cast<int32_t *>(context_->allocator->Malloc(buffer_size * sizeof(int32_t)));
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed.";
      return RET_ERROR;
    }
    data_buffers_.emplace_back(buffer);
  }

  auto input = in_tensors_.at(0);
  begin_src_data_ = reinterpret_cast<int32_t *>(context_->allocator->Malloc(sizeof(int32_t) * input->ElementsNum()));
  if (begin_src_data_ == nullptr) {
    return RET_NULL_PTR;
  }

  return RET_OK;
}

void ReduceInt8CPUKernel::FreeTmpBuffer() {
  for (auto buffer : data_buffers_) {
    if (buffer != nullptr) {
      context_->allocator->Free(buffer);
      buffer = nullptr;
    }
  }
  data_buffers_.clear();

  if (begin_src_data_ != nullptr) {
    context_->allocator->Free(begin_src_data_);
    begin_src_data_ = nullptr;
  }
}

int ReduceInt8CPUKernel::ReSize() { return ReduceBaseCPUKernel::ReSize(); }

int ReduceInt8Impl(void *cdata, int task_id) {
  auto reduce = reinterpret_cast<ReduceInt8CPUKernel *>(cdata);
  auto error_code = reduce->CallReduceUnit(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceMeanPatternInt8Impl(void *cdata, int task_id) {
  auto reduce = reinterpret_cast<ReduceInt8CPUKernel *>(cdata);
  auto error_code = reduce->Reduce4DExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void ReduceInt8CPUKernel::GetQuantArgs(size_t i) {
  MS_ASSERT(i < static_cast<size_t>(num_axes_));
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
}

int ReduceInt8CPUKernel::Reduce4DExecute(int task_id) {
  auto input = in_tensors_.at(0);
  auto in_data = reinterpret_cast<int8_t *>(input->data_c());
  auto in_shape = input->shape();
  MS_ASSERT(in_shape.size() == 4);
  int n = in_shape.at(0);
  int h = in_shape.at(1);
  int w = in_shape.at(2);
  int c = in_shape.at(3);
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data_c());
  switch (pattern_) {
    case N:
      return ReduceMeanN(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case H:
      return ReduceMeanH(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case W:
      return ReduceMeanW(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case C:
      return ReduceMeanC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NH:
      return ReduceMeanNH(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NW:
      return ReduceMeanNW(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NC:
      return ReduceMeanNC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case HW: {
      // data has been convert into NCHW format for efficiently
      int num = UP_DIV(c, ctx_->thread_num_);
      int count = c - task_id * num;
      count = count > num ? num : count;
      int plane = h * w;
      return ReduceMeanHW(n, plane, count, c, nchw_in_data_ + task_id * num * plane, output_data + task_id * num,
                          reduce_mean_quant_param_, bias_);
    }
    case HC:
      return ReduceMeanHC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case WC:
      return ReduceMeanWC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NHW:
      return ReduceMeanNHW(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NHC:
      return ReduceMeanNHC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NWC:
      return ReduceMeanNWC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case HWC:
      return ReduceMeanHWC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
    case NHWC:
      return ReduceMeanNHWC(n, h, w, c, in_data, output_data, reduce_mean_quant_param_);
  }
  return RET_OK;
}

int ReduceInt8CPUKernel::Fast4DReduceMeanHWImpl() {
  auto input = in_tensors_.at(0);
  auto input_data = reinterpret_cast<int8_t *>(input->data_c());
  nchw_in_data_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(input->ElementsNum()));
  if (nchw_in_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc nchw_in_data_ failed.";
    return RET_ERROR;
  }
  PackNHWCToNCHWInt8(reinterpret_cast<void *>(input_data), reinterpret_cast<void *>(nchw_in_data_), input->Batch(),
                     input->Height() * input->Width(), input->Channel());
  auto ret = ParallelLaunch(this->context_->thread_pool_, ReduceMeanPatternInt8Impl, this, context_->thread_num_);
  if (ret != RET_OK) {
    ctx_->allocator->Free(nchw_in_data_);
    MS_LOG(ERROR) << "Reduce run error, error_code[" << ret << "]";
    return RET_ERROR;
  }
  ctx_->allocator->Free(nchw_in_data_);
  return RET_OK;
}

int ReduceInt8CPUKernel::Run() {
  if (!this->valid_shape_) {
    auto ret = CalculateQuantArgs();
    if (ret != RET_OK) {
      return ret;
    }
  }
  // now only implement reduce mean mode 4d reduce HW case, otherwise go into reference impl
  if (mode_ == static_cast<int>(schema::ReduceMode_ReduceMean) && pattern_impl_ && pattern_ == kernel::HW) {
    return Fast4DReduceMeanHWImpl();
  }

  auto ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  is_last_axis_ = false;

  auto input = in_tensors().at(0);
  auto input_data = reinterpret_cast<int8_t *>(input->MutableData());
  for (auto i = 0; i < input->ElementsNum(); i++) {
    begin_src_data_[i] = static_cast<int32_t>(input_data[i]);
  }
  src_data_ = begin_src_data_;
  for (size_t i = 0; i < data_buffers_.size(); ++i) {
    GetQuantArgs(i);
    dst_data_ = data_buffers_[i];
    outer_size_ = outer_sizes_[i];
    inner_size_ = inner_sizes_[i];
    axis_size_ = axis_sizes_[i];
    auto error_code = ParallelLaunch(this->context_->thread_pool_, ReduceInt8Impl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      FreeTmpBuffer();
      MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
    src_data_ = dst_data_;
  }

  GetQuantArgs(static_cast<size_t>(num_axes_ - 1));
  outer_size_ = outer_sizes_.back();
  inner_size_ = inner_sizes_.back();
  axis_size_ = axis_sizes_.back();
  last_dst_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  is_last_axis_ = true;
  auto error_code = ParallelLaunch(this->context_->thread_pool_, ReduceInt8Impl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Reduce run error, error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  FreeTmpBuffer();
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceInt8CPUKernel>)
}  // namespace mindspore::kernel

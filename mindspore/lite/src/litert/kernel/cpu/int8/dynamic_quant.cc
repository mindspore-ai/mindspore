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
#include "src/litert/kernel/cpu/int8/dynamic_quant.h"
#include <set>
#include <vector>
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"
#include "nnacl/int8/dynamic_quant_int8.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/fp32/transpose_fp32.h"
#include "nnacl/int8/transpose_int8.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DynamicQuant;
namespace mindspore::kernel {
namespace {
constexpr int kBucketNums = 8;
constexpr int k8Bit = 8;
constexpr int kMinNums = 512;
constexpr float kDefaultRange = 0.01;
}  // namespace
int DynamicQuantCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  auto in_tensor = in_tensors_.front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors_.front();
  CHECK_NULL_RETURN(out_tensor);
  param_ = reinterpret_cast<DynamicQuantParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);
  MS_CHECK_TRUE_MSG(param_->dst_type_ == out_tensor->data_type(), lite::RET_ERROR,
                    "param data type and tensor data type do not match.");
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DynamicQuantCPUKernel::ReSize() {
  auto in_tensor = in_tensors_.front();
  auto ele_num = static_cast<int>(in_tensor->ElementsNum());
  auto shape = in_tensor->shape();
  int segment_num = 1;
  if (param_->axis_num_ == 0) {
    segment_num = MSMIN(kBucketNums, ele_num / kMinNums);
  } else {
    std::set<int> prefer_axes;
    for (int i = 0; i < param_->axis_num_; ++i) {
      int axis = param_->prefer_axes_[i] < 0 ? param_->prefer_axes_[i] + static_cast<int>(shape.size())
                                             : param_->prefer_axes_[i];
      MS_CHECK_TRUE_MSG(axis >= 0 && axis < static_cast<int>(shape.size()), lite::RET_ERROR,
                        "The prefer axis is out of range.");
      if (prefer_axes.find(axis) != prefer_axes.end()) {
        continue;
      }
      segment_num *= shape[axis];
      (void)prefer_axes.insert(axis);
    }
    pre_perm_.resize(shape.size());
    post_perm_.resize(shape.size());
    int pre_point0 = 0;
    int pre_point1 = param_->axis_num_;
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
      if (prefer_axes.find(i) != prefer_axes.end()) {
        pre_perm_[pre_point0] = i;
        post_perm_[i] = pre_point0;
        ++pre_point0;
      } else {
        pre_perm_[pre_point1] = i;
        post_perm_[i] = pre_point1;
        ++pre_point1;
      }
    }
  }
  need_transpose_ = false;
  for (size_t i = 0; i < pre_perm_.size(); ++i) {
    if (pre_perm_[i] != static_cast<int>(i)) {
      need_transpose_ = true;
    }
  }
  if (segment_num <= 0) {
    segment_num = 1;
  }
  real_min_.resize(segment_num);
  real_max_.resize(segment_num);
  scale_.resize(segment_num);
  zero_point_.resize(segment_num);
  for (int i = 0; i < segment_num; ++i) {
    real_min_[i] = FLT_MAX;
    real_max_[i] = -FLT_MAX;
  }
  thread_num_ = MSMIN(segment_num, op_parameter_->thread_num_);
  unit_num_ = UP_DIV(ele_num, segment_num);
  unit_segment_num_ = UP_DIV(segment_num, thread_num_);
  return RET_OK;
}

int DynamicQuantCPUKernel::CalculateMinMax(int task_id) {
  int task_unit = unit_segment_num_ * unit_num_;
  int offset = task_id * task_unit;
  int ele_num = static_cast<int>(in_tensors_.front()->ElementsNum());
  int remain = ele_num - offset;
  if (task_unit <= remain) {
    for (int i = 0; i < unit_segment_num_; ++i) {
      CalculateMinMaxFp32(float32_ptr_ + offset + i * unit_num_, unit_num_, &real_min_[task_id * unit_segment_num_ + i],
                          &real_max_[task_id * unit_segment_num_ + i]);
    }
  } else {
    int segment_num = remain / unit_num_;
    int remain_ele_num = remain - segment_num * unit_num_;
    for (int i = 0; i < segment_num; ++i) {
      CalculateMinMaxFp32(float32_ptr_ + offset + i * unit_num_, unit_num_, &real_min_[task_id * unit_segment_num_ + i],
                          &real_max_[task_id * unit_segment_num_ + i]);
    }
    if (remain_ele_num == 0) {
      return RET_OK;
    }
    CalculateMinMaxFp32(float32_ptr_ + offset + segment_num * unit_num_, remain_ele_num,
                        &real_min_[task_id * unit_segment_num_ + segment_num],
                        &real_max_[task_id * unit_segment_num_ + segment_num]);
  }
  return RET_OK;
}

int CalculateMinMaxRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<DynamicQuantCPUKernel *>(cdata);
  auto ret = g_kernel->CalculateMinMax(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CalculateMinMaxRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

void DynamicQuantCPUKernel::CalculatePerlayerScaleZp() {
  float real_min = FLT_MAX;
  float real_max = -FLT_MAX;
  for (size_t i = 0; i < real_min_.size(); ++i) {
    real_min = (real_min_[i] < real_min) ? real_min_[i] : real_min;
    real_max = (real_max_[i] > real_max) ? real_max_[i] : real_max;
  }

  lite::LiteQuantParam quant_parm;
  double scale;
  int zp = 0;
  constexpr int kQSymmetricRange = 255;
  constexpr int kQAsymmetricRange = 254;
  if (!param_->symmetric_) {
    auto range = real_max - real_min;
    if (range <= 0) {
      range = kDefaultRange;
      MS_LOG(WARNING) << name_ << " range is 0 and set the range to 0.01.";
    }
    scale = range / kQSymmetricRange;  // -128 ~ 127
    zp = static_cast<int>(std::round(INT8_MIN - real_min / scale));
  } else {
    auto max = std::max(abs(real_max), abs(real_min));
    scale = 2 * max / kQAsymmetricRange;  // -127 ~ 127
  }
  quant_parm.scale = scale;
  quant_parm.zeroPoint = zp;
  quant_parm.bitNum = k8Bit;
  quant_parm.inited = true;
  this->out_tensors_.front()->set_quant_params({quant_parm});
}

void DynamicQuantCPUKernel::CalculatePerChannelScaleZp() {
  std::vector<lite::LiteQuantParam> quant_params;
  for (size_t i = 0; i < real_min_.size(); ++i) {
    float real_min = real_min_[i];
    float real_max = real_max_[i];

    lite::LiteQuantParam quant_parm;
    double scale;
    int zp = 0;
    constexpr int kQSymmetricRange = 255;
    constexpr int kQAsymmetricRange = 254;
    if (!param_->symmetric_) {
      auto range = real_max - real_min;
      if (range <= 0) {
        range = kDefaultRange;
        MS_LOG(WARNING) << name_ << " range is 0 and set the range to 0.01.";
      }
      scale = range / kQSymmetricRange;  // -128 ~ 127
      zp = static_cast<int>(std::round(INT8_MIN - real_min / scale));
    } else {
      auto max = std::max(abs(real_max), abs(real_min));
      scale = 2 * max / kQAsymmetricRange;  // -127 ~ 127
    }
    quant_parm.scale = scale;
    quant_parm.zeroPoint = zp;
    quant_parm.bitNum = k8Bit;
    quant_parm.inited = true;
    quant_params.push_back(quant_parm);
  }
  this->out_tensors_.front()->set_quant_params(quant_params);
}

int DynamicQuantCPUKernel::QuantData(int task_id) {
  int task_unit = unit_segment_num_ * unit_num_;
  int offset = task_id * task_unit;
  int ele_num = static_cast<int>(in_tensors_.front()->ElementsNum());
  int remain = ele_num - offset;
  task_unit = MSMIN(task_unit, remain);
  if (param_->axis_num_ == 0) {  // per-tensor
    auto quant_arg = out_tensors_.front()->quant_params().front();
    auto ret = DoQuantizeFp32ToInt8(float32_ptr_ + offset, int8_ptr_ + offset, quant_arg.scale, quant_arg.zeroPoint,
                                    task_unit, (int32_t)INT8_MIN, (int32_t)INT8_MAX);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "QuantDTypeCast error task_id[" << task_id << "] error_code[" << ret << "]";
      return RET_ERROR;
    }
    return RET_OK;
  }
  int segment_num = task_unit / unit_num_;
  for (int i = 0; i < segment_num; ++i) {
    auto quant_arg = out_tensors_.front()->quant_params()[task_id * unit_segment_num_ + i];
    auto ret =
      DoQuantizeFp32ToInt8(float32_ptr_ + offset + i * unit_num_, int8_ptr_ + offset + i * unit_num_, quant_arg.scale,
                           quant_arg.zeroPoint, unit_num_, (int32_t)INT8_MIN, (int32_t)INT8_MAX);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "QuantDTypeCast error task_id[" << task_id << "] error_code[" << ret << "]";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int QuantDataRun(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<DynamicQuantCPUKernel *>(cdata);
  auto ret = g_kernel->QuantData(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CalculateMinMaxRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DynamicQuantCPUKernel::MallocTmpBuffer() {
  auto in_size = in_tensors_.front()->Size();
  auto out_size = out_tensors_.front()->Size();
  if (ms_context_ != nullptr && ms_context_->allocator != nullptr) {
    int8_ptr_ = static_cast<int8_t *>(ms_context_->allocator->Malloc(in_size + out_size));
  } else {
    int8_ptr_ = static_cast<int8_t *>(malloc(in_size + out_size));
  }
  MS_CHECK_TRUE_MSG(int8_ptr_ != nullptr, lite::RET_NULL_PTR, "DynamicQuant malloc tmp buffer failed.");
  float32_ptr_ = reinterpret_cast<float *>(int8_ptr_ + out_size);
  return lite::RET_OK;
}

void DynamicQuantCPUKernel::FreeTmpBuffer() {
  if (need_transpose_) {
    if (int8_ptr_ != nullptr) {
      if (ms_context_ != nullptr && ms_context_->allocator != nullptr) {
        ms_context_->allocator->Free(int8_ptr_);
      } else {
        free(int8_ptr_);
      }
    }
  }
  int8_ptr_ = nullptr;
  float32_ptr_ = nullptr;
}

int DynamicQuantCPUKernel::Run() {
  std::vector<int> transpose_shape;
  if (need_transpose_) {
    auto shape = in_tensors_.front()->shape();
    transpose_shape.resize(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      transpose_shape[i] = shape[pre_perm_[i]];
    }
    if (MallocTmpBuffer() != lite::RET_OK) {
      MS_LOG(ERROR) << "DynamicQuant MallocTmpBuffer failed.";
      return lite::RET_NULL_PTR;
    }
    std::vector<int> strides(shape.size(), 1);
    std::vector<int> out_strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - C2NUM; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
      out_strides[i] = transpose_shape[i + 1] * out_strides[i + 1];
    }
    if (shape.size() <= C6NUM) {
      (void)DoTransposeFp32(in_tensors_.front()->data(), float32_ptr_, transpose_shape.data(), pre_perm_.data(),
                            strides.data(), out_strides.data(), in_tensors_.front()->Size(), shape.size());
    } else {
      TransposeDimsFp32(in_tensors_.front()->data(), float32_ptr_, transpose_shape.data(), pre_perm_.data(),
                        strides.data(), out_strides.data(), shape.size(), 0, 1);
    }
  } else {
    int8_ptr_ = reinterpret_cast<int8_t *>(out_tensors_[0]->data());
    float32_ptr_ = reinterpret_cast<float *>(in_tensors_[0]->data());
  }
  if (int8_ptr_ == nullptr || float32_ptr_ == nullptr) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "DynamicQuant's original data exists nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto ret = ParallelLaunch(this->ms_context_, CalculateMinMaxRun, this, thread_num_);
  if (ret != RET_OK) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "Run error error_code[" << ret << "]";
    return RET_ERROR;
  }
  if (param_->axis_num_ != 0) {
    CalculatePerChannelScaleZp();
  } else {
    CalculatePerlayerScaleZp();
  }
  ret = ParallelLaunch(this->ms_context_, QuantDataRun, this, thread_num_);
  if (ret != RET_OK) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "Run error error_code[" << ret << "]";
    return RET_ERROR;
  }
  if (need_transpose_) {
    auto out_shape = out_tensors_.front()->shape();
    TransposeParameter trans_parameter;
    (void)memset(&trans_parameter, 0, sizeof(TransposeParameter));
    trans_parameter.op_parameter_.thread_num_ = 1;
    trans_parameter.num_axes_ = static_cast<int>(out_shape.size());
    trans_parameter.data_num_ = out_tensors_[0]->ElementsNum();
    trans_parameter.perm_size_ = post_perm_.size();
    int last_index = static_cast<int>(out_shape.size()) - 1;
    trans_parameter.perm_[last_index] = post_perm_[last_index];
    trans_parameter.strides_[last_index] = 1;
    trans_parameter.out_strides_[last_index] = 1;
    for (int i = last_index - 1; i >= 0; i--) {
      trans_parameter.perm_[i] = post_perm_[i];
      trans_parameter.strides_[i] = transpose_shape[i + 1] * trans_parameter.strides_[i + 1];
      trans_parameter.out_strides_[i] = out_shape[i + 1] * trans_parameter.out_strides_[i + 1];
    }
    if (out_shape.size() <= C6NUM) {
      (void)DoTransposeInt8(int8_ptr_, reinterpret_cast<int8_t *>(out_tensors_[0]->data()), out_shape.data(),
                            &trans_parameter);
    } else {
      TransposeDimsInt8(int8_ptr_, reinterpret_cast<int8_t *>(out_tensors_[0]->data()), out_shape.data(),
                        &trans_parameter, 0, 1);
    }
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *DynamicQuantCPUCreator(const std::vector<lite::Tensor *> &inputs,
                                           const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                           const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) DynamicQuantCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << "is nullptr.";
    free(parameter);
    return nullptr;
  }
  if (inputs.size() != 1) {
    MS_LOG(ERROR) << "inputs number should be 1, but " << inputs.size() << " is given.";
    return nullptr;
  }
  if (outputs.size() != 1) {
    MS_LOG(ERROR) << "outputs number should be 1, but " << outputs.size() << " is given.";
    return nullptr;
  }
  bool support_dtype =
    inputs[0]->data_type() == TypeId::kNumberTypeFloat32 && outputs[0]->data_type() == TypeId::kNumberTypeInt8;
  if (!support_dtype) {
    MS_LOG(ERROR) << "Unsupported data type input:" << inputs.front()->data_type()
                  << ", output:" << outputs.front()->data_type();
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DynamicQuant, DynamicQuantCPUCreator)
}  // namespace mindspore::kernel

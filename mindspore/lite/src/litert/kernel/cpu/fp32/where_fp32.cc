/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/where_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/fp32/where_fp32.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Where;

namespace mindspore::kernel {
constexpr uint32_t kSingleNum = 1;
constexpr uint32_t kTripleNum = 3;
constexpr int kStrideMaxSize = 8;
int WhereCPUKernel::Prepare() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kSingleNum || in_tensors_.size() == kTripleNum, RET_ERROR);
  MS_CHECK_TRUE_RET(out_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  where_param_->op_parameter_.thread_num_ = ms_context_->thread_num_;
  return RET_OK;
}

int WhereCPUKernel::PreProcess() {
  if (in_tensors_.size() == kTripleNum) {
    return LiteKernel::PreProcess();
  } else {
    return RET_OK;
  }
}

int WhereCPUKernel::DoExcute(int task_id) {
  CHECK_NULL_RETURN(condition_);
  CHECK_NULL_RETURN(x_);
  CHECK_NULL_RETURN(y_);
  CHECK_NULL_RETURN(output_data_);
  CHECK_NULL_RETURN(where_param_);
  WhereWithTripleInputs(condition_, static_cast<float *>(x_), static_cast<float *>(y_),
                        static_cast<float *>(output_data_), where_param_, task_id);
  return RET_OK;
}

int WhereRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<WhereCPUKernel *>(cdata);
  auto ret = kernel->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WhereRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int WhereCPUKernel::RunWithSingleInput() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  auto input_data_type = input->data_type();
  switch (input_data_type) {
    case kNumberTypeInt32:
      int32_condition_ = reinterpret_cast<int32_t *>(input->data());
      CHECK_NULL_RETURN(int32_condition_);
      break;
    case kNumberTypeFloat32:
      fp32_condition_ = reinterpret_cast<float *>(input->data());
      CHECK_NULL_RETURN(fp32_condition_);
      break;
    case kNumberTypeBool:
      condition_ = reinterpret_cast<bool *>(input->data());
      CHECK_NULL_RETURN(condition_);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported data type: " << input_data_type << " of where cpu kernel.";
      return RET_ERROR;
  }
  where_param_->condition_num_ = input->ElementsNum();
  where_param_->rank_ = static_cast<int>(input->shape().size());
  int strides[kStrideMaxSize];
  ComputeStrides(in_tensors_.at(0)->shape().data(), strides, where_param_->rank_);
  MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(where_param_->condition_num_, where_param_->rank_), RET_ERROR, "mul overflow");
  int data_num_int = where_param_->condition_num_ * where_param_->rank_;
  MS_CHECK_TRUE_RET(data_num_int >= 0, RET_ERROR);
  size_t data_num = static_cast<size_t>(data_num_int);
  size_t data_size = data_num * sizeof(int32_t);
  auto data = ms_context_->allocator->Malloc(data_size);
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc data is error!";
    return RET_ERROR;
  }
  int32_t *result = reinterpret_cast<int32_t *>(data);

  int result_index = 0;
  int true_num = 0;
  for (int index = 0; index < where_param_->condition_num_; index++) {
    bool condition = false;
    switch (input_data_type) {
      case kNumberTypeInt32:
        condition = static_cast<bool>(int32_condition_[index]);
        break;
      case kNumberTypeFloat32:
        condition = static_cast<bool>(fp32_condition_[index]);
        break;
      case kNumberTypeBool:
        condition = static_cast<bool>(condition_[index]);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported data type: " << input_data_type << " of where cpu kernel.";
        return RET_ERROR;
    }
    if (condition) {
      true_num++;
      int dim = index;
      for (int j = 0; j < where_param_->rank_; j++) {
        MS_CHECK_FALSE_MSG(strides[j] == 0, RET_ERROR, "div zero");
        result[result_index++] = dim / strides[j];
        dim %= strides[j];
      }
    }
  }
  auto origin_output_shape = out_tensors_.at(0)->shape();
  std::vector<int> output_shape = {true_num, where_param_->rank_};
  out_tensors_.at(0)->set_shape(output_shape);
  out_tensors_.at(0)->FreeData();
  out_tensors_.at(0)->set_shape_changed(origin_output_shape != output_shape);
  if (true_num > 0) {
    auto out_data = out_tensors_.at(0)->MutableData();
    if (out_data == nullptr) {
      MS_LOG(ERROR) << "malloc out tensor failed.";
      return RET_ERROR;
    }
    MS_CHECK_GE(where_param_->condition_num_, true_num, RET_ERROR);
    (void)memcpy(out_data, result, true_num * where_param_->rank_ * static_cast<int>(sizeof(int32_t)));
  }
  ms_context_->allocator->Free(data);
  return RET_OK;
}

int WhereCPUKernel::RunWithTripleInputs() {
  auto condition = in_tensors_.at(0);
  CHECK_NULL_RETURN(condition);
  auto x = in_tensors_.at(1);
  CHECK_NULL_RETURN(x);
  auto y = in_tensors_.at(C2NUM);
  CHECK_NULL_RETURN(y);
  int condition_nums = condition->ElementsNum();
  int x_num = x->ElementsNum();
  int y_num = y->ElementsNum();
  int out_num = out_tensors_.front()->ElementsNum();

  condition_ = reinterpret_cast<bool *>(condition->data());
  CHECK_NULL_RETURN(condition_);
  x_ = x->data();
  CHECK_NULL_RETURN(x_);
  y_ = y->data();
  CHECK_NULL_RETURN(y_);
  output_data_ = out_tensors_.at(0)->data();
  int num_max = condition_nums > x_num ? condition_nums : (x_num > y_num ? x_num : y_num);
  where_param_->condition_num_ = condition_nums;
  where_param_->x_num_ = x_num;
  where_param_->y_num_ = y_num;
  where_param_->max_num_ = num_max;

  CHECK_LESS_RETURN(out_num, num_max);

  if (((condition_nums != 1) && (condition_nums != num_max)) || ((x_num != 1) && (x_num != num_max)) ||
      ((y_num != 1) && (y_num != num_max))) {
    MS_LOG(ERROR) << "The length of three inputs are not equal to 1 or length of output, which is unacceptable";
    return RET_ERROR;
  }
  if (num_max <= 0) {
    MS_LOG(ERROR) << "Error, inputs' length are zero !!!";
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->ms_context_, WhereRun, this, where_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WhereDwRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int WhereCPUKernel::Run() {
  int ret = RET_ERROR;
  if (in_tensors_.size() == kSingleNum) {
    ret = RunWithSingleInput();
  } else if (in_tensors_.size() == kTripleNum) {
    ret = RunWithTripleInputs();
  } else {
    MS_LOG(ERROR) << "in tensor size is invalid. size is " << in_tensors_.size();
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Where op run failed.";
  }
  for (auto *output : this->out_tensors()) {
    output->ResetRefCount();
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Where, LiteKernelCreator<WhereCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Where, LiteKernelCreator<WhereCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Where, LiteKernelCreator<WhereCPUKernel>)
}  // namespace mindspore::kernel

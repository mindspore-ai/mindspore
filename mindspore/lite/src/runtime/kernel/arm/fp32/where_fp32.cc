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
#include "src/runtime/kernel/arm/fp32/where_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/fp32/where_fp32.h"
#include "src/kernel_registry.h"
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
int WhereCPUKernel::Init() {
  MS_CHECK_TRUE_RET(in_tensors_.size() == kSingleNum || in_tensors_.size() == kTripleNum, RET_ERROR);
  MS_CHECK_TRUE_RET(out_tensors_.size() == 1, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  where_param_->op_parameter_.thread_num_ = thread_count_;
  return RET_OK;
}

int WhereCPUKernel::PreProcess() {
  if (in_tensors_.size() == kTripleNum) {
    return InnerKernel::PreProcess();
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
  WhereWithTripleInputs(condition_, x_, y_, output_data_, where_param_, task_id);
  return RET_OK;
}

int WhereRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto wheredata = reinterpret_cast<WhereCPUKernel *>(cdata);
  auto ret = wheredata->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "WhereRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int WhereCPUKernel::RunWithSingleInput() {
  auto input = in_tensors_.at(0);
  MS_ASSERT(input);
  condition_ = reinterpret_cast<bool *>(input->data());
  CHECK_NULL_RETURN(condition_);
  where_param_->condition_num_ = input->ElementsNum();
  where_param_->rank_ = static_cast<int>(input->shape().size());
  int strides[8];
  ComputeStrides(in_tensors_.at(0)->shape().data(), strides, where_param_->rank_);
  auto data = ms_context_->allocator->Malloc(where_param_->condition_num_ * where_param_->rank_ *
                                             static_cast<int>(sizeof(int32_t)));
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc data is error!";
    return RET_ERROR;
  }
  int *result = reinterpret_cast<int *>(data);

  int result_index = 0;
  int true_num = 0;
  for (int index = 0; index < where_param_->condition_num_; index++) {
    if (condition_[index]) {
      true_num++;
      int dim = index;
      for (int j = 0; j < where_param_->rank_; j++) {
        MS_CHECK_FALSE_MSG(strides[j] == 0, RET_ERROR, "div zero");
        result[result_index++] = dim / strides[j];
        dim %= strides[j];
      }
    }
  }
  std::vector<int> output_shape = {true_num, where_param_->rank_};
  out_tensors_.at(0)->set_shape(output_shape);
  out_tensors_.at(0)->FreeData();
  auto out_data = out_tensors_.at(0)->MutableData();
  if (out_data == nullptr) {
    MS_LOG(ERROR) << "malloc out tensor failed.";
    return RET_ERROR;
  }
  MS_CHECK_GE(where_param_->condition_num_, true_num, RET_ERROR);
  memcpy(out_data, result, true_num * where_param_->rank_ * static_cast<int>(sizeof(int32_t)));
  ms_context_->allocator->Free(data);
  return RET_OK;
}

int WhereCPUKernel::RunWithTripleInputs() {
  auto condition = in_tensors_.at(0);
  CHECK_NULL_RETURN(condition);
  auto x = in_tensors_.at(1);
  CHECK_NULL_RETURN(x);
  auto y = in_tensors_.at(2);
  CHECK_NULL_RETURN(y);
  int condition_nums = condition->ElementsNum();
  int x_num = x->ElementsNum();
  int y_num = y->ElementsNum();

  condition_ = reinterpret_cast<bool *>(condition->data());
  CHECK_NULL_RETURN(condition_);
  x_ = reinterpret_cast<float *>(x->data());
  CHECK_NULL_RETURN(x_);
  y_ = reinterpret_cast<float *>(y->data());
  CHECK_NULL_RETURN(y_);
  output_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  int num_max = condition_nums > x_num ? condition_nums : (x_num > y_num ? x_num : y_num);
  where_param_->condition_num_ = condition_nums;
  where_param_->x_num_ = x_num;
  where_param_->y_num_ = y_num;
  where_param_->max_num_ = num_max;

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

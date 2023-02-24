/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp16/reduce_fp16.h"
#include <map>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp16/reduce_fp16.h"
#include "src/litert/kernel/cpu/base/reduce_base.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceASum;
using mindspore::schema::ReduceMode_ReduceL2;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {
int ReduceFp16CPUKernel::CallReduceUnit(int task_id) {
  CHECK_NULL_RETURN(src_data_);
  CHECK_NULL_RETURN(dst_data_);
  if (fp16_reducer_ == nullptr) {
    MS_LOG(ERROR) << "function reducer_ is null.";
    return RET_NULL_PTR;
  }
  fp16_reducer_(outer_size_, inner_size_, axis_size_, static_cast<const float16_t *>(src_data_),
                static_cast<float16_t *>(dst_data_), task_id, thread_num_);
  return RET_OK;
}

void ReduceFp16CPUKernel::InitialKernelList() {
  std::map<schema::ReduceMode, Fp16Reducer> func_list{
    {ReduceMode_ReduceSum, ReduceSumFp16},   {ReduceMode_ReduceMean, ReduceMeanFp16},
    {ReduceMode_ReduceMax, ReduceMaxFp16},   {ReduceMode_ReduceMin, ReduceMinFp16},
    {ReduceMode_ReduceProd, ReduceProdFp16}, {ReduceMode_ReduceSumSquare, ReduceSumFp16},
    {ReduceMode_ReduceASum, ReduceSumFp16},  {ReduceMode_ReduceL2, ReduceL2NormFp16}};
  auto iter = func_list.find(static_cast<schema::ReduceMode>(mode_));
  if (iter != func_list.end()) {
    fp16_reducer_ = iter->second;
  }
}

void ReduceFp16CPUKernel::HandleASumAndSumSquare() {
  int num = in_tensors_[kInputIndex]->ElementsNum();
  float16_t *data = static_cast<float16_t *>(in_tensors_[kInputIndex]->data());
  NNACL_CHECK_NULL_RETURN_VOID(data);

  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceASum)) {
    for (int i = 0; i < num; ++i) {
      if (data[i] < 0.0f) {
        data[i] = 0.0f - data[i];
      }
    }
  }
  if (reduce_param_->mode_ == static_cast<int>(ReduceMode_ReduceSumSquare)) {
    for (int i = 0; i < num; ++i) {
      data[i] = data[i] * data[i];
    }
  }
}

int ReduceFp16CPUKernel::CalculateCoeffOutput() {
  auto out_tensor = out_tensors_[kOutputIndex];
  int num = out_tensor->ElementsNum();
  auto *out_data = reinterpret_cast<float16_t *>(out_tensor->data());
  if (out_data == nullptr) {
    return RET_NULL_PTR;
  }
  for (int i = 0; i < num; ++i) {
    out_data[i] *= reduce_param_->coeff;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ReduceFusion, LiteKernelCreator<ReduceFp16CPUKernel>)
}  // namespace mindspore::kernel

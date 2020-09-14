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

#include "internal/src/kernel/fp32/arithmetic_self.h"
#include "internal/src/kernel/common/common_infershape.h"
#include "internal/include/errorcode.h"
#include "internal/include/ms_tensor.h"
#include "utils/log_adapter.h"
#include "nnacl/fp32/arithmetic_self.h"

int DoArithmeticSelfInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors,
                               OpParameter *param) {
  return DoCommonInferShape(in_tensors, out_tensors);
}

int DoArithmeticSelf(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                     mindspore::lite::Allocator *allocator) {
  size_t data_size = in_tensors[0]->ElementsNum();
  OpParameter *param = node->primitive_;
  int ret;
  if (param->type_ == KernelType::Log) {
    ret = ElementLog((float *)in_tensors[0]->data_, (float *)out_tensors[0]->data_, data_size);
  } else if (param->type_ == KernelType::Neg) {
    ret = ElementNegative((float *)in_tensors[0]->data_, (float *)out_tensors[0]->data_, data_size);
  } else {
    MS_LOG(ERROR) << "Unsupport kernel type: " << param->type_;
    return RET_PARAM_INVALID;
  }
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "do arithmetic " << param->type_ << " fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

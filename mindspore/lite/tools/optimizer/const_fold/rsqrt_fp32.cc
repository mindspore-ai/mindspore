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
#include "tools/optimizer/const_fold/rsqrt_fp32.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int HighAccuracyRsqrtCPUKernel::Prepare() {
  CHECK_NOT_EQUAL_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}
int HighAccuracyRsqrtCPUKernel::ReSize() { return RET_OK; }

int HighAccuracyRsqrtCPUKernel::Run() {
  int elements_num = in_tensors_.at(0)->ElementsNum();
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    float *input_ptr = reinterpret_cast<float *>(in_tensors_.at(0)->data());
    float *output_ptr = reinterpret_cast<float *>(out_tensors_.at(0)->data());
    for (int i = 0; i < elements_num; i++) {
      if (MS_UNLIKELY(input_ptr[i] < 0)) {
        return RET_ERROR;
      }
      output_ptr[i] = 1.0 / sqrt(static_cast<double>(input_ptr[i]));
    }
  } else {
    MS_LOG(ERROR) << "Unsupported type: " << in_tensors_[0]->data_type() << ".";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

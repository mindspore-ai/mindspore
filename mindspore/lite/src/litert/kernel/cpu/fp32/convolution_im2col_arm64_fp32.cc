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

#include "src/litert/kernel/cpu/fp32/convolution_im2col_arm64_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionIm2ColARM64CPUKernel::InitGlobalVariable() {
  oc_tile_ = C8NUM;
  row_tile_ = C12NUM;

  rowMajor2ColNMajorFunc = RowMajor2Col8Major;
}

int ConvolutionIm2ColARM64CPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
  CHECK_NULL_RETURN(ori_input_data);
  if (out_tensors_[0]->format() != NC4HW4) {
    if (use_batch_cut_flag_) {
      ConvFp32CutByBatch(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                         reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    } else {
      ConvFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
               reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    }
  } else {
    ConvFp32OutNC4HW4(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                      reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

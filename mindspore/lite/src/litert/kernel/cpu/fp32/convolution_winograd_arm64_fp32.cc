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

#include "src/litert/kernel/cpu/fp32/convolution_winograd_arm64_fp32.h"
#include "nnacl/fp32/winograd_utils.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionWinogradARM64CPUKernel::InitGlobalVariable() {
  oc_block_ = C8NUM;
  tmp_data_tile_ = C4NUM;
  tile_num_ = C12NUM;
}

int ConvolutionWinogradARM64CPUKernel::ConfigInputOutput() {
  trans_func_.in_func_ = GetInputTransFunc(input_unit_);
  if (trans_func_.in_func_ == nullptr) {
    MS_LOG(ERROR) << "in_func_ is null.";
    return RET_ERROR;
  }

  trans_func_.in_step_func_ = GetInputTransStepFunc(input_unit_);
  if (trans_func_.in_step_func_ == nullptr) {
    MS_LOG(DEBUG) << "in_step_func_ is null.";
  }
  trans_func_.in_pack_func_ = GetInputTransPackFunc(input_unit_);
  if (trans_func_.in_pack_func_ == nullptr) {
    MS_LOG(DEBUG) << "in_pack_func_ is null.";
  }

  trans_func_.out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  if (trans_func_.out_func_ == nullptr) {
    MS_LOG(ERROR) << "out_func_ is null.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

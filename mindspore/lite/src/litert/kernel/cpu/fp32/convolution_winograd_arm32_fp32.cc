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

#include "src/litert/kernel/cpu/fp32/convolution_winograd_arm32_fp32.h"

namespace mindspore::kernel {
void ConvolutionWinogradARM32CPUKernel::InitGlobalVariable() {
  oc_block_ = C8NUM;
  tmp_data_tile_ = C4NUM;
  tile_num_ = C12NUM;
}
}  // namespace mindspore::kernel

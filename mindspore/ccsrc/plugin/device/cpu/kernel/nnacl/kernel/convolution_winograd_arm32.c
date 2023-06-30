/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef ENABLE_ARM32
#include "nnacl/kernel/convolution_winograd_arm32.h"

void ConvWinoARM32InitGlobalVariable(ConvolutionBaseStruct *conv) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)conv;
  winograd->oc_block_ = C8NUM;
  winograd->tmp_data_tile_ = C4NUM;
  winograd->tile_num_ = C12NUM;
}

ConvolutionWinogradBaseStruct *CreateConvWinogradARM32(ConvParameter *conv_param) {
  ConvolutionWinogradBaseStruct *winograd =
    (ConvolutionWinogradBaseStruct *)malloc(sizeof(ConvolutionWinogradBaseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(winograd);
  memset(winograd, 0, sizeof(ConvolutionWinogradBaseStruct));

  winograd->config_input_output_ = ConvWinoBaseConfigInputOutput;
  winograd->conv_.init_global_variable_ = ConvWinoARM32InitGlobalVariable;

  winograd->conv_.base_.Prepare = ConvolutionWinogradBasePrepare;
  winograd->conv_.base_.Resize = ConvolutionWinogradBaseResize;
  winograd->conv_.base_.Release = ConvolutionWinogradBaseRelease;
  winograd->conv_.base_.Compute = ConvolutionWinogradBaseCompute;
  return winograd;
}
#endif

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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/deconvolution_winograd.h"

int deconv_winograd_prepare(KernelBase *self) { return NNACL_OK; }
int deconv_winograd_resize(KernelBase *self) { return NNACL_OK; }
int deconv_winograd_release(KernelBase *self) { return NNACL_OK; }
int deconv_winograd_compute(KernelBase *self) { return NNACL_OK; }

ConvolutionBaseStruct *CreateDeConvWinograd(ConvParameter *param) {
  DeConvWinogradStruct *deconv_winograd = (DeConvWinogradStruct *)malloc(sizeof(DeConvWinogradStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(deconv_winograd);
  deconv_winograd->conv_.base_.prepare = deconv_winograd_prepare;
  deconv_winograd->conv_.base_.resize = deconv_winograd_resize;
  deconv_winograd->conv_.base_.release = deconv_winograd_release;
  deconv_winograd->conv_.base_.compute = deconv_winograd_compute;
  return &deconv_winograd->conv_;
}

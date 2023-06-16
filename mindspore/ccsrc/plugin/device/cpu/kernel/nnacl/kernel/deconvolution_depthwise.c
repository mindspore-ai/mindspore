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

#include "nnacl/kernel/deconvolution_depthwise.h"

int deconv_dw_prepare(KernelBase *self) { return NNACL_OK; }
int deconv_dw_resize(KernelBase *self) { return NNACL_OK; }
int deconv_dw_release(KernelBase *self) { return NNACL_OK; }
int deconv_dw_compute(KernelBase *self) { return NNACL_OK; }

ConvolutionBaseStruct *CreateDeConvDw(ConvParameter *param) {
  DeConvDwStruct *deconv_dw = (DeConvDwStruct *)malloc(sizeof(DeConvDwStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(deconv_dw);
  deconv_dw->conv_.base_.prepare = deconv_dw_prepare;
  deconv_dw->conv_.base_.resize = deconv_dw_resize;
  deconv_dw->conv_.base_.release = deconv_dw_release;
  deconv_dw->conv_.base_.compute = deconv_dw_compute;
  return &deconv_dw->conv_;
}

/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/experimental/conv.h"
#include "nnacl/experimental/conv_fp32_nchwx_avx512.h"
#include "nnacl/tensor_c.h"
#include "nnacl/op_base.h"
#include "nnacl/kernel.h"

KernelBase *CreateConv(OpParameter *param, TensorC *in, size_t insize, TensorC *out, size_t outsize) {
  if (in[0].format_ == Format_NHWC) {
    return NULL;
  } else if (in[0].format_ == Format_NCHW) {
    if (in[0].format_ != Format_NC16HW16) {
      return NULL;
    }
    KConv2d *conv = (KConv2d *)malloc(sizeof(KConv2d));
    if (conv == NULL) {
      return NULL;
    }
    conv->base.param = param;
    conv->base.in = in;
    conv->base.insize = insize;
    conv->base.out = out;
    conv->base.outsize = outsize;

    conv->base.prepare = conv2d_prepare_fp32_nchwx_avx512;
    conv->base.compute = conv2d_compute_fp32_nchwx_avx512;
    conv->base.release = conv2d_release_fp32_nchwx_avx512;
    conv->base.resize = conv2d_resize_fp32_nchwx_avx512;
    conv->base.inferShape = conv2d_infershape_fp32_nchwx_avx512;

    conv->base.funcs = GetCoreFuncs(in[0].data_type_ == kNumberTypeFloat16);

    return (KernelBase *)conv;
  } else {
    return NULL;
  }
  return NULL;
}

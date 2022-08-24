/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "nnacl/fp32/scale_fp32.h"
#include "nnacl/base/scale_base.h"
#include "nnacl/scale_fp32_simd.h"

typedef void (*ScaleComputer)(const float *src, const float *scale, const float *bias, float *out,
                              const ScaleParameter *scale_param, const int block_info[C4NUM]);
ScalePatternOneTemplate(ScalePatternOneNoAct, DoScalePatternOne, ScalarNoAct, float);
ScalePatternTwoTemplate(ScalePatternTwoNoAct, DoScalePatternTwo, ScalarNoAct, float);
ScalePatternThirdTemplate(ScalePatternThirdNoAct, DoScalePatternThird, ScalarNoAct, float);
ScalePatternOneTemplate(ScalePatternOneRelu, DoScalePatternOneRelu, ScalarRelu, float);
ScalePatternTwoTemplate(ScalePatternTwoRelu, DoScalePatternTwoRelu, ScalarRelu, float);
ScalePatternThirdTemplate(ScalePatternThirdRelu, DoScalePatternThirdRelu, ScalarRelu, float);
ScalePatternOneTemplate(ScalePatternOneRelu6, DoScalePatternOneRelu6, ScalarRelu6, float);
ScalePatternTwoTemplate(ScalePatternTwoRelu6, DoScalePatternTwoRelu6, ScalarRelu6, float);
ScalePatternThirdTemplate(ScalePatternThirdRelu6, DoScalePatternThirdRelu6, ScalarRelu6, float);
void DoScaleFp32(const float *src, const float *scale, const float *bias, float *out, const ScaleParameter *scale_param,
                 const int block[C2NUM]) {
  ScaleComputer computer[C3NUM][C3NUM] = {ScalePatternOneNoAct, ScalePatternTwoNoAct, ScalePatternThirdNoAct,
                                          ScalePatternOneRelu,  ScalePatternTwoRelu,  ScalePatternThirdRelu,
                                          ScalePatternOneRelu6, ScalePatternTwoRelu6, ScalePatternThirdRelu6};
  int divisor = scale_param->axis_size_ * scale_param->inner_size_;
  int begin = block[0];
  int block_info[C4NUM];
  block_info[0] = begin;
  block_info[1] = block[1];
  begin %= divisor;
  divisor = scale_param->inner_size_;
  block_info[C2NUM] = begin / divisor;
  block_info[C3NUM] = begin % divisor;
  int pre_index = scale_param->activation_type_ == 0 ? 0 : (scale_param->activation_type_ == 1 ? 1 : C2NUM);
  int post_index = scale_param->offset_align_to_axis_ ? (scale_param->inner_size_ == 1 ? 1 : 0) : C2NUM;
  computer[pre_index][post_index](src + block[0], scale, bias, out + block[0], scale_param, block_info);
}

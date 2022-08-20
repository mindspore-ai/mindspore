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

#include "nnacl/fp16/scale_fp16.h"
#include "nnacl/base/scale_base.h"
#ifdef ENABLE_NEON
#include "nnacl/fp16/neon/scale_fp16_neon.h"
#endif

typedef void (*ScaleFp16Computer)(const float16_t *src, const float16_t *scale, const float16_t *bias, float16_t *out,
                                  const ScaleParameter *scale_param, const int block_info[C4NUM]);
ScalePatternOneTemplate(ScaleF16PatternOneNoAct, DoScaleF16PatternOne, ScalarNoAct, float16_t);
ScalePatternTwoTemplate(ScaleF16PatternTwoNoAct, DoScaleF16PatternTwo, ScalarNoAct, float16_t);
ScalePatternThirdTemplate(ScaleF16PatternThirdNoAct, DoScaleF16PatternThird, ScalarNoAct, float16_t);
ScalePatternOneTemplate(ScaleF16PatternOneRelu, DoScaleF16PatternOneRelu, ScalarRelu, float16_t);
ScalePatternTwoTemplate(ScaleF16PatternTwoRelu, DoScaleF16PatternTwoRelu, ScalarRelu, float16_t);
ScalePatternThirdTemplate(ScaleF16PatternThirdRelu, DoScaleF16PatternThirdRelu, ScalarRelu, float16_t);
ScalePatternOneTemplate(ScaleF16PatternOneRelu6, DoScaleF16PatternOneRelu6, ScalarRelu6, float16_t);
ScalePatternTwoTemplate(ScaleF16PatternTwoRelu6, DoScaleF16PatternTwoRelu6, ScalarRelu6, float16_t);
ScalePatternThirdTemplate(ScaleF16PatternThirdRelu6, DoScaleF16PatternThirdRelu6, ScalarRelu6, float16_t);
void DoScaleFp16(const float16_t *src, const float16_t *scale, const float16_t *bias, float16_t *out,
                 const ScaleParameter *scale_param, const int block[C2NUM]) {
  ScaleFp16Computer computer[C3NUM][C3NUM] = {
    ScaleF16PatternOneNoAct, ScaleF16PatternTwoNoAct, ScaleF16PatternThirdNoAct,
    ScaleF16PatternOneRelu,  ScaleF16PatternTwoRelu,  ScaleF16PatternThirdRelu,
    ScaleF16PatternOneRelu6, ScaleF16PatternTwoRelu6, ScaleF16PatternThirdRelu6};
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

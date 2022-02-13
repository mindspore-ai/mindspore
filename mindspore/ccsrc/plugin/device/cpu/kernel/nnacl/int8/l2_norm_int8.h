/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_NNACL_INT8_L2_NORM_INT8_H_
#define MINDSPORE_NNACL_INT8_L2_NORM_INT8_H_

#include "nnacl/l2_norm_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif

int L2NormalizationInt8(const int8_t *input_data, int8_t *output_data, const L2NormParameter *param,
                        const L2NormQuantArg *quant_param, const int begin, const int end);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_L2_NORM_INT8_H_

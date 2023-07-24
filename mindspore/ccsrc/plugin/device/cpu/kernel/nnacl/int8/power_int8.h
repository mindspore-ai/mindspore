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

#ifndef NNACL_INT8_POWER_INT8_H_
#define NNACL_INT8_POWER_INT8_H_

#include "nnacl/op_base.h"
#include "nnacl/pow_parameter.h"
#include "nnacl/int8/quantize.h"

#ifdef __cplusplus
extern "C" {
#endif
int PowerInt8(const int8_t *input, const int8_t *exp_ptr, int8_t *output, int count, const PowQuantArg *args,
              bool broadcast, int scale, int shift);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_INT8_POWER_INT8_H_

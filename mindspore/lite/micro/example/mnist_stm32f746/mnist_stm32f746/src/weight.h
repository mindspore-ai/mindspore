
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

#include "CMSIS/NN/Include/arm_nnfunctions.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include <stdlib.h>
#include <string.h>
extern unsigned char *g_Buffer;
enum STATUS {
  RET_OK = 0,
  RET_ERROR = 1,
};

extern int g_thread_num;
extern const int8_t g_Weight1[];
extern const int32_t g_Weight2[];
extern const int8_t g_Weight3[];
extern const int32_t g_Weight4[];
extern const int8_t g_Weight6[];
extern const int32_t g_Weight7[];
extern const int8_t g_Weight8[];
extern const int32_t g_Weight9[];

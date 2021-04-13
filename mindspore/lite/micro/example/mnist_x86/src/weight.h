
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

#include "nnacl/common_func.h"
#include "nnacl/errorcode.h"
#include "nnacl/int8/common_func_int8.h"
#include "nnacl/int8/conv3x3_int8.h"
#include "nnacl/int8/conv_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/int8/pooling_int8.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"
#include "nnacl/int8/reshape_int8.h"
#include "nnacl/int8/softmax_int8.h"
#include "wrapper/int8/matmul_int8_wrapper.h"
#include <stdlib.h>
#include <string.h>
extern unsigned char *g_Buffer;
enum STATUS {
  RET_OK = 0,
  RET_ERROR = 1,
};

extern int g_thread_num;
extern int16_t g_Weight10[];
extern int32_t g_Weight11[];
extern int16_t g_Weight12[];
extern int32_t g_Weight13[];
extern int32_t *g_Weight14;
extern int8_t *g_Weight15;
extern int32_t *g_Weight16;
extern int32_t *g_Weight17;
extern int8_t *g_Weight18;
extern int32_t *g_Weight19;

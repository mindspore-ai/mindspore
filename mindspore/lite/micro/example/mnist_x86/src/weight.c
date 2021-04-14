
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

#include "weight.h"

int  g_thread_num = 1; 
unsigned char * g_Buffer = 0; 
int16_t g_Weight10[1536];
int32_t g_Weight11[12];
int16_t g_Weight12[3072];
int32_t g_Weight13[12];
int32_t *g_Weight14 = NULL;
int8_t *g_Weight15 = NULL;
int32_t *g_Weight16 = NULL;
int32_t *g_Weight17 = NULL;
int8_t *g_Weight18 = NULL;
int32_t *g_Weight19 = NULL;

int Init(void *weight_buffer, int weight_size) {
  if (weight_buffer == NULL) {
    return RET_ERROR;
  }
  struct ModelParameter {
    void *addr;
    size_t size;
    size_t offset;
  };
  int8_t *g_Weight6 = (weight_buffer + 9312);
  int32_t *g_Weight7 = (weight_buffer + 15312);
  int8_t *g_Weight8 = (weight_buffer + 15392);
  int32_t *g_Weight9 = (weight_buffer + 15592);

  struct ModelParameter model_params[] = {
    {g_Weight10, 3072, 0},
    {g_Weight11, 48, 3072},
    {g_Weight12, 6144, 3120},
    {g_Weight13, 48, 9264},
  };

  for(int i = 0; i < 4; ++i) {
    if (model_params[i].offset + model_params[i].size > weight_size) {
      return RET_ERROR;
    }
    memcpy(model_params[i].addr, (weight_buffer + model_params[i].offset), model_params[i].size);
  }
{
g_Weight14 = malloc(80);
if (g_Weight14 == NULL) {
  return RET_ERROR;
}
memset(g_Weight14, 0, 80);
memcpy(g_Weight14, g_Weight7, 80);
g_Weight16 = malloc(80);
if (g_Weight16 == NULL) {
  return RET_ERROR;
}
memset(g_Weight16, 0, 80);
g_Weight15 = malloc(6080);
if (g_Weight15 == NULL) {
  return RET_ERROR;
}
memset(g_Weight15, 0, 6080);
int init_filter_zp[1] = {0};
InitInt8MatrixB(g_Weight6, g_Weight16, g_Weight15, 1, 300, 20, 20, 304, 31, init_filter_zp, g_Weight14, true, false);
}
{
g_Weight17 = malloc(48);
if (g_Weight17 == NULL) {
  return RET_ERROR;
}
memset(g_Weight17, 0, 48);
memcpy(g_Weight17, g_Weight9, 48);
g_Weight19 = malloc(48);
if (g_Weight19 == NULL) {
  return RET_ERROR;
}
memset(g_Weight19, 0, 48);
g_Weight18 = malloc(384);
if (g_Weight18 == NULL) {
  return RET_ERROR;
}
memset(g_Weight18, 0, 384);
int init_filter_zp[1] = {0};
InitInt8MatrixB(g_Weight8, g_Weight19, g_Weight18, 1, 20, 10, 12, 32, 11, init_filter_zp, g_Weight17, true, false);
}
  return RET_OK;
}


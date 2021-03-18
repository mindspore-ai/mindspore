
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

#include "net_weight.h"

unsigned char * net_B = 0 ; 
int16_t net_W10[1536];
int32_t net_W11[12];
int16_t net_W12[3072];
int32_t net_W13[12];
int32_t *net_W14 = NULL;
int8_t *net_W15 = NULL;
int32_t *net_W16 = NULL;
int32_t *net_W17 = NULL;
int8_t *net_W18 = NULL;
int32_t *net_W19 = NULL;

int net_Init(void *weight_buffer, int weight_size) {
  if (weight_buffer == NULL) {
    return RET_ERROR;
  }
  int g_thread_num = 1;

  struct ModelParameter {
    void *addr;
    size_t size;
    size_t offset;
  };
  int8_t *net_W6 = (weight_buffer + 9312);
  int32_t *net_W7 = (weight_buffer + 15312);
  int8_t *net_W8 = (weight_buffer + 15392);
  int32_t *net_W9 = (weight_buffer + 15592);

  struct ModelParameter model_params[] = {
    {net_W10, 3072, 0},
    {net_W11, 48, 3072},
    {net_W12, 6144, 3120},
    {net_W13, 48, 9264},
  };

  for(int i = 0; i < 4; ++i) {
    if (model_params[i].offset + model_params[i].size > weight_size) {
      return RET_ERROR;
    }
    memcpy(model_params[i].addr, (weight_buffer + model_params[i].offset), model_params[i].size);
  }
{
net_W14 = malloc(80);
if (net_W14 == NULL) {
  return RET_ERROR;
}
memset(net_W14, 0, 80);
memcpy(net_W14, net_W7, 80);
net_W16 = malloc(80);
if (net_W16 == NULL) {
  return RET_ERROR;
}
memset(net_W16, 0, 80);
net_W15 = malloc(6080);
if (net_W15 == NULL) {
  return RET_ERROR;
}
memset(net_W15, 0, 6080);
const int init_filter_zp[20] = {1, 12, 3, 2, -10, -5, -11, 5, 12, 22, 16, 1, -5, 15, 13, 5, -10, -5, -6, 0};
InitInt8MatrixB(net_W6, net_W16, net_W15, 1, 300, 20, 20, 304, 0, init_filter_zp, net_W14, true, true);
}
{
net_W17 = malloc(48);
if (net_W17 == NULL) {
  return RET_ERROR;
}
memset(net_W17, 0, 48);
memcpy(net_W17, net_W9, 48);
net_W19 = malloc(48);
if (net_W19 == NULL) {
  return RET_ERROR;
}
memset(net_W19, 0, 48);
net_W18 = malloc(384);
if (net_W18 == NULL) {
  return RET_ERROR;
}
memset(net_W18, 0, 384);
const int init_filter_zp[10] = {7, -2, 9, 2, -6, 21, 16, 10, -19, 8};
InitInt8MatrixB(net_W8, net_W19, net_W18, 1, 20, 10, 12, 32, 0, init_filter_zp, net_W17, true, true);
}
  return RET_OK;
}


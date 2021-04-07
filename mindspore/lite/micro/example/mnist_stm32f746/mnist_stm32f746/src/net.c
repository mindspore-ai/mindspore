
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
#include "net.h"

static const unsigned char *g_Input0 = 0;
int SetInputs(const void **inputs, int num) {
  if (inputs == NULL) {
    return RET_ERROR;
  }
  if (num !=1) {
    return RET_ERROR;
  }
	g_Input0 = inputs[0];
  return RET_OK;
}
int CopyOutputsData(void **outputs, int num) {
  if (outputs == NULL) {
    return RET_ERROR;
  }
  if (num != 1) {
    return RET_ERROR;
  }
  memcpy(outputs[0], g_Buffer+32, 40);
  return RET_OK;
}

int GetBufferSize() {
  return 10576;
}
int SetBuffer( void *buffer) {
  if (buffer == NULL) {
    return RET_ERROR;
  }
  g_Buffer = buffer;
  return RET_OK;
}
void FreeResource() {
  g_Buffer= NULL;
  g_Input0 = NULL;
  void *allocated[] = {  };
  for (int i = 0; i < 0; ++i) {
    free(allocated[i]);
    allocated[i] = NULL;
  }
}
void Inference() {
  {
memset((int16_t *)(g_Buffer+10144), 0, 36);
const int output_shift[12] = {-9, -9, -9, -9, -9, -9, -9, -10, -10, -9, -9, -9};
const int output_mult[12] = {1354133566, 1485574432, 1737792646, 1225484872, 1221530746, 1184403831, 1344308820, 1080459089, 1432168625, 1245831715, 1804167149, 1092395059};
arm_convolve_s8((int8_t *)(g_Input0), 28, 28, 1, 1, g_Weight1, 12, 3, 3, 0, 0, 1, 1, g_Weight2, (int8_t *)(g_Buffer+0), output_shift, output_mult, 17, 128, -128, 127, 26, 26, (int16_t *)(g_Buffer+10144));
  }
  {
arm_max_pool_s8(26, 26, 13, 13, 2, 2, 2, 2, 0, 0, -128, 127, 12, (int8_t *)(g_Buffer+0), NULL, (int8_t *)(g_Buffer+8112));
  }
  {
memset((int16_t *)(g_Buffer+10144), 0, 432);
const int output_shift[12] = {-10, -10, -10, -9, -10, -10, -10, -10, -10, -9, -9, -10};
const int output_mult[12] = {2143437276, 1710269977, 1664140445, 1275314678, 2121906679, 1591651427, 1589631258, 1721320620, 1939131746, 1186858310, 1223164752, 1583392613};
arm_convolve_s8((int8_t *)(g_Buffer+8112), 13, 13, 12, 1, g_Weight3, 12, 3, 3, 0, 0, 1, 1, g_Weight4, (int8_t *)(g_Buffer+0), output_shift, output_mult, 31, -17, -128, 127, 11, 11, (int16_t *)(g_Buffer+10144));
  }
  {
arm_max_pool_s8(11, 11, 5, 5, 2, 2, 2, 2, 0, 0, -128, 127, 12, (int8_t *)(g_Buffer+0), NULL, (int8_t *)(g_Buffer+1456));
  }
  {
memcpy((int8_t *)(g_Buffer+0), (int8_t *)(g_Buffer+1456), 300);
  }
  {
arm_fully_connected_s8((int8_t *)(g_Buffer+0), g_Weight6, 300, 20, 1, -31, 0, 1379728884, -8, 11, g_Weight7, (int8_t *)(g_Buffer+304), -128, 127, NULL);
  }
  {
arm_fully_connected_s8((int8_t *)(g_Buffer+304), g_Weight8, 20, 10, 1, -11, 0, 1282256809, -8, -20, g_Weight9, (int8_t *)(g_Buffer+0), -128, 127, NULL);
  }
  {
arm_softmax_s8((int8_t *)(g_Buffer+0), 1, 10, 1152553088, 27, -15, (int8_t *)(g_Buffer+16));
  }
  {
DoDequantizeInt8ToFp32((int8_t *)(g_Buffer+16), (float *)(g_Buffer+32), 0.00390625, -128, 10);
  }
}

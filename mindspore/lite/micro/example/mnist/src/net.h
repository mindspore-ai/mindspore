
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

#include "microtensor.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
  * set input tensors
  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than one.
  * @param num, the input data's number of the model.
  **/
int net_SetInputs(const void **inputs, int num);

/**
  * get output tensor of the model 
  **/
const MicroTensorList *net_GetOutputs();

int CopyOutputsData(void **outputs, int num);

/**
  * @param weight_buffer, the address of the weight binary file
  * @param weight_size, the size of the model file in bytes
  **/
int net_Init(void *weight_buffer, int weight_size);

/**
  * get the memory space size of the inference.
  **/
int net_GetBufferSize();
/**
  * set the memory space for the inference
  **/
int net_SetBuffer(void *buffer);

/**
  * free the memory of packed weights, and set the membuf buffer and input address to NULL
  **/
void net_FreeResource();
/**
  * net inference function
  **/
void net_Inference();


#ifdef __cplusplus
}
#endif


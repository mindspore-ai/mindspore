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

#include "nnacl/base/fill_base.h"

int FillFp32(float *output, int size, float data) {
  for (int i = 0; i < size; ++i) {
    output[i] = data;
  }
  return NNACL_OK;
}

int FillInt32(int *output, int size, int data) {
  for (int i = 0; i < size; ++i) {
    output[i] = data;
  }
  return NNACL_OK;
}

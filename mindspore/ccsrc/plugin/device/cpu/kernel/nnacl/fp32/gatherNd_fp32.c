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

#include "nnacl/fp32/gatherNd_fp32.h"
#include <string.h>
#include "nnacl/errorcode.h"

int GatherNd(const void *input, void *output, const int32_t *in_offset, int area, int count, int data_type_len) {
  int i = 0;
  for (i = 0; i < count; i++) {
    (void)memcpy((int8_t *)output + area * i * data_type_len, (int8_t *)input + in_offset[i] * data_type_len,
                 (size_t)(area)*data_type_len);
  }
  return NNACL_OK;
}

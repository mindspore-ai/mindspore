/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_INTERFACES_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_INTERFACES_H_

#include "src/nnie_common.h"

namespace mindspore {
namespace nnie {
typedef struct {
  int load_model_;
  int roi_used_;
  char *model_buf_;
  int buf_size_;
  NnieRunCfg cfg_;
} NnieHandle;

typedef enum { NNIE_INT8, NNIE_UINT8, NNIE_FLOAT32 } NnieDataType;

typedef struct {
  void *data_[SVP_NNIE_MAX_INPUT_NUM];
  char *name_[SVP_NNIE_MAX_INPUT_NUM];
  int *shape_[SVP_NNIE_MAX_INPUT_NUM];
  int shape_len_[SVP_NNIE_MAX_INPUT_NUM];
  NnieDataType dtype_[SVP_NNIE_MAX_INPUT_NUM];
  int size_;
} NnieTensors;

int NnieInit(NnieHandle *h, NnieTensors *inputs);
int NnieFillData(NnieHandle *h, NnieTensors *inputs);
int NnieRun(NnieHandle *h, NnieTensors *outputs);
void NnieClose(NnieHandle *h);
}  // namespace nnie
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_PROVIDERS_NNIE_NNIE_INTERFACES_H_

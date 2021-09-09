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
#ifndef MINDSPORE_NNACL_CUSTOM_PARAMETER_H_
#define MINDSPORE_NNACL_CUSTOM_PARAMETER_H_
#include "nnacl/op_base.h"

#define MAX_STR_LEN 64
#define MAX_ATTR_NUM 8

typedef struct CustomParameter {
  OpParameter op_parameter_;
  char type[MAX_STR_LEN];
  char attr_name[MAX_ATTR_NUM][MAX_STR_LEN];
  char *attr_data[MAX_ATTR_NUM];
  int attr_num;
} CustomParameter;
#endif  // MINDSPORE_NNACL_CUSTOM_PARAMETER_H_

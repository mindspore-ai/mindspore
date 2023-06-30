/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef NNACL_BASE_CROP_BASE_H_
#define NNACL_BASE_CROP_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/crop_parameter.h"

#define CROP_OFFSET_MAX_SIZE 4

#ifdef __cplusplus
extern "C" {
#endif

int CropPadOffset(int input_dim, CropParameter *crop_para, int64_t *in_offset);

#ifdef __cplusplus
}
#endif

#endif  // NNACL_BASE_CROP_BASE_H_

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
#ifndef NNACL_DEPTH_TO_SPACE_H_
#define NNACL_DEPTH_TO_SPACE_H_

#include <string.h>
#include "nnacl/kernel/depth_to_space.h"

#ifdef __cplusplus
extern "C" {
#endif
void DepthToSpaceForNHWC(const void *input, void *output, const int *in_shape, const DepthToSpaceArgs *param);
void DepthToSpaceCRDForNHWC(const void *input, void *output, const int *in_shape, const DepthToSpaceArgs *param);
#ifdef __cplusplus
}
#endif

#endif  // NNACL_DEPTH_TO_SPACE_H_

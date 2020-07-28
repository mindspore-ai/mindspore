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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_OFFSET_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_OFFSET_UTILS_H_

#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif

inline int offset(const int *shape, const int dim0, const int dim1, const int dim2, const int dim3) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3] + dim3;
}

inline int offsetComm(const int *shape, const int dim0, const int dim1, const int dim2) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3];
}

inline int offset4d(const int *shape, const int *dims) { return offset(shape, dims[0], dims[1], dims[2], dims[3]); }
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_OFFSET_UTILS_H_


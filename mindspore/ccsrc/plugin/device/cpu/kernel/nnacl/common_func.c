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

#include "nnacl/common_func.h"

int Offset(const int *shape, const int dim0, const int dim1, const int dim2, const int dim3) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3] + dim3;
}

int64_t OffsetComm(const int *shape, const int dim0, const int dim1, const int dim2) {
  return ((dim0 * shape[1] + dim1) * shape[2] + dim2) * shape[3];
}

int Offset4d(const int *shape, const int *dims) { return Offset(shape, dims[0], dims[1], dims[2], dims[3]); }

int64_t Offset6d(const int *shape, const int *dims) {
  return ((OffsetComm(shape, dims[0], dims[1], dims[2]) + dims[3]) * shape[4] + dims[4]) * shape[5];
}

int8_t MinInt8(int8_t a, int8_t b) { return b ^ ((a ^ b) & -(a < b)); }

int8_t MaxInt8(int8_t a, int8_t b) { return a ^ ((a ^ b) & -(a < b)); }

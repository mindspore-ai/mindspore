/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_GE_RUNTIME_TENSOR_DESC_H_
#define INC_GE_RUNTIME_TENSOR_DESC_H_

namespace ge {
constexpr int64_t kMaxDimSize = 32;
constexpr int64_t DIM_SIZE2 = 2;

#pragma pack(push, 1)
struct RuntimeTensorDesc {
  uint64_t data_addr;
  int64_t data_offset_size;
  int64_t dtype;
  int64_t shape[kMaxDimSize + 1];           // shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t original_shape[kMaxDimSize + 1];  // original_shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t format;
  int64_t sub_format;
  uint8_t reserved[456];  // padding to 1024 bytes
};
#pragma pack(pop)
}  // namespace ge

#endif  // INC_GE_RUNTIME_TENSOR_DESC_H_
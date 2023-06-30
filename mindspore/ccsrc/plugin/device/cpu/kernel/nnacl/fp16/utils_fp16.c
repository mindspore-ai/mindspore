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

#include "nnacl/fp16/utils_fp16.h"
#include "nnacl/fp16/common_func_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/tensor_c_utils.h"

void *GetOrAllocFp16Data(TensorC *t, ExecEnv *env, bool cast) {
  if (t->data_type_ == kNumberTypeFloat16) {
    return t->data_;
  }
  if (t->data_type_ == kNumberTypeFloat32) {
    int ele_num = GetElementNum(t);
    void *fp16_data = env->Alloc(env->allocator_, ele_num * sizeof(float16_t));
    NNACL_MALLOC_CHECK_NULL_RETURN_NULL(fp16_data);
    if (cast) {
      Float32ToFloat16((float *)t->data_, (float16_t *)fp16_data, ele_num);
    }
    return fp16_data;
  }
  return NULL;
}

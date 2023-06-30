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

#include "nnacl/kernel/rank.h"
#include "nnacl/kernel/default_kernel_base.h"

int RankCompute(KernelBase *self) {
  size_t rank = self->in_[FIRST_INPUT]->shape_size_;
  void *output_data = self->out_[OUTPUT_INDEX]->data_;
  if (self->in_[FIRST_INPUT]->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    *(float16_t *)output_data = (float16_t)rank;
#endif
  } else {
    *(float *)output_data = (float)rank;
  }
  return NNACL_OK;
}

KernelBase *CreateRank(OpParameter *param, int data_type) {
  RankStruct *rank = (RankStruct *)malloc(sizeof(RankStruct));
  NNACL_CHECK_NULL_RETURN_NULL(rank);
  rank->base_.Release = DefaultRelease;
  rank->base_.Prepare = DefaultPrepare1In1Out;
  rank->base_.Resize = DefaultResize;
  rank->base_.Compute = RankCompute;
  return (KernelBase *)rank;
}

REG_KERNEL_CREATOR(PrimType_Rank, kNumberTypeFloat32, CreateRank)
REG_KERNEL_CREATOR(PrimType_Rank, kNumberTypeFloat16, CreateRank)

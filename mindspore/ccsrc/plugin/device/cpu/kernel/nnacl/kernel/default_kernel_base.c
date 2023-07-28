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

#include "nnacl/kernel/default_kernel_base.h"

int DefaultPrepare3In1Out(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int DefaultPrepare3In2Out(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < TWO_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int DefaultPrepare1In2Out(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < TWO_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int DefaultPrepare1In1Out(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int DefaultPrepare2In1Out(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);
  return NNACL_OK;
}

int DefaultResize(KernelBase *self) {
  NNACL_CHECK_NULL_RETURN_ERR(self->in_[FIRST_INPUT]);
  NNACL_CHECK_NULL_RETURN_ERR(self->out_[OUTPUT_INDEX]);
  return NNACL_OK;
}

int DefaultRelease(KernelBase *self) { return NNACL_OK; }

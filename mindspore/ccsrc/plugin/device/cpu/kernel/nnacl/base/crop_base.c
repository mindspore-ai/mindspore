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

#include "nnacl/base/crop_base.h"
#include "nnacl/errorcode.h"

int CropPadOffset(int input_dim, CropParameter *crop_para, int64_t *in_offset) {
  int64_t axis = crop_para->axis_;
  int offsets_size = crop_para->offset_size_;
  if (offsets_size > 1) {
    NNACL_CHECK_TRUE_RET(axis + offsets_size == input_dim, NNACL_ERR);
  }
  for (int i = 0; i < input_dim; i++) {
    int crop_offset = 0;
    if (i >= axis) {
      if (offsets_size == 1) {
        crop_offset = crop_para->offset_[0];
      } else if (offsets_size > 1) {
        if (i - axis < CROP_OFFSET_MAX_SIZE) {
          crop_offset = crop_para->offset_[i - axis];
        }
      }
    }
    in_offset[i] = crop_offset;
  }
  return NNACL_OK;
}

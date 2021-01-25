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

#ifndef LITE_MINDSPORE_LITE_C_OPS_OP_ATTR_TRANSFER_COMMON_H_
#define LITE_MINDSPORE_LITE_C_OPS_OP_ATTR_TRANSFER_COMMON_H_

#include <vector>
#include "ir/dtype/type_id.h"
#include "src/tensor.h"
#include "include/errorcode.h"
#include "src/common/common.h"
#include "src/ops/compat/compat_register.h"

namespace mindspore {
namespace lite {
schema::Tensor *AttrToTensor(void *data, int data_size, bool is_array, TypeId type_id,
                             std::vector<char *> *tensor_bufs);
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_OP_ATTR_TRANSFER_COMMON_H_

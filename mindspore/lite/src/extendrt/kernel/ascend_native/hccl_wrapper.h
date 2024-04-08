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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_HCCL_ADAPTER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_HCCL_ADAPTER_H_
#include <iostream>

namespace mindspore::ascend_native {

int HcclAllReduceSumFP32(void *send_buf, void *recv_buf, uint64_t count, void *stream);
}
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_NATIVE_ASCEND_NATIVE_IMPL_HCCL_ADAPTER_H_

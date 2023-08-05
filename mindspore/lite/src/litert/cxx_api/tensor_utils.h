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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_TENSOR_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_TENSOR_UTILS_H_

#include <limits.h>
#include <vector>
#include <memory>
#include "ir/dtype/type_id.h"
#include "src/tensor.h"
#include "include/api/types.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore {
std::vector<int32_t> MS_API TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len,
                                          bool verify_size);

size_t MS_API CalTensorDataSize(const std::vector<int64_t> &shape, enum DataType type);

Status MS_API LiteTensorToMSTensor(lite::Tensor *srcTensor, MSTensor *dstTensor, bool fromSession = true);

std::vector<MSTensor> MS_API LiteTensorsToMSTensors(const std::vector<mindspore::lite::Tensor *> &srcTensors,
                                                    bool fromSession = true);

}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_TENSOR_UTILS_H_

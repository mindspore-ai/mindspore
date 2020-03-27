/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_UTIL_DE_ERROR_H_
#define DATASET_UTIL_DE_ERROR_H_

#include <map>
#include "utils/error_code.h"

namespace mindspore {
namespace dataset {
DE_ERRORNO_DATASET(CATCH_EXCEPTION, 0, "try catch exception error");
DE_ERRORNO_DATASET(FILE_NOT_FOUND, 1, "file is not found");
DE_ERRORNO_DATASET(PARSE_FAILED, 2, "parse failed");
DE_ERRORNO_DATASET(COPY_ERROR, 3, "copy data error");
DE_ERRORNO_DATASET(BOUND_ERROR, 4, "variable overflow or lost of precision");
DE_ERRORNO_DATASET(ALLOC_FAILED, 5, "dynamic memory allocation failed");
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_UTIL_DE_ERROR_H_

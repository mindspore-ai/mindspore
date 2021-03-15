/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_VALIDATORS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_VALIDATORS_H_

#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Helper function to validate tokenizer directory parameter
Status ValidateTokenizerDirParam(const std::string &tokenizer_name, const std::string &tokenizer_file);

// Helper function to validate data type passed by user
bool IsTypeNumeric(const std::string &data_type);

// Helper function to validate data type is boolean
bool IsTypeBoolean(const std::string &data_type);

// Helper function to validate data type is string
bool IsTypeString(const std::string &data_type);
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_VALIDATORS_H_

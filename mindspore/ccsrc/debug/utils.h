/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_UTILS_H
#define MINDSPORE_UTILS_H

#include <string>

namespace mindspore {
bool CheckStoull(uint64_t *const output_digit, const std::string &input_str);

bool CheckStoul(size_t *const output_digit, const std::string &input_str);

bool CheckStoi(int64_t *const output_digit, const std::string &input_str);

void CheckStringMatch(size_t start, size_t end, std::string *matched_str, const std::string &input_str);
}  // namespace mindspore

#endif  // MINDSPORE_UTILS_H

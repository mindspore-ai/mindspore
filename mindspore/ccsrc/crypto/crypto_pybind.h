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

#ifndef MINDSPORE_CCSRC_CRYPTO_CRYPTO_PYBIND_H
#define MINDSPORE_CCSRC_CRYPTO_CRYPTO_PYBIND_H
#include "crypto/crypto.h"
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace mindspore {
namespace crypto {
py::bytes PyEncrypt(char *plain_data, const int64_t plain_len, char *key, const int32_t key_len, std::string enc_mode);
py::bytes PyDecrypt(std::string encrypt_data_path, char *key, const int32_t key_len, std::string dec_mode);
bool PyIsCipherFile(std::string file_path);
}  // namespace crypto
}  // namespace mindspore
#endif

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
#include "coder/generator/component/component.h"

namespace mindspore::lite::micro {

const char *kModelName = "net";
const char *kSession = "session";

const char *kByteType = "unsigned char *";
const char *kConstByteType = "const unsigned char *";

const char *kNameSpaceMindSpore = "namespace mindspore";
const char *kNameSpaceLite = "namespace lite";

const char *kExternCpp = R"RAW(
#ifdef __cplusplus
extern "C" {
#endif

)RAW";

const char *kEndExternCpp = R"RAW(
#ifdef __cplusplus
}
#endif

)RAW";

}  // namespace mindspore::lite::micro

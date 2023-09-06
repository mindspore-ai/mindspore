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

#include "transform/graph_ir/op_declare/spectral_op_declare.h"

namespace mindspore::transform {
// BlackmanWindow
CUST_INPUT_MAP(BlackmanWindow) = {{1, INPUT_DESC(window_length)}};
CUST_ATTR_MAP(BlackmanWindow) = {{"periodic", ATTR_DESC(periodic, AnyTraits<bool>())},
                                 {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(BlackmanWindow) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BlackmanWindow, kNameBlackmanWindow, CUST_ADPT_DESC(BlackmanWindow));

// BartlettWindow
CUST_INPUT_MAP(BartlettWindow) = {{1, INPUT_DESC(window_length)}};
CUST_ATTR_MAP(BartlettWindow) = {{"periodic", ATTR_DESC(periodic, AnyTraits<bool>())},
                                 {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};
CUST_OUTPUT_MAP(BartlettWindow) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(BartlettWindow, kNameBartlettWindow, CUST_ADPT_DESC(BartlettWindow));
}  // namespace mindspore::transform

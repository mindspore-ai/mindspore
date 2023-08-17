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

#include "./add_dsl.h"
namespace ge {
IMPLEMT_COMMON_INFERFUNC(AddDSLInferShape) { return GRAPH_SUCCESS; }

IMPLEMT_VERIFIER(AddDSL, AddDSLVerify) { return GRAPH_SUCCESS; }

COMMON_INFER_FUNC_REG(AddDSL, AddDSLInferShape);
VERIFY_FUNC_REG(AddDSL, AddDSLVerify);

}  // namespace ge

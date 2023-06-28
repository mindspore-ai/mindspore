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

#include "transform/graph_ir/op_declare/transform_fusion_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
// KVCacheMgr
INPUT_MAP(KVCacheMgr) = {{1, INPUT_DESC(past)}, {2, INPUT_DESC(cur)}, {3, INPUT_DESC(index)}};
ATTR_MAP(KVCacheMgr) = EMPTY_ATTR_MAP;
OUTPUT_MAP(KVCacheMgr) = {{0, OUTPUT_DESC(past)}};
REG_ADPT_DESC(KVCacheMgr, "KVCacheMgr", ADPT_DESC(KVCacheMgr))
}  // namespace mindspore::transform

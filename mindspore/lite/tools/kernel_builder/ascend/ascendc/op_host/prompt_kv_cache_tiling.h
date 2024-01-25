
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

#ifndef PROMPT_KV_CACHE_TILING_H
#define PROMPT_KV_CACHE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PromptKvTilingData)
TILING_DATA_FIELD_DEF(int64_t, core_num);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, ub);
TILING_DATA_FIELD_DEF(int64_t, us);
TILING_DATA_FIELD_DEF(int64_t, former_each_core_bs_num);
TILING_DATA_FIELD_DEF(int64_t, tail_each_core_bs_num);
TILING_DATA_FIELD_DEF(int64_t, split_us);
TILING_DATA_FIELD_DEF(int64_t, former_block_us);
TILING_DATA_FIELD_DEF(int64_t, tail_block_us);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PromptKvCache, PromptKvTilingData)
}  // namespace optiling
#endif  // PROMPT_KV_CACHE_TILING_H

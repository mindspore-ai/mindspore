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

#ifndef DECODER_KV_CACHE_TILING_H
#define DECODER_KV_CACHE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DecoderKvTilingData)
TILING_DATA_FIELD_DEF(int64_t, core_num);
TILING_DATA_FIELD_DEF(int64_t, b);
TILING_DATA_FIELD_DEF(int64_t, h);
TILING_DATA_FIELD_DEF(int64_t, s);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, us);
TILING_DATA_FIELD_DEF(int64_t, former_bh);
TILING_DATA_FIELD_DEF(int64_t, tail_bh);
TILING_DATA_FIELD_DEF(int64_t, f_split_bh);
TILING_DATA_FIELD_DEF(int64_t, f_f_bh);
TILING_DATA_FIELD_DEF(int64_t, f_t_bh);
TILING_DATA_FIELD_DEF(int64_t, t_split_bh);
TILING_DATA_FIELD_DEF(int64_t, t_f_bh);
TILING_DATA_FIELD_DEF(int64_t, t_t_bh);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DecoderKvCache, DecoderKvTilingData)
}  // namespace optiling
#endif  // DECODER_KV_CACHE_TILING_H

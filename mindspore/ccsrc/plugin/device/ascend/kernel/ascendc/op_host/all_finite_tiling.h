/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef ALL_FINITE_TILING_H
#define ALL_FINITE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
struct AllFiniteTilingDataLocal {
  uint32_t avg_block_count;
  uint32_t avg_block_ub_num;
  uint32_t avg_block_ub_tail;
  uint32_t avg_block_ub_loop;
  uint32_t avg_block_ub_real;
  uint32_t avg_block_ub_pad;
  uint32_t tail_block_count;
  uint32_t tail_block_ub_num;
  uint32_t tail_block_ub_tail;
  uint32_t tail_block_ub_loop;
  uint32_t tail_block_ub_real;
  uint32_t tail_block_ub_pad;
  uint32_t buffer_num;
  uint32_t block_dim;
  uint32_t in_dtype;
};

BEGIN_TILING_DATA_DEF(AllFiniteTilingData)
TILING_DATA_FIELD_DEF(uint32_t, avg_block_count);
TILING_DATA_FIELD_DEF(uint32_t, avg_block_ub_num);
TILING_DATA_FIELD_DEF(uint32_t, avg_block_ub_tail);
TILING_DATA_FIELD_DEF(uint32_t, avg_block_ub_loop);
TILING_DATA_FIELD_DEF(uint32_t, avg_block_ub_real);
TILING_DATA_FIELD_DEF(uint32_t, avg_block_ub_pad);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_count);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_ub_num);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_ub_tail);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_ub_loop);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_ub_real);
TILING_DATA_FIELD_DEF(uint32_t, tail_block_ub_pad);
TILING_DATA_FIELD_DEF(uint32_t, buffer_num);
TILING_DATA_FIELD_DEF(uint32_t, in_dtype);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AllFinite, AllFiniteTilingData);
}  // namespace optiling
#endif  // ALL_FINITE_TILING_H

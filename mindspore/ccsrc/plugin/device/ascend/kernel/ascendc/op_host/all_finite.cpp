/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "all_finite_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
#define CEIL_DIV(x, y) (((y) == 0) ? 0 : (((x) + (y)-1) / (y)))
#define UP_ROUND(in, round) ((((in) + (round)-1) / (round)) * (round))
void ElewiseTailCoreTiling(const uint32_t &aligned, const uint32_t &total_num, uint32_t &avg_block_count,
                           uint32_t &tail_block_count, uint32_t &core_num) {
  avg_block_count = CEIL_DIV(total_num, core_num);
  avg_block_count = UP_ROUND(avg_block_count, aligned);
  core_num = CEIL_DIV(total_num, avg_block_count);
  tail_block_count = total_num - (core_num - 1) * avg_block_count;
}

void ElewiseTailUbTiling(const uint32_t aligned_factor, const uint32_t max_factor, const uint32_t total_num,
                         uint32_t &ub_num, uint32_t &ub_loop, uint32_t &ub_tail, uint32_t &ub_real) {
  ub_loop = CEIL_DIV(total_num, max_factor);
  uint32_t ub_tmp_count = CEIL_DIV(total_num, ub_loop);
  ub_num = (ub_tmp_count + aligned_factor - 1) / aligned_factor * aligned_factor;
  if (ub_num > max_factor) {
    ub_num = ub_tmp_count / aligned_factor * aligned_factor;
  }
  ub_loop = CEIL_DIV(total_num, ub_num);
  ub_real = total_num - (ub_loop - 1) * ub_num;
  ub_tail = UP_ROUND(ub_real, aligned_factor);
}

void ElewiseTailTiling(AllFiniteTilingDataLocal *tiling, const uint32_t total_num, const uint32_t aligned_factor,
                       const uint32_t max_ub_factor) {
  tiling->buffer_num = 1;

  tiling->avg_block_count = 0;
  tiling->avg_block_ub_num = 0;
  tiling->avg_block_ub_tail = 0;
  tiling->avg_block_ub_loop = 0;
  tiling->avg_block_ub_real = 0;
  tiling->avg_block_ub_pad = 0;

  tiling->tail_block_count = 0;
  tiling->tail_block_ub_num = 0;
  tiling->tail_block_ub_tail = 0;
  tiling->tail_block_ub_loop = 0;
  tiling->tail_block_ub_real = 0;
  tiling->tail_block_ub_pad = 0;

  bool need_multi_core = total_num > aligned_factor;

  if (need_multi_core) {
    /* core count */
    ElewiseTailCoreTiling(aligned_factor, total_num, tiling->avg_block_count, tiling->tail_block_count,
                          tiling->block_dim);

    /* buffer num */
    bool need_double_buffer = tiling->avg_block_count > max_ub_factor;
    tiling->buffer_num = need_double_buffer ? 2 : 1;

    /* avg core ub count */
    ElewiseTailUbTiling(aligned_factor, max_ub_factor, tiling->avg_block_count, tiling->avg_block_ub_num,
                        tiling->avg_block_ub_loop, tiling->avg_block_ub_tail, tiling->avg_block_ub_real);

    /* tail core ub count */
    ElewiseTailUbTiling(aligned_factor, max_ub_factor, tiling->tail_block_count, tiling->tail_block_ub_num,
                        tiling->tail_block_ub_loop, tiling->tail_block_ub_tail, tiling->tail_block_ub_real);
  } else {
    tiling->block_dim = 1;
    tiling->tail_block_count = total_num;
    tiling->tail_block_ub_num = UP_ROUND(total_num, aligned_factor);
    tiling->tail_block_ub_tail = UP_ROUND(total_num, aligned_factor);
    tiling->tail_block_ub_real = total_num;
    tiling->tail_block_ub_loop = 1;
  }

  uint32_t pad_aligned = tiling->in_dtype == 0 ? 8 : 16;
  tiling->avg_block_ub_pad = UP_ROUND(tiling->avg_block_ub_real, pad_aligned) - tiling->avg_block_ub_real;
  tiling->tail_block_ub_pad = UP_ROUND(tiling->tail_block_ub_real, pad_aligned) - tiling->tail_block_ub_real;
  return;
}

int32_t GetAllFiniteMaxUbCount(uint32_t ele_size) {
  const uint32_t bit_block = 8;
  uint32_t ele_dsize = ele_size * bit_block;
  ele_dsize = ele_dsize + ele_size * 2 * bit_block + 1;
  uint32_t ub_buffer = 192 * 1024 - 80 - 64 * 2;
  uint32_t ub_count = ub_buffer / ele_dsize / 2 * bit_block;
  return ub_count;
}

void GetAllFiniteTiling(AllFiniteTilingDataLocal *tiling, uint32_t total) {
  uint32_t max_ub_factor = GetAllFiniteMaxUbCount(tiling->in_dtype == 0 ? 4 : 2);
  const uint32_t aligned_factor = 256;
  ElewiseTailTiling(tiling, total, aligned_factor, max_ub_factor);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  AllFiniteTilingDataLocal local_tiling;
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto coreNum = ascendcPlatform.GetCoreNum();

  local_tiling.block_dim = coreNum;
  local_tiling.in_dtype = context->GetInputTensor(0)->GetDataType();
  GetAllFiniteTiling(&local_tiling, totalLength);

  AllFiniteTilingData tiling;
  tiling.set_avg_block_count(local_tiling.avg_block_count);
  tiling.set_avg_block_ub_num(local_tiling.avg_block_ub_num);
  tiling.set_avg_block_ub_tail(local_tiling.avg_block_ub_tail);
  tiling.set_avg_block_ub_loop(local_tiling.avg_block_ub_loop);
  tiling.set_avg_block_ub_real(local_tiling.avg_block_ub_real);
  tiling.set_avg_block_ub_pad(local_tiling.avg_block_ub_pad);
  tiling.set_tail_block_count(local_tiling.tail_block_count);
  tiling.set_tail_block_ub_num(local_tiling.tail_block_ub_num);
  tiling.set_tail_block_ub_tail(local_tiling.tail_block_ub_tail);
  tiling.set_tail_block_ub_loop(local_tiling.tail_block_ub_loop);
  tiling.set_tail_block_ub_real(local_tiling.tail_block_ub_real);
  tiling.set_tail_block_ub_pad(local_tiling.tail_block_ub_pad);
  tiling.set_buffer_num(local_tiling.buffer_num);
  tiling.set_in_dtype(local_tiling.in_dtype);

  context->SetBlockDim(local_tiling.block_dim);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  auto outputShape = context->GetOutputShape(0);
  *outputShape = {1};
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AllFinite : public OpDef {
 public:
  explicit AllFinite(const char *name) : OpDef(name) {
    this->Input("gradient")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("is_finite")
      .ParamType(REQUIRED)
      .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);

    this->AICore().AddConfig("ascend910").AddConfig("ascend910b").AddConfig("ascend310p");
  }
};

OP_ADD(AllFinite);
}  // namespace ops

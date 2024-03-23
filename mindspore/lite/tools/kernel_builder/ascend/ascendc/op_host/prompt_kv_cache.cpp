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

#include "prompt_kv_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int index0 = 0;
constexpr int index1 = 1;
constexpr int index2 = 2;
constexpr int index3 = 3;
constexpr int index4 = 4;
constexpr int index5 = 5;
constexpr int index6 = 6;
constexpr size_t kDim3 = 3;
constexpr size_t kDim4 = 4;
constexpr int32_t kSize1 = 1;
constexpr int32_t kSize2 = 2;
constexpr int32_t kSize4 = 4;
constexpr size_t k910bWS = 16 * 1024 * 1024;
constexpr int64_t kBufferNum = 1;
constexpr int64_t kSmallBufferSize = 81920;
constexpr int64_t kSmallCoreNum = 4;
constexpr int64_t kMaxBlockLen = 65535;
const int64_t kDivisor = 4;
static inline int64_t CeilRound(int64_t value, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor * divisor;
}
}  // namespace

namespace optiling {
struct PKVTilingInfo {
  bool is_dim4 = true;
  bool if_copy_all = false;
  int type_size = 0;
  uint32_t aiv_num = 0;
  uint64_t ub_size = 0;
  int64_t core_num = 0;
  int64_t b = 0;
  int64_t h = 0;
  int64_t s = 0;
  int64_t d = 0;
  int64_t ub = 0;
  int64_t us = 0;
  int64_t former_each_core_bs_num = 0;
  int64_t tail_each_core_bs_num = 0;
  int64_t split_us = 0;
  int64_t former_block_us = 0;
  int64_t tail_block_us = 0;
};

static void IfCopyAll(PKVTilingInfo *info) {
  if (info->b != info->ub || info->s != info->us) {
    info->if_copy_all = false;
    return;
  }
  int64_t bs = info->b * info->h * info->s * info->d;
  int64_t aiv_num = info->aiv_num;
  if (bs <= kSmallBufferSize) {
    aiv_num = kSmallCoreNum;
  }
  int64_t former_bs = (bs + aiv_num - 1) / aiv_num;
  int64_t core_num = (bs + former_bs - 1) / former_bs;
  int64_t tail_bs = bs - (core_num - 1) * former_bs;
  if (former_bs != tail_bs) {
    info->if_copy_all = false;
  } else {
    info->if_copy_all = true;
  }
  return;
}

static ge::graphStatus SetTilingKey(gert::TilingContext *context, PKVTilingInfo *info) {
  auto past_input = context->GetInputDesc(index0);
  auto dtype = past_input->GetDataType();
  info->type_size = ge::GetSizeByDataType(dtype);
  if (info->type_size != kSize1 && info->type_size != kSize2 && info->type_size != kSize4) {
    return ge::GRAPH_FAILED;
  }

  if (info->if_copy_all) {
    context->SetTilingKey(info->type_size * 10);
  } else {
    context->SetTilingKey(info->type_size);
  }
  return ge::GRAPH_SUCCESS;
}

static void SetDimsValue(PKVTilingInfo *info, const gert::StorageShape *cache_shape,
                         const gert::StorageShape *update_shape) {
  info->is_dim4 = true;
  if (cache_shape->GetStorageShape().GetDimNum() == kDim4) {
    info->is_dim4 = true;
  } else if (cache_shape->GetStorageShape().GetDimNum() == kDim3) {
    info->is_dim4 = false;
  }

  info->b = cache_shape->GetStorageShape().GetDim(index0);
  info->ub = update_shape->GetStorageShape().GetDim(index0);
  if (info->is_dim4) {
    info->h = update_shape->GetStorageShape().GetDim(index1);
    info->s = cache_shape->GetStorageShape().GetDim(index2);
    info->us = update_shape->GetStorageShape().GetDim(index2);
    info->d = update_shape->GetStorageShape().GetDim(index3);
  } else {
    info->h = 1;
    info->s = cache_shape->GetStorageShape().GetDim(index1);
    info->us = update_shape->GetStorageShape().GetDim(index1);
    info->d = update_shape->GetStorageShape().GetDim(index2);
  }
  return;
}

static void GetSysInfo(gert::TilingContext *context, PKVTilingInfo *info) {
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  info->aiv_num = platform.GetCoreNumAiv();

  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info->ub_size);
}

static ge::graphStatus FinishTiling(gert::TilingContext *context, const PKVTilingInfo *info) {
  // set workspace for 910B
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = k910bWS;
  context->SetBlockDim(info->core_num);

  PromptKvTilingData tiling;
  tiling.set_core_num(info->core_num);
  tiling.set_b(info->b);
  tiling.set_h(info->h);
  tiling.set_s(info->s);
  tiling.set_d(info->d);
  tiling.set_ub(info->ub);
  tiling.set_us(info->us);
  tiling.set_former_each_core_bs_num(info->former_each_core_bs_num);
  tiling.set_tail_each_core_bs_num(info->tail_each_core_bs_num);
  tiling.set_split_us(info->split_us);
  tiling.set_former_block_us(info->former_block_us);
  tiling.set_tail_block_us(info->tail_block_us);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GeneralTiling(gert::TilingContext *context, PKVTilingInfo *info) {
  int64_t bs = info->ub * info->h;
  info->former_each_core_bs_num = (bs + info->aiv_num - 1) / info->aiv_num;
  info->core_num = (bs + info->former_each_core_bs_num - 1) / info->former_each_core_bs_num;
  info->tail_each_core_bs_num = bs - (info->core_num - 1) * info->former_each_core_bs_num;
  info->split_us = 1;
  int64_t block_us = info->us / info->split_us;
  int64_t remain_ub_size = info->ub_size - CeilRound(info->ub, kDivisor) * 2 * sizeof(int64_t);
  while (kBufferNum * block_us * info->d * info->type_size >= remain_ub_size || block_us > kMaxBlockLen) {
    (info->split_us)++;
    block_us = (info->us + info->split_us - 1) / info->split_us;
  }
  info->former_block_us = block_us;
  info->tail_block_us = info->us - (info->split_us - 1) * info->former_block_us;

  return FinishTiling(context, info);
}

static ge::graphStatus CopyAllTiling(gert::TilingContext *context, PKVTilingInfo *info) {
  int64_t bs = info->b * info->h * info->s * info->d;
  // set this according the test data. data_shape: (1,64,640)
  // core_num   data_copy_len    profile_time(us)
  //  1         40960            8.412
  //  2         20480            6.76
  //  4         10240            6.32
  //  8         5120             6.92
  //  16        2560             7.819
  //  40        1024             14.386

  if (bs <= kSmallBufferSize && info->aiv_num >= kSmallCoreNum) {
    info->aiv_num = kSmallCoreNum;
  }
  info->former_each_core_bs_num = (bs + info->aiv_num - 1) / info->aiv_num;
  info->core_num = (bs + info->former_each_core_bs_num - 1) / info->former_each_core_bs_num;

  info->tail_each_core_bs_num = bs - (info->core_num - 1) * info->former_each_core_bs_num;

  info->split_us = 1;
  int64_t block_bs = info->former_each_core_bs_num / info->split_us;
  while (kBufferNum * block_bs * info->type_size >= (int64_t)info->ub_size || block_bs > kMaxBlockLen) {
    (info->split_us)++;
    block_bs = (info->former_each_core_bs_num + info->split_us - 1) / info->split_us;
  }
  info->former_block_us = block_bs;
  info->tail_block_us = info->former_each_core_bs_num - (info->split_us - 1) * info->former_block_us;

  return FinishTiling(context, info);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  PKVTilingInfo info;
  GetSysInfo(context, &info);

  const gert::StorageShape *cache_shape = context->GetInputShape(index0);
  const gert::StorageShape *update_shape = context->GetInputShape(index1);
  SetDimsValue(&info, cache_shape, update_shape);

  IfCopyAll(&info);

  auto status = SetTilingKey(context, &info);
  if (status != ge::GRAPH_SUCCESS) {
    return status;
  }

  if (info.if_copy_all) {
    return CopyAllTiling(context, &info);
  }
  return GeneralTiling(context, &info);
}

static ge::graphStatus CheckSupported(const ge::Operator &op, ge::AscendString &result) {
  std::string resultStr{};
  constexpr size_t input_num = 7;
  if (op.GetInputsSize() != input_num) {
    resultStr = R"({"ret_code": "1", "reason": "input num is not 7"})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }
  if (op.GetOutputsSize() != 1) {
    resultStr = R"({"ret_code": "1", "reason": "output num is not 1"})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  for (size_t i = 0; i < input_num; ++i) {
    if (op.GetInputDesc(i).GetFormat() != ge::FORMAT_ND) {
      resultStr = R"({"ret_code": "1", "reason": "input format is not supported, only support ND."})";
      result = ge::AscendString(resultStr.c_str());
      return ge::GRAPH_FAILED;
    }
  }

  const int64_t input_dim3_num = 3;
  const int64_t input_dim4_num = 4;
  if (op.GetInputDesc(index0).GetShape().GetDimNum() != input_dim3_num &&
      op.GetInputDesc(index0).GetShape().GetDimNum() != input_dim4_num) {
    resultStr = R"({"ret_code": "1", "reason": "input dim is not supported, cache and update dim must be 4."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  if (op.GetInputDesc(index1).GetShape().GetDimNum() != input_dim3_num &&
      op.GetInputDesc(index1).GetShape().GetDimNum() != input_dim4_num) {
    resultStr = R"({"ret_code": "1", "reason": "input dim is not supported, cache and update dim must be 4."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  if (op.GetInputDesc(index2).GetShape().GetDimNum() != 1 || op.GetInputDesc(index3).GetShape().GetDimNum() != 1 ||
      op.GetInputDesc(index4).GetShape().GetDimNum() != 1 || op.GetInputDesc(index5).GetShape().GetDimNum() != 1 ||
      op.GetInputDesc(index6).GetShape().GetDimNum() != 1) {
    resultStr =
      R"({"ret_code": "1", "reason": "input dim is not supported, valid_seq_len, batch_index, seq_len_axis,
      new_max_seq_len and cur_max_seq_len dim must be 1."})";
    result = ge::AscendString(resultStr.c_str());
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x1_shape = context->GetInputShape(0);
  gert::Shape *y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}
ge::graphStatus InferPromptKvCacheDataType(gert::InferDataTypeContext *context) {
  const ge::DataType datatype = context->GetInputDataType(0);
  context->SetOutputDataType(0, datatype);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class PromptKvCache : public OpDef {
 public:
  explicit PromptKvCache(const char *name) : OpDef(name) {
    this->Input("cache")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("update")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("valid_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("batch_index")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("seq_len_axis")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("new_max_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("cur_max_seq_len")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                 ge::DT_INT64})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16,
                 ge::DT_BF16})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
               ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                           ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);
    this->SetInferDataType(ge::InferPromptKvCacheDataType);

    this->AICore()
      .SetTiling(optiling::TilingFunc)
      .AddConfig("ascend910")
      .AddConfig("ascend910b")
      .AddConfig("ascend310p")
      .SetCheckSupport(optiling::CheckSupported);
  }
};

OP_ADD(PromptKvCache);
}  // namespace ops

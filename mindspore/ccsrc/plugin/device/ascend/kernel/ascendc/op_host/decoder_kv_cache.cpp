
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

#include "decoder_kv_cache_tiling.h"
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
constexpr int64_t kBufferNum = 2;
const int64_t kDivisor = 4;
static inline int64_t CeilRound(int64_t value, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  return (value + divisor - 1) / divisor * divisor;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  auto past_input = context->GetInputDesc(index0);
  auto dtype = past_input->GetDataType();
  auto type_size = ge::GetSizeByDataType(dtype);
  if (type_size != kSize1 && type_size != kSize2 && type_size != kSize4) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(type_size);

  const gert::StorageShape *cache_shape = context->GetInputShape(index0);
  const gert::StorageShape *update_shape = context->GetInputShape(index1);

  bool is_dim4 = true;
  if (update_shape->GetStorageShape().GetDimNum() == kDim4) {
    is_dim4 = true;
  } else if (update_shape->GetStorageShape().GetDimNum() == kDim3) {
    is_dim4 = false;
  } else {
    return ge::GRAPH_PARAM_INVALID;
  }

  int64_t b = update_shape->GetStorageShape().GetDim(index0);
  int64_t h = 0;
  int64_t s = 0;
  int64_t us = 0;
  // s need get when run
  int64_t d = 0;

  if (is_dim4) {
    h = update_shape->GetStorageShape().GetDim(index1);
    us = update_shape->GetStorageShape().GetDim(index2);
    s = cache_shape->GetStorageShape().GetDim(index2);
    d = update_shape->GetStorageShape().GetDim(index3);
  } else {
    h = 1;
    us = update_shape->GetStorageShape().GetDim(index1);
    s = cache_shape->GetStorageShape().GetDim(index1);
    d = update_shape->GetStorageShape().GetDim(index2);
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  auto aiv_num = platform.GetCoreNumAiv();
  uint64_t ub_size = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  int64_t remain_ub_size = ub_size - CeilRound(b, kDivisor) * sizeof(int64_t);

  int64_t bs = b * h;
  int64_t former_bh = (bs + aiv_num - 1) / aiv_num;
  int64_t core_num = (bs + former_bh - 1) / former_bh;
  int64_t tail_bh = bs - (core_num - 1) * former_bh;

  int64_t f_split_bh = 1;
  int64_t f_f_bh = former_bh;
  while (kBufferNum * f_f_bh * us * d * type_size >= remain_ub_size) {
    f_split_bh++;
    f_f_bh = (former_bh + f_split_bh - 1) / f_split_bh;
  }
  int64_t f_t_bh = former_bh - (f_split_bh - 1) * f_f_bh;

  int64_t t_split_bh = 1;
  int64_t t_f_bh = tail_bh;
  while (kBufferNum * t_f_bh * us * d * type_size >= remain_ub_size) {
    t_split_bh++;
    t_f_bh = (tail_bh + t_split_bh - 1) / t_split_bh;
  }
  int64_t t_t_bh = tail_bh - (t_split_bh - 1) * t_f_bh;

  context->SetBlockDim(core_num);

  // set workspace for 910B
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = k910bWS;

  DecoderKvTilingData tiling;
  tiling.set_core_num(core_num);
  tiling.set_b(b);
  tiling.set_h(h);
  tiling.set_s(s);
  tiling.set_d(d);
  tiling.set_us(us);
  tiling.set_former_bh(former_bh);
  tiling.set_tail_bh(tail_bh);
  tiling.set_f_split_bh(f_split_bh);
  tiling.set_f_f_bh(f_f_bh);
  tiling.set_f_t_bh(f_t_bh);
  tiling.set_t_split_bh(t_split_bh);
  tiling.set_t_f_bh(t_f_bh);
  tiling.set_t_t_bh(t_t_bh);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
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
ge::graphStatus InferDecoderKvCacheDataType(gert::InferDataTypeContext *context) {
  const ge::DataType datatype = context->GetInputDataType(0);
  context->SetOutputDataType(0, datatype);
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DecoderKvCache : public OpDef {
 public:
  explicit DecoderKvCache(const char *name) : OpDef(name) {
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
    this->SetInferDataType(ge::InferDecoderKvCacheDataType);

    this->AICore()
      .SetTiling(optiling::TilingFunc)
      .AddConfig("ascend910")
      .AddConfig("ascend910b")
      .AddConfig("ascend310p")
      .SetCheckSupport(optiling::CheckSupported);
  }
};

OP_ADD(DecoderKvCache);
}  // namespace ops

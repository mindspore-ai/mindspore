/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
  TilingData tiling;
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
  context->SetBlockDim(BLOCK_DIM);
  tiling.set_totalLength(totalLength);
  tiling.set_tileNum(TILE_NUM);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(1);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
  const auto inputShape = context->GetInputShape(0);
  auto outputShape = context->GetOutputShape(0);
  *outputShape = *inputShape;
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AddCustom : public OpDef {
 public:
  explicit AddCustom(const char *name) : OpDef(name) {
    this->Input("x")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("y")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("z")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->SetInferShape(ge::InferShape);

    this->AICore().SetTiling(optiling::TilingFunc);

    this->AICore().AddConfig("ascend910");
    this->AICore().AddConfig("ascend310p");
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AddCustom);
}  // namespace ops

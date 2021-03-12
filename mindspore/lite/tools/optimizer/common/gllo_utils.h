/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_GLLO_UTILS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_GLLO_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include "ops/primitive_c.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "src/common/utils.h"
#include "backend/optimizer/common/pattern_engine.h"
#include "schema/inner/model_generated.h"
#include "src/param_value_lite.h"
#include "tools/converter/converter_context.h"

using PrimitiveCPtr = std::shared_ptr<mindspore::ops::PrimitiveC>;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace opt {
inline const PrimitivePtr kPrimReturn = std::make_shared<Primitive>("Return");
inline const PrimitivePtr kPrimMakeTuple = std::make_shared<Primitive>("MakeTuple");
inline const PrimitivePtr kPrimMakeTupleV2 = std::make_shared<Primitive>("make_tuple");
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("Identity");
std::vector<int> CastToInt(const ValuePtr &value);

std::vector<std::vector<int>> CastToVec2DInt(const ValuePtr &value);

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type);

bool IsRealCNodeKernel(const AnfNodePtr &node);

bool IsGraphKernel(const AnfNodePtr &node);

bool CheckInputs(const CNodePtr &cnode);

int CheckIfFuncGraphIsNull(const FuncGraphPtr &graph);

int CheckIfAnfNodeIsNull(const AnfNodePtr &node);

int CheckIfCNodeIsNull(const CNodePtr &node);

int CheckIfVarIsNull(const VarPtr &var);

int CheckInputSize(const CNodePtr &node, int size);

int CheckIfNodeIsParam(const AnfNodePtr &node);

int CheckLeastInputSize(const CNodePtr &node, int size);

ParameterPtr AddNewBiasNode(float *bias_data, const FuncGraphPtr &func_graph, int kernel_num,
                            const ParamValueLitePtr &weight_tensor);

bool IsParamNode(const BaseRef &n);

bool IsConvNode(const BaseRef &n);

bool IsPoolingNode(const BaseRef &n);

bool IsQuantNode(const BaseRef &n);

bool IsActivationNode(const BaseRef &n);

bool CheckIsAllInputsParam(const AnfNodePtr &node);

size_t GetOutputTensorNum(const AnfNodePtr &node);

bool IsMultiOutputTensors(const FuncGraphPtr &graph, const AnfNodePtr &node);

size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item);

ParamValueLitePtr GetLiteParamValue(const AnfNodePtr &node);

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index);

enum kTransFilterType {
  kKCHW2HWCK,  // 0
  kKCHW2KHWC,
  kCKHW2KHWC,
  kCKHW2HWCK,
  kKCHW2HWKC,
  kCKHW2HWKC,
  kHWCK2KCHW,
  kHWCK2CKHW,
  kHWKC2KCHW,
  kHWKC2CKHW,
  kNHWC2KCHW,  // 10
  kNHWC2CKHW,
  kNHWC2HWCK,
  kKHWC2HWCK,
  kCHWK2HWCK,
  kKHWC2CHWK,
  kCHWK2KHWC,
  kKHWC2KCHW,
  kCKHW2KCHW,
  kCHWK2KCHW,
  kKCHW2CKHW,  // 20
  kHWCK2KHWC,
  kHWKC2KHWC
};

STATUS GetFilterDim(const std::vector<int32_t> &oriDims, kTransFilterType type, int32_t *filterK, int32_t *filterC,
                    int32_t *filterH, int32_t *filterW);

STATUS SetFilterDim(const ParamValueLitePtr &tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                    int32_t filterH, int32_t filterW);

template <typename T>
static STATUS TransFilterData(const ParamValueLitePtr &tensor, kTransFilterType type, int32_t filterK, int32_t filterC,
                              int32_t filterH, int32_t filterW);

template <typename T>
static lite::STATUS TransFilterFormat(const ParamValueLitePtr &tensor, kTransFilterType type);

STATUS TransFilterFormat(const ParamValueLitePtr &tensor, schema::Format dst_format);

ParameterPtr BuildParameterNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                const ParamValueLitePtr &param_value);

ParameterPtr BuildIntValueParameterNode(const FuncGraphPtr &func_graph, const int32_t &data,
                                        const std::string &node_name);

ParameterPtr BuildIntVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                      const std::string &node_name);

ParameterPtr BuildIntVec2DParameterNode(const FuncGraphPtr &func_graph, const std::vector<std::vector<int32_t>> &data,
                                        const std::string &node_name);

ParameterPtr BuildFloatValueParameterNode(const FuncGraphPtr &func_graph, const float &data,
                                          const std::string &node_name);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_GLLO_UTILS_H_

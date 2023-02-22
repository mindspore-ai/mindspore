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
#ifndef USE_DEPRECATED_API
#define USE_DEPRECATED_API
#endif
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "ops/primitive_c.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "src/common/utils.h"
#include "backend/common/optimizer/pattern_engine.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_context.h"

using PrimitiveCPtr = std::shared_ptr<mindspore::ops::PrimitiveC>;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;
namespace mindspore {
namespace opt {
// used for common op, which corresponding value is a boolean.
constexpr auto kInferDone = "infer_done";
// used for control_flow op(while and if), which corresponding value is a boolean vec.
constexpr auto kInferFlags = "infer_flags";
inline constexpr int kInputIndexOne = 1;
inline constexpr int kInputIndexTwo = 2;
inline constexpr int kInputIndexThree = 3;
inline constexpr int kInputIndexFour = 4;
inline constexpr int kInputIndexFive = 5;
inline constexpr int kInputIndexSix = 6;
inline constexpr int kInputIndexSeven = 7;
inline constexpr size_t kInputSizeTwo = 2;
inline constexpr size_t kInputSizeThree = 3;
inline constexpr size_t kInputSizeFour = 4;
inline constexpr size_t kInputSizeFive = 5;
inline const std::vector<int> kNH2NC = {0, 3, 1, 2};
inline const std::vector<int> kNC2NH = {0, 2, 3, 1};
inline const PrimitivePtr kPrimMakeTupleV2 = std::make_shared<Primitive>("make_tuple");
inline const PrimitivePtr kPrimIdentity = std::make_shared<Primitive>("Identity");
inline const PrimitivePtr kPrimConv2DBackpropInputFusion = std::make_shared<Primitive>("Conv2DBackpropInputFusion");

using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

std::vector<int> CastToInt(const ValuePtr &value);

std::vector<std::vector<int>> CastToVec2DInt(const ValuePtr &value);

std::vector<float> CastToFloat(const ValuePtr &value);

bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type);

int GetPrimitiveType(const AnfNodePtr &node, std::string *name);

bool IsRealCNodeKernel(const AnfNodePtr &node);

bool IsGraphKernel(const AnfNodePtr &node);

bool CheckInputs(const CNodePtr &cnode);

ParameterPtr AddNewBiasNode(const float *bias_data, const FuncGraphPtr &func_graph, int kernel_num, TypeId type_id);

bool IsParamNode(const BaseRef &n);

bool IsParamOrValueNodeWithData(const BaseRef &n);

bool IsParallelSplitConvNode(const BaseRef &n);

bool IsConvNode(const BaseRef &n);

bool IsOpType(const BaseRef &n, const PrimitivePtr &prim);

bool CheckIsAllInputsParam(const AnfNodePtr &node);

size_t GetOutputTensorNum(const AnfNodePtr &node);

bool IsMultiOutputTensors(const FuncGraphPtr &graph, const AnfNodePtr &node);

AnfNodePtr GetTupleGetItemRealInput(const CNodePtr &tuple_get_item);

size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item);

tensor::TensorPtr GetTensorInfo(const AnfNodePtr &node);

AbstractBasePtr GetCNodeInputAbstract(const CNodePtr &cnode, size_t index);

STATUS TransFilterFormat(const tensor::TensorPtr &tensor, schema::Format src_format, schema::Format dst_format);

ParameterPtr BuildParameterNode(const FuncGraphPtr &func_graph, const tensor::TensorPtr &tensor_info,
                                const std::string &node_name);

ParameterPtr BuildIntValueParameterNode(const FuncGraphPtr &func_graph, const int32_t &data,
                                        const std::string &node_name, bool empty_shape = false);

ParameterPtr BuildIntVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                      const std::string &node_name);

ParameterPtr BuildIntVec2DParameterNode(const FuncGraphPtr &func_graph, const std::vector<std::vector<int32_t>> &data,
                                        const std::string &node_name);

ParameterPtr BuildFloatValueParameterNode(const FuncGraphPtr &func_graph, const float &data,
                                          const std::string &node_name, bool empty_shape = false);

ParameterPtr BuildFloatVecParameterNode(const FuncGraphPtr &func_graph, const std::vector<float> &data,
                                        const std::string &node_name);

CNodePtr GenTransposeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &perm,
                          const std::string &cnode_name);

CNodePtr GenCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input_node, const std::string &cnode_name,
                     const TypeId dst_type, const AbstractBasePtr &abstract);

CNodePtr GenGatherNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const std::vector<int> &indices,
                       const std::string &cnode_name);

CNodePtr GenTupleGetItemNode(const FuncGraphPtr &func_graph, const CNodePtr &input, size_t index);

STATUS FetchShapeFromAbstract(const abstract::AbstractBasePtr &abstract, ShapeVector *shape);

STATUS GetTensorInfoFromAbstract(tensor::TensorPtr *tensor_info, const CNodePtr &cnode, size_t index);

bool IsTrainOp(const CNodePtr &cnode);

bool IsMarkedTrainOp(const CNodePtr &cnode);

ShapeVector GetAnfNodeOutputShape(const AnfNodePtr &node, size_t output_idx);

int GetDataTypeFromAnfNode(const AnfNodePtr &anf_node, TypeId *type_id);

size_t GetOutputSize(const AnfNodePtr &anf_node);

bool IsQuantParameterNode(const PrimitivePtr &prim);

void UpdateManager(const FuncGraphPtr &func_graph);

std::pair<CNodePtr, int> GetRealCertainVarInput(const CNodePtr &cnode, size_t index);

int DetermineCertainVarInputHasInferred(const CNodePtr &cnode, size_t index, bool *infer_succ);

bool CheckAndGetCnodeIndex(const CNodePtr &cnode, size_t *index, const PrimitivePtr &primitive_type);

void PrintFuncGraph(const FuncGraphPtr &func_graph, const std::string &output_file);

std::vector<KernelWithIndex> GetNodeInputs(const AnfNodePtr &anf_node);

bool IsReduceModeMeetOutEqualIn(const PrimitivePtr &prim);

template <const PrimitivePtr *prim = nullptr>
inline bool IsSpecifiedNode(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    auto anf_node = utils::cast<AnfNodePtr>(n);
    return CheckPrimitiveType(anf_node, *prim);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_GLLO_UTILS_H_

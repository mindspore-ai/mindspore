/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_ANF_UTIL_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_ANF_UTIL_H_

#include <vector>
#include <string>
#include "mindapi/ir/tensor.h"
#include "include/errorcode.h"
#include "mindapi/base/logging.h"
#include "mindapi/ir/anf.h"
#include "mindapi/ir/func_graph.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NO_CHANGE;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;

namespace mindspore {
namespace dpico {
bool CheckPrimitiveType(const api::AnfNodePtr &node, const api::PrimitivePtr &primitive_type);
STATUS GetPrimitiveType(const api::AnfNodePtr &node, std::string *name);
STATUS GetShapeVectorFromParameter(const api::AnfNodePtr &weight, ShapeVector *shape_vector);
std::vector<int> CastToInt(const api::ValuePtr &value);
size_t GetTupleGetItemOutIndex(const api::CNodePtr &tuple_get_item);
STATUS GetOutputShapesFromCNode(const api::CNodePtr &cnode, std::vector<ShapeVector> *output_shapes);
STATUS GetInputShapeFromCNode(const api::CNodePtr &cnode, size_t input_idx, ShapeVector *shape);
STATUS FetchShapeFromAbstract(const api::AbstractBasePtr &abstract, ShapeVector *shape);
STATUS FetchTypeIdFromAbstract(const api::AbstractBasePtr &abstract, TypeId *type_id);
int GetAnfNodeOutputShape(const api::AnfNodePtr &input, ShapeVector *shape_vector);
std::string TypeIdToString(TypeId type_id);
bool CheckInputs(const api::CNodePtr &cnode);
std::string GetCustomOutputName(const api::AnfNodePtr &node);
api::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                TypeId data_type);
api::AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, TypeId data_type);
int InitParameterFromTensorInfo(const api::ParameterPtr &param_node, const api::TensorPtr &tensor_info);
api::AbstractBasePtr GetCNodeInputAbstract(const api::CNodePtr &cnode, size_t index);
api::AbstractBasePtr GetAbstractFromAnfNode(const api::AnfNodePtr &cnode);
api::ParameterPtr BuildIntValueParameterNode(const api::FuncGraphPtr &func_graph, const int32_t &data,
                                             const std::string &node_name);
api::ParameterPtr BuildIntVecParameterNode(const api::FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                           const std::string &node_name);
api::ParameterPtr BuildIntVec2DParameterNode(const api::FuncGraphPtr &func_graph,
                                             const std::vector<std::vector<int32_t>> &data,
                                             const std::string &node_name);
api::ParameterPtr BuildFloatValueParameterNode(const api::FuncGraphPtr &func_graph, const float &data,
                                               const std::string &node_name);
api::CNodePtr GenTransposeNode(const api::FuncGraphPtr &func_graph, const api::AnfNodePtr &input_node,
                               const std::vector<int> &perm, const std::string &cnode_name);
api::TensorPtr GetTensorInfo(const api::AnfNodePtr &node);
std::vector<std::vector<int>> CastToVec2DInt(const api::ValuePtr &value);
bool GetBoolAttr(const api::AnfNodePtr &node, const std::string &attr_name);
STATUS GetDataTypeAndShape(const api::ParameterPtr &param_node, TypeId *data_type, ShapeVector *shape_vector);
STATUS GetShapeVectorFromStringTensor(const api::TensorPtr &tensor_info, ShapeVector *shape_vector, size_t *offset);
inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(WARNING) << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_ANF_UTIL_H_

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

#ifndef DPICO_COMMON_ANF_UTIL_H_
#define DPICO_COMMON_ANF_UTIL_H_

#include <vector>
#include <string>
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "api/ir/func_graph.h"
#include "ops/primitive_c.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NO_CHANGE;
using mindspore::lite::RET_OK;
using mindspore::lite::STATUS;

namespace mindspore {
namespace dpico {
bool CheckPrimitiveType(const mindspore::AnfNodePtr &node, const mindspore::PrimitivePtr &primitive_type);
STATUS GetPrimitiveType(const mindspore::AnfNodePtr &node, std::string *name);
STATUS GetShapeVectorFromParameter(const mindspore::AnfNodePtr &weight, ShapeVector *shape_vector);
std::vector<int> CastToInt(const mindspore::ValuePtr &value);
size_t GetTupleGetItemOutIndex(const mindspore::CNodePtr &tuple_get_item);
STATUS GetOutputShapesFromCNode(const mindspore::CNodePtr &cnode, std::vector<ShapeVector> *output_shapes);
STATUS GetInputShapeFromCNode(const mindspore::CNodePtr &cnode, size_t input_idx, ShapeVector *shape);
STATUS FetchShapeFromAbstract(const mindspore::abstract::AbstractBasePtr &abstract, ShapeVector *shape);
STATUS FetchTypeIdFromAbstract(const mindspore::abstract::AbstractBasePtr &abstract, TypeId *type_id);
int GetAnfNodeOutputShape(const AnfNodePtr &input, ShapeVector *shape_vector);
std::string TypeIdToString(TypeId type_id);
bool CheckInputs(const mindspore::CNodePtr &cnode);
std::string GetCustomOutputName(const AnfNodePtr &node);
mindspore::tensor::TensorPtr CreateTensorInfo(const void *data, size_t data_size, const std::vector<int64_t> &shape,
                                              mindspore::TypeId data_type);
mindspore::AbstractBasePtr CreateTensorAbstract(const std::vector<int64_t> &shape, mindspore::TypeId data_type);
int InitParameterFromTensorInfo(const mindspore::ParameterPtr &param_node,
                                const mindspore::tensor::TensorPtr &tensor_info);
mindspore::abstract::AbstractBasePtr GetCNodeInputAbstract(const mindspore::CNodePtr &cnode, size_t index);
mindspore::abstract::AbstractBasePtr GetAbstractFromAnfNode(const AnfNodePtr &cnode);
mindspore::ParameterPtr BuildIntValueParameterNode(const api::FuncGraphPtr &func_graph, const int32_t &data,
                                                   const std::string &node_name);
mindspore::ParameterPtr BuildIntVecParameterNode(const api::FuncGraphPtr &func_graph, const std::vector<int32_t> &data,
                                                 const std::string &node_name);
mindspore::ParameterPtr BuildIntVec2DParameterNode(const api::FuncGraphPtr &func_graph,
                                                   const std::vector<std::vector<int32_t>> &data,
                                                   const std::string &node_name);
mindspore::ParameterPtr BuildFloatValueParameterNode(const api::FuncGraphPtr &func_graph, const float &data,
                                                     const std::string &node_name);
mindspore::CNodePtr GenTransposeNode(const api::FuncGraphPtr &func_graph, const mindspore::AnfNodePtr &input_node,
                                     const std::vector<int> &perm, const std::string &cnode_name);
mindspore::tensor::TensorPtr GetTensorInfo(const mindspore::AnfNodePtr &node);
std::vector<std::vector<int>> CastToVec2DInt(const mindspore::ValuePtr &value);
bool GetBoolAttr(const mindspore::AnfNodePtr &node, const std::string &attr_name);
STATUS GetDataTypeAndShape(const mindspore::ParameterPtr &param_node, mindspore::TypeId *data_type,
                           ShapeVector *shape_vector);
STATUS GetShapeVectorFromStringTensor(const mindspore::tensor::TensorPtr &tensor_info, ShapeVector *shape_vector,
                                      size_t *offset);
inline size_t IntToSize(int u) {
  if (u < 0) {
    MS_LOG(WARNING) << "The int value(" << u << ") is less than 0.";
    return SIZE_MAX;
  }
  return static_cast<size_t>(u);
}
}  // namespace dpico
}  // namespace mindspore

#endif  // DPICO_COMMON_ANF_UTIL_H_

/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_C_API_IR_ATTRIBUTE_H_
#define MINDSPORE_CCSRC_C_API_IR_ATTRIBUTE_H_

#include <stdbool.h>
#include <stdlib.h>
#include "c_api/base/macros.h"
#include "c_api/base/handle_types.h"
#include "c_api/base/types.h"
#include "c_api/base/status.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Create a tensor with input data buffer.
///
/// \param[in] op The Operator node handle.
/// \param[in] attr_name The attribute name.
/// \param[in] value The attribute value.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetScalarAttrFloat32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, float value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetScalarAttrBool(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, bool value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetScalarAttrInt32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int32_t value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetScalarAttrInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int64_t value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrType(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, TypeId value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
/// \param[in] vec_size number of elements in the array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrTypeArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, TypeId value[],
                                       size_t vec_size);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
/// \param[in] vec_size number of elements in the array.
/// \param[in] dataType Data type id. Currently support kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat32,
/// kNumberTypeBool.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, void *value,
                                   size_t vec_size, TypeId dataType);

/// \brief Set the attribute of the target node with the given name and value as ValueList.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
/// \param[in] vec_size Number of elements in the array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrStringArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name,
                                         const char *value[], size_t vec_size);

/// \brief Set the attribute of the target node with the given name and string value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrString(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, const char *value);

/// \brief Get the attribute of the target node with the given attribute name.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target Node.
/// \param[in] attr_name The attribute name associates with the node.
/// \param[in] error  Error code indicates whether the function executed successfully.
///
/// \return Attribute value
MIND_C_API int64_t MSOpGetScalarAttrInt64(ResMgrHandle res_mgr, ConstNodeHandle op, const char *attr_name,
                                          STATUS *error);

/// \brief Get the attribute of the target node with the given attribute name.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target Node.
/// \param[in] attr_name The attribute name associates with the node.
/// \param[in] values Array for storing the Atrrubute value.
/// \param[in] value_num Size of the given array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpGetAttrArrayInt64(ResMgrHandle res_mgr, ConstNodeHandle op, const char *attr_name,
                                        int64_t values[], size_t value_num);

/// \brief Create new Int64 attribute scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Attribute value handle.
MIND_C_API AttrHandle MSNewAttrInt64(ResMgrHandle res_mgr, const int64_t v);

/// \brief Create new flaot32 attribute scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Attribute value handle.
MIND_C_API AttrHandle MSNewAttrFloat32(ResMgrHandle res_mgr, const float v);

/// \brief Create new Bool attribute scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Attribute value handle.
MIND_C_API AttrHandle MSNewAttrBool(ResMgrHandle res_mgr, const bool v);

/// \brief Create new attribute value with array.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] value Given array.
/// \param[in] vec_size Given array size.
/// \param[in] dataType Datatype of the array.
///
/// \return Attribute value handle
MIND_C_API AttrHandle MSOpNewAttrs(ResMgrHandle res_mgr, void *value, size_t vec_size, TypeId data_type);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_IR_ATTRIBUTE_H_

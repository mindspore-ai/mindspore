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

#ifndef MINDSPORE_CCSRC_C_API_INCLUDE_ABSTRACT_H_
#define MINDSPORE_CCSRC_C_API_INCLUDE_ABSTRACT_H_

#include <stdlib.h>
#include "c_api/base/macros.h"
#include "c_api/base/types.h"
#include "c_api/base/handle_types.h"
#include "c_api/base/status.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Assign Abstract from the input node to the current node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] cur_node The current node.
/// \param[in] input_node The input node which contains the Abstract.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSAssignAbstract(ResMgrHandle res_mgr, NodeHandle cur_node, ConstNodeHandle input_node);

/// \brief Set Abstract to the node with type and shape.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The target node.
/// \param[in] type The output type of the node.
/// \param[in] shape The shape array which describe the output shape of node.
/// \param[in] shape_size The size of the shape array, i.e., the dimension of node output.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSSetAbstract(ResMgrHandle res_mgr, NodeHandle node, DataTypeC type, const int64_t shape[],
                                size_t shape_size);

/// \brief Get multiple Abstract to the node. Usually used in the case that the node has multiple outputs.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The target node.
/// \param[in] type The output type of the node.
/// \param[in] shapes A 2D array which contains multiple shape arrays response to node's multiple outputs.
/// \param[in] shape_sizes The array contains the size of all shape, i.e., the dimension of all node output.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSSetMultiAbstract(ResMgrHandle res_mgr, NodeHandle node, DataTypeC type, const int64_t **shapes,
                                     const size_t shape_sizes[], size_t abs_num);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_INCLUDE_ABSTRACT_H_

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

#ifndef MINDSPORE_CCSRC_C_API_IR_GRAPH_H_
#define MINDSPORE_CCSRC_C_API_IR_GRAPH_H_

#include <stdbool.h>
#include <stdlib.h>
#include "c_api/include/node.h"
#include "c_api/base/macros.h"
#include "c_api/base/status.h"
#include "c_api/base/types.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Creates an empty function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
///
/// \return The created function graph.
MIND_C_API GraphHandle MSFuncGraphCreate(ResMgrHandle res_mgr);

/// \brief Get the input node of the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] i Index of the input node.
///
/// \return The created function graph.
MIND_C_API NodeHandle MSFuncGraphGetInput(ResMgrHandle res_mgr, ConstGraphHandle graph, size_t i);

/// \brief Get the inputs number of the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return the inputs number of the function graph.
MIND_C_API size_t MSFuncGraphGetInputNum(ResMgrHandle res_mgr, ConstGraphHandle graph, STATUS *error);

/// \brief Get all inputs of the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] inputs The array to contain input nodes.
/// \param[in] input_num The length of the array.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSFuncGraphGetInputs(ResMgrHandle res_mgr, ConstGraphHandle graph, NodeHandle inputs[],
                                       size_t input_num);

/// \brief Set the output node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] op_node The output operator node to be set.
/// \param[in] force_new_ret If true, a new return node is always created.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSFuncGraphSetOutput(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle op_node,
                                       bool force_new_ret);

/// \brief Set the output node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] outputs The array of output operator nodes.
/// \param[in] force_new_ret If true, a new return node is always created.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSFuncGraphSetOutputs(ResMgrHandle res_mgr, GraphHandle graph, Handle const outputs[],
                                        size_t output_num, bool force_new_ret);

/// \brief Get the output node according to the index.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] i The index to get the output. If there is only one output for graph, the i should be 0;
///
/// \return The output node, nullptr if output not set.
MIND_C_API NodeHandle MSFuncGraphGetOutput(ResMgrHandle res_mgr, ConstGraphHandle graph, size_t i);

/// \brief Get the outputs number of the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return the outputs number of the function graph.
MIND_C_API size_t MSFuncGraphGetOutputNum(ResMgrHandle res_mgr, ConstGraphHandle graph, STATUS *error);

/// \brief Get all outputs of the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] outputs The array to contain input nodes.
/// \param[in] output_num The length of the array.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSFuncGraphGetOutputs(ResMgrHandle res_mgr, ConstGraphHandle graph, NodeHandle outputs[],
                                        size_t output_num);

/// \brief Replace a node in a function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] new_node The node needs to be replaced.
/// \param[in] new_node The replace node.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSFuncGraphReplace(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle old_node,
                                     ConstNodeHandle new_node);

/// \brief Compile the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
///
/// \return Error code that indicate whether the function graph compiled successfully.
MIND_C_API STATUS MSFuncGraphCompile(ResMgrHandle res_mgr, GraphHandle graph);

/// \brief Run the function graph.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] inputs The function graph (model) inputs in Tensor form.
/// \param[in] input_num The input size.
/// \param[in] outputs The function graph (model) outputs in Tensor form.
/// \param[in] outputs_num The output size.
///
/// \return Error code that indicate whether the function graph executed successfully.
MIND_C_API STATUS MSFuncGraphRun(ResMgrHandle res_mgr, GraphHandle graph, TensorHandle const inputs[], size_t input_num,
                                 TensorHandle outputs[], size_t outputs_num);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_IR_GRAPH_H_

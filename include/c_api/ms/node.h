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

#ifndef MINDSPORE_CCSRC_C_API_INCLUDE_NODE_H_
#define MINDSPORE_CCSRC_C_API_INCLUDE_NODE_H_

#include <stdbool.h>
#include <stdlib.h>
#include "include/c_api/ms/context.h"
#include "include/c_api/ms/base/macros.h"
#include "include/c_api/ms/base/status.h"
#include "include/c_api/ms/base/handle_types.h"
#include "include/c_api/ms/base/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief The struct to describe custom op's basic info. For output_dtypes and dtype_infer_func, only one of them need
/// to be specified. For output_shapes and shape_infer_func, only one of them need to be specified as well, and
/// output_dims must be given if output_shapes is specified.
typedef struct CustomOpInfo {
  const char *func_name;
  const char *func_type;
  const char *target;
  const char **input_names;
  size_t input_num;
  const char **output_names;
  size_t output_num;
  const char **attr_names;
  ValueHandle *attr_values;
  size_t attr_num;
  DTypeFormat **dtype_formats;
  size_t dtype_formats_num;
  int64_t **output_shapes;
  size_t *output_dims;
  DataTypeC *output_dtypes;
  STATUS(*dtype_infer_func)
  (const DataTypeC *input_types, size_t input_num, DataTypeC *output_types, size_t output_num);
  STATUS(*shape_infer_func)
  (int64_t **input_shapes, const size_t *input_dims, size_t input_num, int64_t **output_shapes, size_t *output_dims,
   size_t output_num);
} CustomOpInfo;

/// \brief The struct to describe the extra information for single op execution.
typedef struct DynamicOpInfo {
  const char **attr_names;   // An array of of attribute names.
  ValueHandle *attr_values;  // An array of of attributes which has same length as [attr_names].
  size_t attr_num;           // The number of attributes.
  int64_t **output_shapes;   // An array of each output's shapes. Auto infer will be denied if specified.
  size_t *output_dims;       // An array of each output's dimension. Must be specified if [output_shapes] is not NULL.
  DataTypeC *output_dtypes;  // An array of each output's dtypes. Must be specified if [output_shapes] is not NULL.
} DynamicOpInfo;

/// \brief Create a new Operator node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] op_type The primitive name.
/// \param[in] inputs An array of operator's input nodes.
/// \param[in] input_num The number of nodes in the array.
/// \param[in] attr_names An array of of attribute names.
/// \param[in] attrs An array of of attributes which has same length as [attr_names].
/// \param[in] attr_num The number of attributes.
///
/// \return The created Operator node handle
MIND_C_API NodeHandle MSNewOp(ResMgrHandle res_mgr, GraphHandle graph, const char *op_type, Handle const inputs[],
                              size_t input_num, const char *const *attr_names, ValueHandle attrs[], size_t attr_num);

/// \brief Create a tensor with input data buffer.
///
/// \param[in] op The Operator node handle.
/// \param[in] attr_name The attribute name.
/// \param[in] value The Value.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrScalarFloat32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, float value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrScalarBool(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, bool value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrScalarInt32(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int32_t value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrScalarInt64(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, int64_t value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value of the attribute.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrType(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, DataTypeC value);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
/// \param[in] vec_size number of elements in the array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrTypeArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, DataTypeC value[],
                                       size_t vec_size);

/// \brief Set the attribute of the target node with the given name and value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] op Target node.
/// \param[in] attr_name  The attribute name associates with the node.
/// \param[in] value The input value array of the attribute.
/// \param[in] vec_size number of elements in the array.
/// \param[in] data_type Data type id. Currently support kNumberTypeInt32, kNumberTypeInt64, kNumberTypeFloat32,
/// kNumberTypeBool.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSOpSetAttrArray(ResMgrHandle res_mgr, NodeHandle op, const char *attr_name, void *value,
                                   size_t vec_size, DataTypeC data_type);

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
/// \return Value
MIND_C_API int64_t MSOpGetAttrScalarInt64(ResMgrHandle res_mgr, ConstNodeHandle op, const char *attr_name,
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

/// \brief Set Operator node name.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The target node.
/// \param[in] name The op node name to be set.
///
/// \return The error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSOpSetName(ResMgrHandle res_mgr, NodeHandle node, const char *name);

/// \brief Get the name of node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The target node.
/// \param[in] str_buf The char array to contain the name string.
/// \param[in] str_len The size of the char array.
///
/// \return The error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSNodeGetName(ResMgrHandle res_mgr, ConstNodeHandle node, char str_buf[], size_t str_len);

/// \brief Pack nodes into a Tuple node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] nodes The input nodes.
/// \param[in] node_num The number of nodes in the array.
///
/// \return The created Tuple node handle.
MIND_C_API NodeHandle MSPackNodesTuple(ResMgrHandle res_mgr, GraphHandle graph, Handle const nodes[], size_t node_num);

/// \brief Get specified output branch from a multi-output Operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] op The Operator.
/// \param[in] i The index of the output branch.
///
/// \return The obtained output node.
MIND_C_API NodeHandle MSOpGetSpecOutput(ResMgrHandle res_mgr, GraphHandle graph, ConstNodeHandle op, size_t i);

/// \brief Create a Switch operator for control-flow scene.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] cond The condition of Switch which can be an Operator or a Subgraph.
/// \param[in] true_br The true branch of Switch which must be a Subgraph.
/// \param[in] false_br The false branch of Switch which must be a Subgraph.
///
/// \return The created Switch operator node.
MIND_C_API NodeHandle MSNewSwitch(ResMgrHandle res_mgr, GraphHandle graph, Handle cond, ConstGraphHandle true_br,
                                  ConstGraphHandle false_br);

/// \brief Create a While operator for control-flow scene.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] cond The condition of While which can be an Operator or a Subgraph.
/// \param[in] body_graph The loop body of While which must be a Subgraph.
/// \param[in] after_graph The graph after stepping out the While which must be a Subgraph.
///
/// \return The created While operator node.
MIND_C_API NodeHandle MSNewWhile(ResMgrHandle res_mgr, GraphHandle graph, Handle cond, GraphHandle body_graph,
                                 GraphHandle after_graph);

/// \brief Create a custom operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] inputs An array of operator's input nodes.
/// \param[in] input_num The number of nodes in the array.
/// \param[in] info An CustomOpInfo struct which describes the information of custom operator.
///
/// \return The created custom operator node.
MIND_C_API NodeHandle MSNewCustomOp(ResMgrHandle res_mgr, GraphHandle graph, Handle const inputs[], size_t input_num,
                                    CustomOpInfo info);

/// \brief Get specified input node of Operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] op The Operator.
/// \param[in] i The index of the input.
///
/// \return The obtained input node handle.
MIND_C_API NodeHandle MSOpGetInput(ResMgrHandle res_mgr, ConstNodeHandle op, size_t i);

/// \brief Get the input nodes number of Operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] op The Operator.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The input nodes number.
MIND_C_API size_t MSOpGetInputsNum(ResMgrHandle res_mgr, ConstNodeHandle op, STATUS *error);

/// \brief Get all input nodes of the Operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] op The Operator.
/// \param[in] inputs The array to contained the nodes' input.
/// \param[in] input_num The size of the input array.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSOpGetInputs(ResMgrHandle res_mgr, ConstNodeHandle op, NodeHandle inputs[], size_t input_num);

/// \brief Get dimension value of the infer shape from the given operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] op The Operator.
/// \param[in] ret A pointer to the Error code that indicates whether the functions executed successfully.
///
/// \return Dimension Value.
MIND_C_API size_t MSOpGetOutputDimension(ResMgrHandle res_mgr, ConstNodeHandle op, size_t output_index, STATUS *ret);

/// \brief Get shape vector of the infer shape from the given operator.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] op The Operator.
/// \param[in] shape_ret The provided array for shape value storage.
/// \param[in] dim Dimenesion of shape, which also indicates the numbers of elements in the shape vector.
///
/// \return Error code that indicates whether the functions executed successfully.
MIND_C_API STATUS MSOpGetOutputShape(ResMgrHandle res_mgr, ConstNodeHandle op, int64_t shape_ret[], size_t dim,
                                     size_t output_index);

/// \brief Create a subgraph node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] sub_graph The given sub function graph pointer handle.
/// \param[in] inputs An array of input nodes of the subgraph node.
/// \param[in] input_num The number of the input array.
///
/// \return The created subgraph node handle.
MIND_C_API NodeHandle MSNewFuncCallNode(ResMgrHandle res_mgr, GraphHandle graph, ConstGraphHandle sub_graph,
                                        Handle const inputs[], size_t input_num);

/// \brief Create a Placeholder node, which is usually the input of graph without data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] type The data type of the Placeholder.
/// \param[in] shape The shape array.
/// \param[in] shape_size The size of shape, i.e., the dimension of the Placeholder.
///
/// \return The created Placeholder node handle.
MIND_C_API NodeHandle MSNewPlaceholder(ResMgrHandle res_mgr, GraphHandle graph, DataTypeC type, const int64_t shape[],
                                       size_t shape_size);

/// \brief Create a Variable node of tensor without shape, which contains variable scalar data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] value The float scalar number.
///
/// \return The created Variable node handle.
MIND_C_API NodeHandle MSNewVariableScalarFloat32(ResMgrHandle res_mgr, GraphHandle graph, float value);

/// \brief Create a Variable node of tensor without shape, which contains variable scalar data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] value The int scalar number.
///
/// \return The created Variable node handle.
MIND_C_API NodeHandle MSNewVariableScalarInt32(ResMgrHandle res_mgr, GraphHandle graph, int value);

/// \brief Create a Variable node of tensor, which contains variable array data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] data The data address.
/// \param[in] shape The shape array.
/// \param[in] shape_size The size of shape, i.e., the dimension of the Variable.
/// \param[in] type The data type of the Variable.
/// \param[in] data_len The length of data.
///
/// \return The created Variable node handle.
MIND_C_API NodeHandle MSNewVariableArray(ResMgrHandle res_mgr, GraphHandle graph, void *data, DataTypeC type,
                                         const int64_t shape[], size_t shape_size, size_t data_len);

/// \brief Create a Variable node from a Tensor instance with data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] graph The given function graph pointer handle.
/// \param[in] tensor The given Tensor instance.
///
/// \return The created Variable node handle.
MIND_C_API NodeHandle MSNewVariableFromTensor(ResMgrHandle res_mgr, GraphHandle graph, ConstTensorHandle tensor);

/// \brief Get data size of a Tensor Variable.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The tensor variable node
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The data byte size.
MIND_C_API size_t MSVariableArrayGetDataSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get data from a Tensor Variable.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The tensor variable node
///
/// \return The data.
MIND_C_API void *MSVariableArrayGetData(ResMgrHandle res_mgr, ConstNodeHandle node);

/// \brief Create a Constant node of tensor, which contains constant tensor data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] data The data address.
/// \param[in] shape The shape array.
/// \param[in] shape_size The size of shape, i.e., the dimension of the Constant.
/// \param[in] type The data type of the Constant.
/// \param[in] data_len The length of data.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantArray(ResMgrHandle res_mgr, void *data, DataTypeC type, const int64_t shape[],
                                         size_t shape_size, size_t data_len);

/// \brief Create a Constant node from a Tensor instance with data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The given Tensor instance.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantFromTensor(ResMgrHandle res_mgr, TensorHandle tensor);

/// \brief Get data size of a Tensor Constant.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The tensor constant node
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The data byte size.
MIND_C_API size_t MSConstantArrayGetDataSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get data from a Tensor Constant.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The tensor constant node
///
/// \return The data.
MIND_C_API void *MSConstantArrayGetData(ResMgrHandle res_mgr, ConstNodeHandle node);

/// \brief Create Constant node of a float scalar.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] value The float32 scalar value.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantScalarFloat32(ResMgrHandle res_mgr, float value);

/// \brief Create Constant node of a bool scalar.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] value The bool scalar value.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantScalarBool(ResMgrHandle res_mgr, bool value);

/// \brief Create Constant node of a int32 scalar.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] value The int32 scalar value.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantScalarInt32(ResMgrHandle res_mgr, int value);

/// \brief Create Constant node of a int64 scalar.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] value The int64 scalar value.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantScalarInt64(ResMgrHandle res_mgr, int64_t value);

/// \brief Create Constant node of a int64 tuple.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] vec The array of int64 value.
/// \param[in] vec The size of the value.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantTupleInt64(ResMgrHandle res_mgr, const int64_t vec[], size_t size);

/// \brief Create Constant node of a string.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] str The string.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantString(ResMgrHandle res_mgr, const char *str);

/// \brief Create Constant node of a type.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] str The type.
///
/// \return The created Constant node handle.
MIND_C_API NodeHandle MSNewConstantType(ResMgrHandle res_mgr, DataTypeC type);

/// \brief Get value from the int32 scalar Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The obtained int32 value.
MIND_C_API int MSConstantScalarGetValueInt32(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get value from the float32 scalar Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The obtained float32 value.
MIND_C_API float MSConstantScalarGetValueFloat32(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get value from the bool scalar Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The obtained bool value.
MIND_C_API bool MSConstantScalarGetValueBool(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get value from the int64 scalar Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The obtained int64 value.
MIND_C_API int64_t MSConstantScalarGetValueInt64(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get value from the string Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] str_buf The char array to contain the string.
/// \param[in] str_len The size of the char array.
///
/// \return The error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSConstantStringGetValue(ResMgrHandle res_mgr, ConstNodeHandle node, char str_buf[], size_t str_len);

/// \brief Get value from the tuple Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The size of the Tuple.
MIND_C_API size_t MSConstantTupleGetSize(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Get value from the Tuple Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] vec The int64 array to contain the value.
/// \param[in] size The size of the value vector.
///
/// \return The error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSConstantTupleGetValueInt64(ResMgrHandle res_mgr, ConstNodeHandle node, int64_t vec[], size_t size);

/// \brief Get value from the Type Constant node.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] node The Constant node.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The obtained type value.
MIND_C_API DataTypeC MSConstantTypeGetValue(ResMgrHandle res_mgr, ConstNodeHandle node, STATUS *error);

/// \brief Run single op in eager way.
///
/// \param[in] op_type The primitive name.
/// \param[in] inputs An array of operator's input nodes.
/// \param[in] input_num The number of nodes in the array.
/// \param[in] outputs An array to store outputs tensor which is allocated by used.
/// \param[in] output_num The number of outputs.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSRunOp(ResMgrHandle res_mgr, const char *op_type, TensorHandle const inputs[], size_t input_num,
                          TensorHandle outputs[], size_t output_num);

/// \brief Run single op with extra information in eager way.
///
/// \param[in] op_type The primitive name.
/// \param[in] inputs An array of operator's input nodes.
/// \param[in] input_num The number of nodes in the array.
/// \param[in] outputs An array to store outputs tensor which is allocated by used.
/// \param[in] output_num The number of outputs.
/// \param[in] extra_info A struct that holds the extra information
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSRunOpWithInfo(ResMgrHandle res_mgr, const char *op_type, TensorHandle const inputs[],
                                  size_t input_num, TensorHandle outputs[], size_t output_num,
                                  DynamicOpInfo extra_info);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_INCLUDE_NODE_H_

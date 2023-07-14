/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

/*!
 * \file util.h
 * \brief
 */
#ifndef CUSTOMIZE_OP_PROTO_UTIL_UTIL_H_
#define CUSTOMIZE_OP_PROTO_UTIL_UTIL_H_

#include <memory.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>

#include "error_util.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "transfer_shape_according_to_format.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "graph/axis_type_info.h"

#include "op_log.h"

#define CHECK_KEY_IN_MAP(map, key, name, re_expr)                \
  if (map.find(key) == map.end()) {                              \
    CUBE_INNER_ERR_REPORT("", "not found %s in %s", name, #map); \
    re_expr;                                                     \
  }

#define CHECK_PTR_NULL(ptr, name, re_expr)             \
  if (ptr == nullptr) {                                \
    CUBE_INNER_ERR_REPORT("", "Get %s failed.", name); \
    re_expr;                                           \
  }

#define CHECK_FALSE(val, re_expr) \
  do {                            \
    if (!(val)) {                 \
      re_expr;                    \
    }                             \
  } while (0)

#define DYNAMIC_SHAPE_NOT_SUPPORTED(op)                                        \
  do {                                                                         \
    auto static_op_desc = OpDescUtils::GetOpDescFromOperator(op);              \
    for (size_t i = 0; i < static_op_desc->GetAllInputsSize(); ++i) {          \
      auto input_i_desc = static_op_desc->MutableInputDesc(i);                 \
      const GeShape &input_i_shape = input_i_desc->MutableShape();             \
      if (input_i_shape.IsUnknownShape() || input_i_shape.IsUnknownDimNum()) { \
        OP_LOGW(TbeGetName(op), OtherErrMsg("Not Support dynamic shape now")); \
      }                                                                        \
    }                                                                          \
  } while (0)

namespace ge {
// enum type and string type mapping
const std::map<ge::DataType, std::string> DTYPE_STR_MAP{
  {ge::DT_DOUBLE, "double"},   {ge::DT_COMPLEX64, "complex64"}, {ge::DT_COMPLEX128, "complex128"},
  {ge::DT_FLOAT16, "float16"}, {ge::DT_FLOAT, "float32"},       {ge::DT_INT8, "int8"},
  {ge::DT_INT16, "int16"},     {ge::DT_INT32, "int32"},         {ge::DT_INT64, "int64"},
  {ge::DT_UINT8, "uint8"},     {ge::DT_UINT16, "uint16"},       {ge::DT_UINT32, "uint32"},
  {ge::DT_UINT64, "uint64"},   {ge::DT_BOOL, "bool"},           {ge::DT_INT4, "int4"},
  {ge::DT_BF16, "bfloat16"}};

// define the input num of shape
const size_t INPUT_NUM0 = 0;
const size_t INPUT_NUM1 = 1;
const size_t INPUT_NUM2 = 2;
const size_t INPUT_NUM3 = 3;
const size_t INPUT_NUM4 = 4;
const size_t INPUT_NUM5 = 5;
const size_t INPUT_NUM6 = 6;
const size_t INPUT_NUM7 = 7;
const size_t INPUT_NUM8 = 8;
const size_t INPUT_NUM9 = 9;

// define the dims size of shape
const size_t DIM_SIZE0 = 0;
const size_t DIM_SIZE1 = 1;
const size_t DIM_SIZE2 = 2;
const size_t DIM_SIZE3 = 3;
const size_t DIM_SIZE4 = 4;
const size_t DIM_SIZE5 = 5;
const size_t DIM_SIZE6 = 6;
const size_t DIM_SIZE7 = 7;
const size_t DIM_SIZE8 = 8;

// define the index of shape dim
const size_t DIM_INDEX0 = 0;
const size_t DIM_INDEX1 = 1;
const size_t DIM_INDEX2 = 2;
const size_t DIM_INDEX3 = 3;
const size_t DIM_INDEX4 = 4;
const size_t DIM_INDEX5 = 5;
const size_t DIM_INDEX6 = 6;
const size_t DIM_INDEX7 = 7;
const size_t DIM_INDEX8 = 8;

/*
 * get the datatype of input
 * param[in] dataType  input datatype of enum value
 * param[in] supportList  the support range of op
 * return true :get type success
 *        false:get type failed
 */
bool GetInputDataType(const ge::DataType &data_type, const std::vector<ge::DataType> &supportList);

bool GetInputDataType(const ge::DataType &dataType, const std::vector<ge::DataType> &supportList, std::string &dType);

/* infer shape of two input and on output with broadcast
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */
bool CheckInputDataType(const Operator &op, const std::string &input_name,
                        const std::vector<ge::DataType> &support_list);

/*
 * check the datatype and shape of input
 * param[in] op  the operator
 * param[in] inputTensorMap  the map of input name and support datatype
 * param[in] paramType  the mode of input param, tensor or scalar
 * return true
 *        false
 */
bool CheckInputDtypeAndShape(const Operator &op, const std::map<std::string, std::vector<DataType>> &inputTensorMap);

/*
 * infer shape of two input and on output with broadcast
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */
bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name);

/*
 * infer shape of two input and on output with broadcast use name
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * param[in] is_dynamic  whether the shape of output is dynamic shape
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name, bool &is_dynamic);

/*
 * infer shape of two input and on output with broadcast use index
 * param[in] op  op desc supply by ge
 * param[in] input_idx_1  first input idx
 * param[in] input_idx_2  second input idx
 * param[in] output_idx  output idx
 * param[in] is_dynamic  whether the shape of output is dynamic shape
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */
bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const int64_t &input_idx_1, const int64_t &input_idx_2,
                                           const int64_t &output_idx, bool &is_dynamic);
bool InferShapeAndTypeBroadcast(Operator &op, std::vector<int64_t> input_idxs, const int64_t &output_idx,
                                bool &is_dynamic);

bool InferShapeRangeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                         const string &output_name);

bool CheckInputDataType(const Operator &op, std::string *data_type, const std::string &input_name,
                        const std::vector<ge::DataType> &supportList);

std::vector<int64_t> TwoBroadcastShape(const std::vector<int64_t> &dimsX, const std::vector<int64_t> &dimsY);

std::vector<std::pair<int64_t, int64_t>> TwoShapeAndRangeBroadcast(
  const std::vector<int64_t> &dims_out, const std::vector<std::pair<int64_t, int64_t>> &shape_range_x,
  std::vector<std::pair<int64_t, int64_t>> &shape_range_y);

/*
 * infer shape of two input and on output with broadcast,Change the original value dimVec and Vec_range.
 * param[in] op  op desc supply by ge
 * param[in] dimVec  Before a shape broadcast value
 * param[in] Vec_range   Before a range broadcast value
 * param[in] dims   The value of broadcast is currently required
 * param[in] range      The value of broadcast is currently required
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 */
bool TwoShapeAndRangeBroadcastIntegration(const Operator &op, std::vector<int64_t> &dimVec,
                                          std::vector<std::pair<int64_t, int64_t>> &Vec_range,
                                          std::vector<int64_t> dims, std::vector<std::pair<int64_t, int64_t>> range,
                                          const string &input_name1, const string &input_name2);

bool CheckTwoInputDtypeSame(const Operator &op, const string &input_name1, const string &input_name2);

bool CheckInputDtypeSame(const Operator &op, const std::vector<std::string> &input_names);

bool CheckInputsShapeDtypeSame(const Operator &op, const std::vector<std::string> &input_names);

bool GetConstValue(const ge::Operator &op, const std::string &key_name, float &attr_value);

bool GetConstValue(const ge::Operator &op, const std::string &key_name, int64_t &attr_value);

bool GetConstValue(const ge::Operator &op, const std::string &key_name, bool &attr_value);

bool GetConstValue(const ge::Operator &op, const std::string &key_name, std::vector<int32_t> &attr_value);

bool IsSliceUnknownShape(const std::vector<int64_t> &dim_vec, const int64_t &begin, const int64_t &end);

/**
 * Get int type const value from tensor data
 * @param [in] data const tensor data
 * @param [in] data_type DT_INT8, DT_INT16, DT_INT32, DT_INT64
 * @param [out] const_values const int values
 * @return true:success, false:failed.
 */
bool GetConstIntData(const Tensor &data, DataType data_type, std::vector<int64_t> &const_values);

bool GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data);
bool GetConstValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype,
                   std::vector<uint64_t> &const_data);
bool GetConstValue(const Operator &op, const GeTensorPtr &const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data);
bool GetConstValue(const Operator &op, const GeTensor *const_tensor, const DataType &dtype,
                   std::vector<int64_t> &const_data);
bool GetScalerValue(const Operator &op, const Tensor &const_tensor, const DataType &dtype, std::int64_t &const_data);
bool InferShapeAndTypeTwoInOneOutBroadcast(Operator &op, const string &input_name1, const string &input_name2,
                                           const string &output_name);

std::vector<int64_t> GetNewAxis4NewFormat(std::size_t ori_shape_len, int64_t axis, const std::string &ori_format,
                                          const std::string &new_format, bool reduce_mode = false);
std::string ToFormatString(ge::Format format);

/*
 * Check input dtype and format is supported in supportList from inputNumBeg to inputNumEnd
 * param[in] op  op desc supply by ge
 * param[in] inputNumBeg input index begin, [0, N]
 * param[in] inputNumEnd input index end need to be checked
 * param[in] supportList, support type of ge::DataType and ge::Format
 * return true: check pass
 *        false: check failed
 */
template <typename T>
bool CheckSimilarInputDtypeAndFormat(const Operator &op, std::size_t inputNumBeg, std::size_t inputNumEnd,
                                     const std::vector<T> &supportList) {
  for (std::size_t i = inputNumBeg; i < inputNumEnd; i++) {
    if (std::is_same<typename std::decay<T>::type, ge::DataType>::value) {
      ge::DataType inType = op.GetInputDesc(i).GetDataType();
      const auto &findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    } else if (std::is_same<typename std::decay<T>::type, ge::Format>::value) {
      ge::Format inType = op.GetInputDesc(i).GetFormat();
      const auto &findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    }
  }
  return true;
}

/*
 * Check input dtype and format is supported in supportList from inputNumBeg to inputNumEnd
 * param[in] op  op desc supply by ge
 * param[in] indexNeedCheck input index need to be checked
 * param[in] supportList, support type of ge::DataType and ge::Format
 * return true: check pass
 *        false: check failed
 */
template <typename T>
bool CheckSimilarInputDtypeAndFormat(const Operator &op, const std::vector<std::size_t> &indexNeedCheck,
                                     const std::vector<T> &supportList) {
  for (auto i : indexNeedCheck) {
    if (std::is_same<typename std::decay<T>::type, ge::DataType>::value) {
      ge::DataType inType = op.GetInputDesc(i).GetDataType();
      const auto &findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    } else if (std::is_same<typename std::decay<T>::type, ge::Format>::value) {
      ge::Format inType = op.GetInputDesc(i).GetFormat();
      const auto &findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    }
  }
  return true;
}

/*
 * get const attr
 * param[in] op op desc supply by ge
 * param[in] attrName list need to be get
 * param[out] attr vector
 * return true: get success
 *        false: get failed
 */
template <typename T>
bool GetConstAttr(const Operator &op, const std::vector<std::string> &attrNameList, std::vector<T> &attrVec) {
  T value;
  for (auto name : attrNameList) {
    if (op.GetAttr(name.c_str(), value) != ge::GRAPH_SUCCESS) {
      return false;
    }
    attrVec.push_back(value);
  }
  return true;
}

/*
 * get const attr list
 * param[in] op op desc supply by ge
 * param[in] attrName list need to be get
 * param[out] attr vector
 * return true: get success
 *        false: get failed
 */
template <typename T>
bool GetConstAttr(const Operator &op, const std::vector<std::string> &attrNameList,
                  std::vector<std::vector<T>> &attrListVec) {
  for (auto name : attrNameList) {
    std::vector<T> valueList;
    if (op.GetAttr(name, valueList) != ge::GRAPH_SUCCESS) {
      return false;
    }
    attrListVec.push_back(valueList);
  }
  return true;
}

std::string to_string(const vector<int64_t> &shape);
std::string to_string(const ge::Shape &shape);
std::string to_string(const ge::GeShape &shape);
std::string to_string(const vector<pair<int64_t, int64_t>> &ranges);

class DynamicShapeInfer {
 public:
  std::map<std::string, Format> map_format;
  std::map<std::string, DataType> map_dtype;
  std::map<std::string, uint32_t> inputs;
  std::map<std::string, uint32_t> outputs;
  Operator &op;
  OpDescPtr &op_desc;

  DynamicShapeInfer(Operator &op_v, OpDescPtr &opDesc_v) : op(op_v), op_desc(opDesc_v) {}
  bool CatchFormatAndShape();
  bool UpdateFormatAndShape();

  ~DynamicShapeInfer() { UpdateFormatAndShape(); }
};

#define PREPARE_DYNAMIC_SHAPE(depends_names)               \
  do {                                                     \
    auto op_desc = OpDescUtils::GetOpDescFromOperator(op); \
    if (!depends_names.empty()) {                          \
      op_desc->SetOpInferDepends(depends_names);           \
    }                                                      \
  } while (0)

bool IsEmptyTensor(const std::vector<int64_t> &dims);

bool IsUnknownRank(const Operator &op, const std::string &tensor_name, const std::string &types = "input");

bool IsUnknownRankShape(const std::vector<int64_t> &shape_vec);

/*
 * @brief: check where the shape is unknown rank
 * @param [in] input_shape: GeShape
 * @return bool: true when shape is [-2] else false
 */
bool IsUnknownRankShape(const GeShape &input_shape);

bool IsUnKnownShape(const std::vector<int64_t> &shape_vec);

bool IsUnknownVec(std::vector<int64_t> &shape_vec);

bool IsUnknown(const std::vector<int64_t> &shape_vec);

void MakeUpShapeRange(const std::vector<int64_t> &shape, std::vector<std::pair<int64_t, int64_t>> &range);
void MakeUpShapeRange(const ge::GeShape &shape, std::vector<std::pair<int64_t, int64_t>> &range);

std::string DataTypeToStringDesc(const ge::DataType &dataType);

bool OneInOneOutDynamicInfer(const Operator &op, const std::string &input_name,
                             const std::vector<std::string> &output_name_list);

/*
 * @brief: do infershape for output = input
 * @param [in] op: ge Operator
 * @param [in] input_idx: the input idx
 * @param [in] output_idx_list: the output idx list
 * @return bool: the status of infershape
 */

bool OneInOneOutDynamicInfer(const Operator &op, const int64_t &input_idx, const std::vector<int64_t> &output_idx_list);

bool TwoInOneOutDynamicInferNoBroadcast(Operator &op, const string &input1_name, const string &input2_name,
                                        const std::vector<string> &output_name_list);

void FixShapeRangeWithDims(const std::vector<int64_t> &dims, std::vector<int64_t> &shape_1,
                           std::vector<int64_t> &shape_2, std::vector<std::pair<int64_t, int64_t>> &range_1,
                           std::vector<std::pair<int64_t, int64_t>> &range_2);

bool SetScalarOutputDesc(const string &input, const string &output, OpDescPtr op_desc, GeShape &output_shape);

bool IsEmptyTensor(GeTensorDescPtr tensor_desc);

bool IsEmptyTensor(const GeShape &ge_shape);

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges);

/**
 * @brief: Sort and unique AxisTypeInfo, then serial to string
 * @param axis_type_info: Result of infer axis type for operator
 * @return string: the serial string of AxisTypeInfos
 * Result template such as:
 * string expect_info = "[{"                     \
 *       "type: 0,"                              \
 *       "relate_inputs: [{0, {0}}, {1, {0}}],"  \
 *       "relate_outputs: [{0, {0}}]"            \
 *   "}, {"                                      \
 *       "type: 0,"                              \
 *       "relate_inputs: [{0, {1}}, {1, {1}}],"  \
 *       "relate_outputs: [{0, {1}}]"            \
 *   "}]".DeleteWhitespace();
 */
std::string AxisTypeInfoToString(const std::vector<ge::AxisTypeInfo> &axis_type_info);

/**
 * @brief: Get all valid inputs desc with indices, contains: INPUT, OPTIONAL_INPUT, DYNAMIC_INPUT.
 * @param [in] op: ge operator
 * @param [out] inputs: the map of index and inputs desc
 * @return graphStatus: the status whether executed successfully
 */
ge::graphStatus GetAllValidInputsWithIndices(const Operator &op, map<uint32_t, ConstGeTensorDescPtr> &inputs);

/**
 * @brief: Get all valid outputs desc with indices, contains: OUTPUT, DYNAMIC_OUTPUT.
 * @param [in] op: ge operator
 * @param [out] outputs: the map of index and outputs desc
 * @return graphStatus: the status whether executed successfully
 */
ge::graphStatus GetAllValidOutputsWithIndices(const Operator &op, map<uint32_t, ConstGeTensorDescPtr> &outputs);

/**
 * @brief: Get all input/output shapes with indices
 * @param [in] op: ge operator
 * @param [out] input_shapes: the map of index and input shapes
 * @param [out] output_shapes: the map of index and output shapes
 * @param [out] is_unknown_dim_num: the flag whether some of input/output shape are unknown rank
 * @param [out] dim_num: the max dimension num of input/output shape
 * @return graphStatus: the status whether executed successfully
 */
ge::graphStatus GetShapesForInferAxisType(const Operator &op, map<uint32_t, vector<int64_t>> &input_shapes,
                                          map<uint32_t, vector<int64_t>> &output_shapes, bool &is_unknown_dim_num,
                                          size_t &dim_num);

class AxisTypeInfoBuilder {
 public:
  AxisTypeInfoBuilder() = default;

  AxisTypeInfoBuilder &AxisType(const AxisType axis_type) {
    axis_type_info_.SetAxisType(axis_type);
    return *this;
  }

  AxisTypeInfoBuilder &AxisTypes(std::vector<ge::AxisType> axis_types) {
    axis_type_info_.SetAxisTypes(axis_types);
    return *this;
  }

  AxisTypeInfoBuilder &AddInputCutInfo(std::pair<int64_t, std::vector<int64_t>> cut_info) {
    axis_type_info_.AddInputCutInfo(cut_info);
    return *this;
  }

  AxisTypeInfoBuilder &AddOutputCutInfo(std::pair<int64_t, std::vector<int64_t>> cut_info) {
    axis_type_info_.AddOutputCutInfo(cut_info);
    return *this;
  }

  AxisTypeInfoBuilder &AddInputValueCutInfo(std::pair<int64_t, std::vector<int64_t>> cut_info) {
    axis_type_info_.AddInputValueCutInfo(cut_info);
    return *this;
  }

  AxisTypeInfoBuilder &AddOutputValueCutInfo(std::pair<int64_t, std::vector<int64_t>> cut_info) {
    axis_type_info_.AddOutputValueCutInfo(cut_info);
    return *this;
  }

  bool IsRelateInputsEmpty() { return axis_type_info_.GetRelateInputs().empty(); }

  bool IsRelateOutputsEmpty() { return axis_type_info_.GetRelateOutputs().empty(); }

  AxisTypeInfo Build() { return axis_type_info_; }

 private:
  AxisTypeInfo axis_type_info_;
};

/*
 * @brief: infer axis type for elemwise op register
 * @param [in] op: ge operator. The parameter name should be same to declaration in macro define
 * @param [out] axis_type: result of axis type info. The parameter name should be same to declaration in macro define
 * @return graphStatus: the status whether success to infer axis type
 */
ge::graphStatus InferAxisType4ElementwiseOp(const Operator &op, vector<AxisTypeInfo> &axis_type);

/*
 * @brief: infer axis type for elemwiseBroadcast op register
 * @param [in] op: ge operator. The parameter name should be same to declaration in macro define
 * @param [out] axis_type: result of axis type info. The parameter name should be same to declaration in macro define
 * @return graphStatus: the status whether success to infer axis type
 */
ge::graphStatus InferAxisType4BroadcastOp(const Operator &op, vector<AxisTypeInfo> &axis_type);
/*
 * @brief: infer elementwise axis type helper
 * @param [in] op: ge operator. The parameter name should be same to declaration in macro define
 * @param [out] axis_type: result of axis type info. The parameter name should be same to declaration in macro define
 * @param [in] allowed_split_inputs: a list of input indices to allow for axis split.
 * @param [in] allowed_split_outputs: a list of output indices to allow for axis split.
 * @param [in] excepted_axes: a list of axis indices to except for inputs/outputs axis split.
 * @return graphStatus: the status whether success to infer axis type
 */
ge::graphStatus InferElementwiseAxisTypeHelper(const Operator &op, vector<AxisTypeInfo> &axis_type,
                                               const vector<int64_t> &allowed_split_inputs,
                                               const vector<int64_t> &allowed_split_outputs,
                                               vector<int64_t> &excepted_axes);

/*
 * @brief: infer axis type for reduce op register
 * @param [in] op: ge operator. The parameter name should be same to declaration in macro define
 * @param [in] reduce_type: reduce op of axis type info
 * @param [in] axis: the dimensions to reduce
 * @param [in] keep_dims: if true, retains reduced dimensions with length 1
 * @param [out] axis_type: result of axis type info. The parameter name should be same to declaration in macro define
 * @return graphStatus: the status whether success to infer axis type
 */
ge::graphStatus InferAxisType4ReduceOpHelper(const Operator &op, const AxisType &reduce_type,
                                             const std::vector<int64_t> &axis, const bool &keep_dims,
                                             std::vector<ge::AxisTypeInfo> &axis_type);

namespace array_ops {
bool CheckInt64MulOverflow(int64_t a, int64_t b);
int64_t CalcMaxElementsCount(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                             const GeShape &x_shape);
void GenerateWorstYShapeAndYShapeRange(int64_t y_rank, int64_t max_elements_count,
                                       std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape);
bool RepairAndCheckRange(const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                         std::vector<std::pair<int64_t, int64_t>> &value_range);
void InferShapeRangeForEmptyTensor(int64_t y_rank, int64_t max_elements_count,
                                   const std::vector<std::pair<int64_t, int64_t>> &value_range,
                                   std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape);
void UpdateDimsAndShapeRange(const Operator &op, int64_t max_elements_count,
                             const std::vector<std::pair<int64_t, int64_t>> &value_range, std::vector<int64_t> &y_dims,
                             std::vector<std::pair<int64_t, int64_t>> &y_shape_range);

void ReshapeRangeInferAllDims(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                              const GeShape &x_shape, const std::vector<std::pair<int64_t, int64_t>> &shape_value_range,
                              int64_t y_rank, std::vector<std::pair<int64_t, int64_t>> &y_shape_range,
                              GeShape &y_shape);

void ReshapeRangeInfer(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_range,
                       std::vector<std::pair<int64_t, int64_t>> &y_range, GeShape &output_shape);
void FixRangeMaxToInt32max(GeShape &shape, std::vector<std::pair<int64_t, int64_t>> &shape_range);
}  // namespace array_ops
}  // namespace ge

// other value
const int64_t INPUT_NEGATIVE_NUM2 = -2;
const int32_t DIM_VALUE0 = 0;
const int32_t DIM_VALUE1 = 1;
const int32_t DIM_VALUE2 = 2;
const int32_t DIM_VALUE3 = 3;
const int32_t DIM_VALUE4 = 4;
const int32_t DIM_VALUE5 = 5;
const int32_t DIM_VALUE6 = 6;
const int32_t DIM_VALUE7 = 7;
const int32_t DIM_VALUE8 = 8;
const double HALF = 0.5;

const float F_HALF = 0.5f;
const float F_ONE_HALF = 1.5f;
const float F_NUM_VALUE0 = 0.0F;
const float F_NUM_VALUE1 = 1.0f;
const float F_NUM_VALUE2 = 2.0f;
const float F_NUM_VALUE3 = 3.0f;
const float F_NUM_VALUE4 = 4.0f;
const float F_NUM_VALUE5 = 5.0f;
const float F_NUM_VALUE6 = 6.0f;
const float F_NUM_VALUE7 = 7.0f;
const float F_NUM_VALUE8 = 8.0f;

const int32_t NUM_VALUE0 = 0;
const int32_t NUM_VALUE1 = 1;
const int32_t NUM_VALUE2 = 2;
const int32_t NUM_VALUE3 = 3;
const int32_t NUM_VALUE4 = 4;
const int32_t NUM_VALUE5 = 5;
const int32_t NUM_VALUE6 = 6;
const int32_t NUM_VALUE7 = 7;
const int32_t NUM_VALUE8 = 8;
const int32_t NUM_VALUE32 = 32;

const int32_t INDEX_VALUE0 = 0;
const int32_t INDEX_VALUE1 = 1;
const int32_t INDEX_VALUE2 = 2;
const int32_t INDEX_VALUE3 = 3;
const int32_t INDEX_VALUE4 = 4;
const int32_t INDEX_VALUE5 = 5;
const int32_t INDEX_VALUE6 = 6;
const int32_t INDEX_VALUE7 = 7;
const int32_t INDEX_VALUE8 = 8;

template <typename T>
inline std::string VectorToString(const std::vector<T> &values) {
  std::stringstream ss;
  for (auto iter = values.begin(); iter != values.end(); ++iter) {
    ss << *iter;
    if (iter != values.end() - 1) {
      ss << ", ";
    }
  }
  return ss.str();
}
#endif  // CUSTOMIZE_OP_PROTO_UTIL_UTIL_H_

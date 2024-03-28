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

#include "op_proto/inc/elewise_calculation_ops.h"
#include "op_proto/inc/nonlinear_fuc_ops.h"
#include "custom_op_proto/cust_math_ops.h"
#include "custom_op_proto/cust_elewise_calculation_ops.h"

#include <string>
#include <vector>
#include "utils/op_attr.h"
#include "utils/op_log.h"
#include "utils/op_const.h"
#include "utils/util.h"
#include "utils/error_util.h"
#include "utils/reduce_infer_util.h"

namespace ge {
// ----------------------------------OneInOneOutCommonInfer-----------------------------
CUST_ONE_IN_ONE_OUT_INFER(BesselI0, x, y);
ONE_IN_ONE_OUT_INFER(Cos, x, y);
ONE_IN_ONE_OUT_INFER(Expm1, x, y);
ONE_IN_ONE_OUT_INFER(Exp, x, y);
ONE_IN_ONE_OUT_INFER(Log1p, x, y);
ONE_IN_ONE_OUT_INFER(Log, x, y);
ONE_IN_ONE_OUT_INFER(Tanh, x, y);
ONE_IN_ONE_OUT_INFER(Sin, x, y);
ONE_IN_ONE_OUT_INFER(Reciprocal, x, y);
ONE_IN_ONE_OUT_INFER(Sign, x, y);
// ----------------------------------OneInOneOutCommonInfer END-----------------------------

// ----------------------------------TowInOneOutCommonInfer-----------------------------
TWO_IN_ONE_OUT_INFER(Div, x1, x2, y);
TWO_IN_ONE_OUT_INFER(DivNoNan, x1, x2, y);
CUST_TWO_IN_ONE_OUT_INFER(Gcd, x1, x2, y);
CUST_TWO_IN_ONE_OUT_INFER(Heaviside, x, values, y);
CUST_TWO_IN_ONE_OUT_INFER(Hypot, x1, x2, y);
CUST_TWO_IN_ONE_OUT_INFER(Lcm, x1, x2, y);
CUST_TWO_IN_ONE_OUT_INFER(LogicalXor, x, y, output);
// ----------------------------------TowInOneOutCommonInfer END-----------------------------

// --------------AcosGrad----------------
IMPLEMT_VERIFIER(AcosGrad, AcosGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AcosGrad, AcosGradVerify);
TWO_IN_ONE_OUT_INFER(AcosGrad, y, dy, z);
// ------------AcosGrad END----------------

// ----------------AcoshGrad-------------------
IMPLEMT_VERIFIER(AcoshGrad, AcoshGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AcoshGrad, AcoshGradVerify);

IMPLEMT_COMMON_INFERFUNC(AcoshGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AcoshGrad, AcoshGradInferShape);
// --------------AcoshGrad END-----------------

// ----------------AsinGrad---------------
IMPLEMT_VERIFIER(AsinGrad, AsinGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AsinGrad, AsinGradVerify);

IMPLEMT_COMMON_INFERFUNC(AsinGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(AsinGrad, AsinGradInferShape);
// --------------AsinGrad END-------------

// ----------------AsinhGrad-------------------
IMPLEMT_VERIFIER(AsinhGrad, AsinhGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(AsinhGrad, AsinhGradVerify);
IMPLEMT_COMMON_INFERFUNC(AsinhGradInferShape) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "y", "dy", "z", is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AsinhGrad, AsinhGradInferShape);
// --------------AsinhGrad END-----------------

// ----------------AddN-------------------
int64_t GetAddNConstValue(const ge::Operator &op) {
  int64_t tensor_num;
  if (ge::GRAPH_SUCCESS != op.GetAttr("N", tensor_num)) {
    std::string err_msg = GetInputInvalidErrMsg("N");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  return tensor_num;
}

int64_t AddNInferClassify(ge::Operator &op, int64_t &tensor_num) {
  const int64_t infer_condition_one_one = 11;
  const int64_t infer_condition_one_two = 12;
  const int64_t infer_condition_two = 2;
  const int64_t infer_condition_three = 3;

  int64_t empty_num = 0;
  int64_t static_num = 0;
  int64_t dynamic_shape_num = 0;
  int64_t dynamic_dim_num = 0;

  for (int64_t i = 0; i < tensor_num; i++) {
    vector<int64_t> tempVector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
    if (tempVector.empty()) {
      empty_num++;
    } else if (std::find(tempVector.begin(), tempVector.end(), ge::UNKNOWN_DIM) != tempVector.end()) {
      dynamic_shape_num++;
    } else if (std::find(tempVector.begin(), tempVector.end(), ge::UNKNOWN_DIM_NUM) != tempVector.end()) {
      dynamic_dim_num++;
    } else {
      static_num++;
    }
  }
  if (tensor_num == empty_num + dynamic_dim_num) {
    if (tensor_num == empty_num) {
      return infer_condition_one_one;
    } else {
      return infer_condition_one_two;
    }
  } else if (tensor_num == static_num || tensor_num == empty_num + static_num ||
             tensor_num == static_num + dynamic_dim_num || tensor_num == empty_num + static_num + dynamic_dim_num) {
    return infer_condition_two;
  } else {
    return infer_condition_three;
  }
}

IMPLEMT_COMMON_INFERFUNC(AddNInferShape) {
  /*
  add_n has four type inputs:
  1.empty 2.static shape 3.-1 4.-2
  The combinations bring 15 scenes, and the 15 scenes can be classify into 4 categories:
  1.input with no range and output no need range, and it can be divided half:
    1.1 all input is empty
    1.2 input only contains empty and -2 shape
  2.input contains static shape and with no -1 shape
  3.input contains -1 shape
  */
  int64_t tensor_num = GetAddNConstValue(op);
  int64_t infer_classify = AddNInferClassify(op, tensor_num);
  // condition 1: all input shape is empty
  if (infer_classify == 11) {
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetShape(Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
    // condition 2: all input is -2 or only empty and -2
  } else if (infer_classify == 12) {
    std::vector<int64_t> shape_vector = {-2};
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetShape(Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    (void)op.UpdateOutputDesc("y", y_desc);
    // condition 3: contains static shape and no -1 shape
  } else if (infer_classify == 2) {
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    std::vector<int64_t> shape_vector = op.GetDynamicInputDesc("x", 0).GetShape().GetDims();
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (!shape_vector.empty() && !IsUnknownRankShape(shape_vector)) {
        shape_vector = temp_vector;
        break;
      }
    }
    TensorDesc y_desc = op.GetOutputDescByName("y");
    y_desc.SetShape(ge::Shape(shape_vector));
    y_desc.SetDataType(x_dtype);
    std::vector<std::pair<int64_t, int64_t>> out_range;
    MakeUpShapeRange(shape_vector, out_range);
    y_desc.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("y", y_desc);
    // condition 4: contains -1 shape, range need to choose the intersection
  } else {
    Shape out_shape = op.GetDynamicInputDesc("x", 0).GetShape();
    DataType x_dtype = op.GetDynamicInputDesc("x", 0).GetDataType();
    std::vector<int64_t> out_vector;
    std::vector<std::pair<int64_t, int64_t>> out_range;
    // Init the output shape and range
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (!temp_vector.empty() && !IsUnknownRankShape(temp_vector)) {
        out_vector = temp_vector;
        op.GetDynamicInputDesc("x", i).GetShapeRange(out_range);
        MakeUpShapeRange(out_vector, out_range);
        break;
      }
    }
    // compute the shape dims and range intersection
    for (int64_t i = 0; i < tensor_num; i++) {
      std::vector<int64_t> temp_vector = op.GetDynamicInputDesc("x", i).GetShape().GetDims();
      if (temp_vector.empty() || IsUnknownRankShape(temp_vector)) {
        continue;
      }
      std::vector<std::pair<int64_t, int64_t>> temp_range;
      op.GetDynamicInputDesc("x", i).GetShapeRange(temp_range);
      MakeUpShapeRange(temp_vector, temp_range);
      for (size_t j = 0; j < temp_vector.size(); j++) {
        // two condition: const == const; const > -1
        if (temp_vector[j] >= out_vector[j]) {
          out_vector[j] = temp_vector[j];
          // update range: left choose the max value
          if (temp_range[j].first >= out_range[j].first) {
            out_range[j].first = temp_range[j].first;
          }
          // update range: right choose the miner value but when it was > 0
          if ((temp_range[j].second <= out_range[j].second && temp_range[j].second > 0) ||
              (out_range[j].second == -1 && temp_range[j].second != -1)) {
            out_range[j].second = temp_range[j].second;
          }
        }
      }
    }
    TensorDesc y_desc = op.GetOutputDescByName("y");
    out_shape = Shape(out_vector);
    y_desc.SetShape(out_shape);
    y_desc.SetDataType(x_dtype);
    y_desc.SetShapeRange(out_range);
    (void)op.UpdateOutputDesc("y", y_desc);
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(AddN, AddNInferShape);
// ----------------AddN END-------------------

// --------------------------------BiasAdd-------------------------------------
IMPLEMT_VERIFIER(BiasAdd, BiasAddVerify) {
  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_FAILED) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NDHWC" && data_format != "NCDHW") {
    string expected_format_list = ConcatString("NHWC, NCHW, NDHWC, NCDHW");
    std::string err_msg = GetInputFormatNotSupportErrMsg(TbeGetName(op).c_str(), expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

ONE_IN_ONE_OUT_INFER(BiasAdd, x, y);
VERIFY_FUNC_REG(BiasAdd, BiasAddVerify);
// ----------------------------------BiasAdd END-----------------------------

// --------------MulNoNan--------------
IMPLEMT_VERIFIER(MulNoNan, MulNoNanVerify) {
  DataType input_type_x1 = op.GetInputDescByName("x1").GetDataType();
  DataType input_type_x2 = op.GetInputDescByName("x2").GetDataType();
  if (input_type_x1 != input_type_x2) {
    string err_msg1 =
      ConcatString("the dtype of input_type_x1 and input_type_x2 must be same! input_type_x1:", input_type_x1,
                   ", input_type_x2:", input_type_x2);
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
VERIFY_FUNC_REG(MulNoNan, MulNoNanVerify);

IMPLEMT_COMMON_INFERFUNC(MulNoNanInferShape) {
  bool is_dynamic_output = true;
  if (InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y", is_dynamic_output)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(MulNoNan, MulNoNanInferShape);
// ------------MulNoNan END--------------

// -------------------LessEqual---------------------
IMPLEMT_VERIFIER(LessEqual, LessEqualVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(LessEqualInferShape) {
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
    return GRAPH_FAILED;
  }
  auto y_desc = op.GetOutputDesc("y");
  auto vec_y = y_desc.GetShape().GetDims();
  if (IsUnknownRankShape(vec_y) || IsUnknownVec(vec_y)) {
    if (!InferShapeRangeTwoInOneOutBroadcast(op, "x1", "x2", "y")) {
      return GRAPH_FAILED;
    }
  }

  y_desc.SetDataType(DT_BOOL);
  op.UpdateOutputDesc("y", y_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(LessEqual, LessEqualInferShape);
VERIFY_FUNC_REG(LessEqual, LessEqualVerify);
// --------------------LessEqual END-----------------------

// --------------------Mul-----------------------
IMPLEMT_VERIFIER(Mul, MulVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

TWO_IN_ONE_OUT_INFER(Mul, x1, x2, y);
VERIFY_FUNC_REG(Mul, MulVerify);
// --------------------Mul END-----------------------

// -------------------FloorDiv-----------------------
IMPLEMT_VERIFIER(FloorDiv, FloorDivVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

TWO_IN_ONE_OUT_INFER(FloorDiv, x1, x2, y);
VERIFY_FUNC_REG(FloorDiv, FloorDivVerify);
// ----------------FloorDiv END------------------------

// ----------------Sinc-------------------
IMPLEMT_COMMON_INFERFUNC(SincInferShape) {
  auto input_desc = op.GetInputDescByName("x");
  auto output_desc = op.GetOutputDescByName("y");

  auto input_dtype = input_desc.GetDataType();
  auto input_shape = input_desc.GetShape();

  std::vector<DataType> int_and_bool_types{DT_INT8,   DT_UINT8, DT_INT16,  DT_UINT16, DT_INT32,
                                           DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL};
  auto output_dtype = input_dtype;
  if (std::find(int_and_bool_types.begin(), int_and_bool_types.end(), input_dtype) != int_and_bool_types.end()) {
    output_dtype = DT_FLOAT;
  }

  output_desc.SetShape(input_shape);
  output_desc.SetDataType(output_dtype);

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Sinc, SincInferShape);
// ----------------Sinc END-------------------

// ----------------SqrtGrad Op Begin-----------------
IMPLEMT_VERIFIER(SqrtGrad, SqrtGradVerify) {
  if (!CheckTwoInputDtypeSame(op, "y", "dy")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SqrtGradInferShape) {
  Shape shape_x = op.GetInputDescByName("y").GetShape();
  DataType input_dtype = op.GetInputDescByName("y").GetDataType();
  TensorDesc tensordesc_output = op.GetOutputDescByName("z");
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op.GetInputDescByName("y").GetShapeRange(shape_range_x);
  tensordesc_output.SetShape(shape_x);
  tensordesc_output.SetDataType(input_dtype);
  tensordesc_output.SetShapeRange(shape_range_x);
  if (op.UpdateOutputDesc("z", tensordesc_output) != GRAPH_SUCCESS) {
    std::string err_msg = UpdateParamErrMsg("z");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SqrtGrad, SqrtGradInferShape);
VERIFY_FUNC_REG(SqrtGrad, SqrtGradVerify);
// ----------------SqrtGrad Op End-------------------
}  // namespace ge

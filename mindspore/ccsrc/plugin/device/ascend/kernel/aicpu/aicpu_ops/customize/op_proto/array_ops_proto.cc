/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
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

#include "custom_op_proto/cust_array_ops.h"
#include "inc/ops/array_ops.h"
#include "register/op_impl_registry.h"
#include "utils/common_shape_fns.h"
#include "utils/util.h"

namespace ge {
// ----------------Expand Begin-------------------
template <typename T>
static bool ExpandCalDim(const Tensor &data, std::vector<int64_t> &vec_dim, std::vector<int64_t> &x_dims,
                         std::vector<std::pair<int64_t, int64_t>> &range_vector) {
  int64_t len_x = x_dims.size();
  int64_t len_shape = data.GetSize() / sizeof(T);
  int64_t diff = abs(len_x - len_shape);
  const char *op_name = "Expand";

  std::string xShape = to_string(x_dims);
  OP_LOGD(op_name, "Get shape of [expand's x] %s", xShape.c_str());

  const T *pdata = reinterpret_cast<const T *>(data.GetData());
  std::vector<int64_t> shape_dims;
  for (int64_t i = 0; i < len_shape; i++) {
    T dim = pdata[i];
    shape_dims.push_back(dim);
  }
  std::string shapeVal = to_string(shape_dims);
  OP_LOGD(op_name, "Get constValue val of [expand's shape] %s", shapeVal.c_str());

  const bool is_shape_less = (len_shape < len_x);

  for (int64_t i = 0; i < diff; i++) {
    T dim = 0;
    if (is_shape_less) {
      dim = x_dims[i];
    } else {
      dim = pdata[i];
    }
    if (dim == -1) {
      range_vector.push_back(std::make_pair(1, -1));
    } else {
      range_vector.push_back(std::make_pair(dim, dim));
    }
    vec_dim.push_back(dim);
  }

  int64_t upb = len_shape;
  if (is_shape_less) {
    upb = len_x;
  }
  for (int64_t i = diff; i < upb; i++) {
    int64_t idx = i - diff;
    T dim = 0;
    if (is_shape_less) {
      idx = i;
      dim = pdata[i - diff];
    } else {
      dim = pdata[i];
    }
    if (dim == -1 || x_dims[idx] == -1) {
      vec_dim.push_back(-1);
      range_vector.push_back(std::make_pair(1, -1));
      continue;
    }
    if (dim == 0) {
      vec_dim.push_back(0);
      range_vector.push_back(std::make_pair(0, 0));
      continue;
    }
    if ((x_dims[idx] != dim) && (x_dims[idx] != 1) && (dim != 1)) {
      return false;
    }
    if (x_dims[idx] > dim) {
      vec_dim.push_back(x_dims[idx]);
      range_vector.push_back(std::make_pair(x_dims[idx], x_dims[idx]));
    } else {
      vec_dim.push_back(dim);
      range_vector.push_back(std::make_pair(dim, dim));
    }
  }

  return true;
}

IMPLEMT_INFERFUNC(Expand, ExpandInferShape) {
  const char *op_name = "Expand";
  OP_LOGD(op_name, "ExpandInferShape start.");
  const vector<string> const_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(const_names);
  TensorDesc tensordesc_input = op.GetInputDescByName("x");
  Shape x_shape = tensordesc_input.GetShape();
  std::vector<int64_t> x_dims = x_shape.GetDims();
  DataType x_dtype = tensordesc_input.GetDataType();

  Tensor data;
  std::vector<int64_t> vec_dim;

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  tensordesc_output.SetDataType(x_dtype);

  TensorDesc tensordesc_shape = op.GetInputDescByName("shape");
  size_t dim_num = tensordesc_shape.GetShape().GetDimNum();
  std::vector<int64_t> empty_dim_vec = tensordesc_shape.GetShape().GetDims();
  for (size_t i = 0; i < dim_num; i++) {
    if (empty_dim_vec[i] == 0) {
      tensordesc_output.SetShape(ge::Shape(empty_dim_vec));
      return op.UpdateOutputDesc("y", tensordesc_output);
    }
  }

  std::vector<std::pair<int64_t, int64_t>> range_vector;

  if (op.GetInputConstData("shape", data) != GRAPH_SUCCESS) {
    OP_LOGD(op_name, "Get constValue failed of [shape]");
    vector<int64_t> shape_dims = tensordesc_shape.GetShape().GetDims();
    size_t dim_num = shape_dims.size();

    if (dim_num > 1) {
      OP_LOGE(op_name, "The dim numbers of shape [%zu] are more than one.", dim_num);
      return GRAPH_FAILED;
    }
    int64_t max_len = x_dims.size();
    if (shape_dims[0] > max_len) {
      max_len = shape_dims[0];
    }
    for (int64_t item = 0; item < max_len; ++item) {
      vec_dim.push_back(-1);
      range_vector.push_back(std::make_pair(1, -1));
    }
  } else {
    OP_LOGD(op_name, "Get constValue succeeded of [shape]");
    vector<int64_t> shape_dims = tensordesc_shape.GetShape().GetDims();
    if (shape_dims.size() > 1) {
      OP_LOGE(op_name, "The dim numbers of shape [%zu] are more than one.", shape_dims.size());
      return GRAPH_FAILED;
    }
    DataType data_type = tensordesc_shape.GetDataType();
    if (data_type == DT_INT32) {
      if (!ExpandCalDim<int32_t>(data, vec_dim, x_dims, range_vector)) {
        OP_LOGE(op_name, "Data shape are not compatible!");
        return GRAPH_FAILED;
      }
    } else if (data_type == DT_INT64) {
      if (!ExpandCalDim<int64_t>(data, vec_dim, x_dims, range_vector)) {
        OP_LOGE(op_name, "Data shape are not compatible!");
        return GRAPH_FAILED;
      }
    } else if (data_type == DT_INT16) {
      if (!ExpandCalDim<int16_t>(data, vec_dim, x_dims, range_vector)) {
        OP_LOGE(op_name, "Data shape are not compatible!");
        return GRAPH_FAILED;
      }
    } else {
      OP_LOGE(op_name, "Data type not supported!");
      return GRAPH_PARAM_INVALID;
    }
  }
  tensordesc_output.SetShape(ge::Shape(vec_dim));
  tensordesc_output.SetShapeRange(range_vector);
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  OP_LOGD(op_name, "ExpandInferShape finish.");

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(Expand, ExpandInferShape);
// ----------------Expand END---------------------

// ---------------SliceGrad-------------------
CUST_IMPLEMT_INFERFUNC(SliceGrad, SliceGradInfer) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  if (op.UpdateOutputDesc("dx", x_desc) != GRAPH_SUCCESS) {
    OP_LOGE("SliceGrad", "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(SliceGrad, SliceGradInfer);
// ---------------SliceGrad End---------------

// ---------------MaskedSelectGrad-------------------
CUST_IMPLEMT_INFERFUNC(MaskedSelectGrad, MaskedSelectGradInfer) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  if (op.UpdateOutputDesc("dx", x_desc) != GRAPH_SUCCESS) {
    OP_LOGE("MaskedSelectGrad", "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(MaskedSelectGrad, MaskedSelectGradInfer);
// ---------------MaskedSelectGrad End---------------

// -------------------------------IdentityN Begin-------------------------------
// //
IMPLEMT_INFERFUNC(IdentityN, IdentityNInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  for (size_t i = 0; i < op.GetInputsSize(); i++) {
    auto input_desc = op_desc->MutableInputDesc(i);
    auto input_dims = input_desc->MutableShape().GetDims();
    auto output_desc = op_desc->MutableOutputDesc(i);
    auto intput_dtype = input_desc->GetDataType();

    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    output_desc->SetShape(GeShape(input_dims));
    output_desc->SetOriginShape(GeShape(input_dims));
    output_desc->SetDataType(intput_dtype);
    output_desc->SetShapeRange(input_range);
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(IdentityN, IdentityNInfer);
// -------------------------------IdentityN End-------------------------------
// //

// -------------------------------LowerBound------------------------------- //
IMPLEMT_INFERFUNC(LowerBound, LowerBoundInfer) {
  TensorDesc sorted_x_desc = op.GetInputDescByName("sorted_x");
  TensorDesc values_desc = op.GetInputDescByName("values");
  Shape unused_shape;
  if (WithRank(sorted_x_desc, 2, unused_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op),
      ConcatString("call WithRank failed, ", GetShapeErrMsg(0, DebugString(sorted_x_desc.GetShape().GetDims()), "2D")));
    return GRAPH_FAILED;
  }
  if (WithRank(values_desc, 2, unused_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op),
      ConcatString("call WithRank failed, ", GetShapeErrMsg(1, DebugString(values_desc.GetShape().GetDims()), "2D")));
    return GRAPH_FAILED;
  }

  DataType out_type;
  if (op.GetAttr("out_type", out_type) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr [out_type] failed.");
    return GRAPH_FAILED;
  }
  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetDataType(out_type);
  y_desc.SetShape(values_desc.GetShape());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update [y] desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(LowerBound, LowerBoundInfer);
// -------------------------------LowerBound END-------------------------------
// //

// -------------------------------ListDiff------------------------------- //
IMPLEMT_INFERFUNC(ListDiff, ListDiffInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);
  auto y_desc = op_desc->MutableInputDesc(1);

  Shape unused_shape;
  std::string error_msg;
  if (WithRank(x_desc, 1, unused_shape, op) != GRAPH_SUCCESS) {
    std::string error_msg = GetShapeErrMsg(0, DebugString(x_desc->GetShape().GetDims()), "1D");
    error_msg = string("failed to call WithRank function, ") + error_msg;
    return GRAPH_FAILED;
  }

  if (WithRank(y_desc, 1, unused_shape, op) != GRAPH_SUCCESS) {
    std::string error_msg = GetShapeErrMsg(1, DebugString(y_desc->GetShape().GetDims()), "1D");
    error_msg = string("failed to call WithRank function, ") + error_msg;
    return GRAPH_FAILED;
  }

  DataType output_type = x_desc->GetDataType();
  DataType index_type;
  if (op.GetAttr("out_idx", index_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("failed to get attr[out_idx]."));
    return GRAPH_FAILED;
  }

  GeShape result({ge::UNKNOWN_DIM});
  auto output_desc = op_desc->MutableOutputDesc(0);
  output_desc->SetShape(GeShape(result));
  output_desc->SetDataType(output_type);

  auto index_desc = op_desc->MutableOutputDesc(1);
  index_desc->SetShape(GeShape(result));
  index_desc->SetDataType(index_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(ListDiff, ListDiffInfer);
// -------------------------------ListDiff END------------------------------- //

// ----------------HammingWindow Begin---------------------
IMPLEMT_COMMON_INFERFUNC(HammingWindowInferShape) {
  std::vector<int64_t> input_dim = op.GetInputDesc(0).GetShape().GetDims();
  if (input_dim.size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "Tensor length input must be 1D.");
    return GRAPH_FAILED;
  }

  Tensor length_tensor;
  int64_t length_data;
  if (op.GetInputConstData("length", length_tensor) == GRAPH_SUCCESS) {
    uint8_t *length = length_tensor.GetData();
    length_data = static_cast<int64_t>(*length);
  } else {
    length_data = UNKNOWN_DIM;
  }
  std::vector<int64_t> output_dim;
  if (length_data != UNKNOWN_DIM && length_data < 0) {
    OP_LOGE(TbeGetName(op).c_str(), "Non-negative window length required, got [%ld].", length_data);
    return GRAPH_FAILED;
  }
  if (length_data != 0) {
    output_dim.push_back(length_data);
  }
  ge::Shape output_shape = ge::Shape(output_dim);

  Operator::OpInt dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    dtype = 0;
  }
  DataType output_dtype = static_cast<DataType>(dtype);

  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(output_dtype);
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(HammingWindow, HammingWindowInferShape);
// ----------------HammingWindow End---------------------

// ----------------Mvlgamma Begin-------------------
CUST_IMPLEMT_INFERFUNC(Mvlgamma, MvlgammaInferShape) {
  const char *op_name = "Mvlgamma";
  OP_LOGD(op_name, "MvlgammaInferShape begin.");
  TensorDesc tensordesc_input = op.GetInputDescByName("x");
  Shape input_shape = tensordesc_input.GetShape();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output1 = op.GetOutputDescByName("y");
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output1.SetShape(ge::Shape(dims_input));

  (void)op.UpdateOutputDesc("y", tensordesc_output1);
  OP_LOGD(op_name, "MvlgammaInferShape end.");
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(Mvlgamma, MvlgammaVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(Mvlgamma, MvlgammaInferShape);
CUST_VERIFY_FUNC_REG(Mvlgamma, MvlgammaVerify);
// ----------------Mvlgamma END---------------------

// ----------------MvlgammaGrad Begin-------------------
CUST_IMPLEMT_INFERFUNC(MvlgammaGrad, MvlgammaGradInferShape) {
  const char *op_name = "MvlgammaGrad";
  OP_LOGD(op_name, "MvlgammaGradInferShape begin.");
  TensorDesc tensordesc_input = op.GetInputDescByName("y_grad");
  Shape input_shape = tensordesc_input.GetShape();
  std::vector<int64_t> dims_input = input_shape.GetDims();
  DataType input_dtype = tensordesc_input.GetDataType();

  TensorDesc tensordesc_output1 = op.GetOutputDescByName("x_grad");
  tensordesc_output1.SetDataType(input_dtype);
  tensordesc_output1.SetShape(ge::Shape(dims_input));

  (void)op.UpdateOutputDesc("x_grad", tensordesc_output1);
  OP_LOGD(op_name, "MvlgammaGradInferShape end.");
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(MvlgammaGrad, MvlgammaGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(MvlgammaGrad, MvlgammaGradInferShape);
CUST_VERIFY_FUNC_REG(MvlgammaGrad, MvlgammaGradVerify);
// ----------------MvlgammaGrad END---------------------

// --------------------------LogSpace---------------------
static bool CheckSteps(const Operator &op, const string &attr_num_steps) {
  int64_t steps = 0;
  int64_t steps_ori = 100;
  if (ge::GRAPH_SUCCESS != op.GetAttr(attr_num_steps.c_str(), steps)) {
    steps = steps_ori;
  }
  if (steps < 0) {
    return false;
  }
  return true;
}

CUST_IMPLEMT_VERIFIER(LogSpace, LogSpaceVerify) {
  AscendString opName;
  op.GetName(opName);
  if (op.GetInputDescByName("start").GetShape().GetDims().size() > 1) {
    OP_LOGE(opName.GetString(), "Input start size must be <= 1.");
    return GRAPH_FAILED;
  }
  if (op.GetInputDescByName("end").GetShape().GetDims().size() > 1) {
    OP_LOGE(opName.GetString(), "Input  end size must be <= 1.");
    return GRAPH_FAILED;
  }
  DataType input_type_start = op.GetInputDescByName("start").GetDataType();
  DataType input_type_end = op.GetInputDescByName("end").GetDataType();
  if (input_type_start != input_type_end) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LogSpaceInferShape) {
  AscendString opName1;
  op.GetName(opName1);
  TensorDesc v_output_desc = op.GetOutputDescByName("y");
  int64_t steps;
  int64_t num_rows = 1;
  op.GetAttr("steps", steps);
  if (!CheckSteps(op, "steps")) {
    OP_LOGE(opName1.GetString(), "the attr 'steps' should be greater than or equal to 0.");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dim_vec;
  dim_vec.push_back(num_rows);
  dim_vec.push_back(steps);
  v_output_desc.SetShape(ge::Shape(dim_vec));
  int64_t dtype = 1;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    v_output_desc.SetDataType(DT_FLOAT16);
  } else {
    if (dtype == 1) {
      v_output_desc.SetDataType(DT_FLOAT16);
    }
    if (dtype == 0) {
      v_output_desc.SetDataType(DT_FLOAT);
    }
  }
  (void)op.UpdateOutputDesc("y", v_output_desc);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogSpace, LogSpaceInferShape);
// Registered verify function
CUST_VERIFY_FUNC_REG(LogSpace, LogSpaceVerify);
// --------------------------LogSpace END---------------------

// ----------------UniqueConsecutive Begin-------------------
IMPLEMT_INFERFUNC(UniqueConsecutive, UniqueConsecutiveInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc_ptr = op_desc->MutableInputDesc(0);
  auto y_desc_ptr = op_desc->MutableOutputDesc(0);
  y_desc_ptr->SetDataType(x_desc_ptr->GetDataType());

  auto idx_desc_ptr = op_desc->MutableOutputDesc(1);
  auto count_desc_ptr = op_desc->MutableOutputDesc(2);

  auto &y_shape = y_desc_ptr->MutableShape();
  auto &idx_shape = idx_desc_ptr->MutableShape();
  auto &count_shape = count_desc_ptr->MutableShape();

  bool return_idx = false;
  bool return_counts = false;
  int64_t axis = 1000;

  op.GetAttr("axis", axis);
  op.GetAttr("return_idx", return_idx);
  op.GetAttr("return_counts", return_counts);
  count_shape.SetIsUnknownDimNum();
  count_desc_ptr->SetDataType(DT_INT64);
  idx_shape.SetIsUnknownDimNum();
  idx_desc_ptr->SetDataType(DT_INT64);
  y_shape.SetIsUnknownDimNum();

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(UniqueConsecutive, UniqueConsecutiveInfer);
// ----------------UniqueConsecutive End-----------------------

// ----------------UpperBound-----------------------
IMPLEMT_INFERFUNC(UpperBound, UpperBoundInfer) {
  Shape unused_shape;
  if (WithRank(op.GetInputDesc(0), 2, unused_shape, op) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("failed to call WithRank function, input[sorted_x] rank must be 2D, got rank[",
                                  op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRank(op.GetInputDesc(1), 2, unused_shape, op) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("failed to call WithRank function, input[values] rank must be 2D, got rank[",
                                  op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  DataType type;
  if (op.GetAttr("out_type", type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[out_type] failed"));
    return GRAPH_FAILED;
  }

  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetShape(op.GetInputDesc(1).GetShape());
  out_desc.SetDataType(type);
  return op.UpdateOutputDesc("y", out_desc);
}

INFER_FUNC_REG(UpperBound, UpperBoundInfer);
// ----------------UpperBound END-----------------------

// ----------------UnravelIndex-----------------------
IMPLEMT_INFERFUNC(UnravelIndex, UnravelIndexInfer) {
  auto indices_desc = op.GetInputDesc(0);
  auto dims_desc = op.GetInputDesc(1);

  Shape dims_shape;
  if (WithRank(dims_desc, 1, dims_shape, op) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[dims] must be 1D, real rank is ", dims_shape.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  Shape indices_shape;
  if (WithRankAtMost(indices_desc, 1, indices_shape, op) != GRAPH_SUCCESS) {
    string err_msg = ConcatString("input[indices] must be less than 1D, real rank is ", dims_shape.GetDimNum());
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  std::vector<int64_t> out_dims({-1, -1});
  std::vector<int64_t> dims_shape_vec = dims_shape.GetDims();
  std::vector<int64_t> indices_shape_vec = indices_shape.GetDims();
  if (indices_shape.GetDimNum() == 0) {
    out_dims.pop_back();
  } else {
    if (indices_shape_vec != ge::UNKNOWN_RANK && indices_shape_vec != ge::UNKNOWN_SHAPE) {
      out_dims[1] = indices_shape_vec[0];
    }
  }
  if (dims_shape_vec != ge::UNKNOWN_RANK && dims_shape_vec != ge::UNKNOWN_SHAPE) {
    out_dims[0] = dims_shape_vec[0];
  }

  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetShape(Shape(out_dims));
  out_desc.SetDataType(indices_desc.GetDataType());
  return op.UpdateOutputDesc("y", out_desc);
}

INFER_FUNC_REG(UnravelIndex, UnravelIndexInfer);
// ----------------UnravelIndex END-----------------------
}  // namespace ge
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/op_common_util.h"
#include "utils/op_const.h"

namespace ge {
const std::string op_prefix = "Cust";
std::string GetOpName(std::string op_name) {
  if (!op_name.compare(0, op_prefix.size(), op_prefix)) {
    op_name.erase(op_name.begin(), op_name.begin() + op_prefix.size());
  }
  return op_name;
}

DataType FFTGetType(std::string op_name, DataType x_dtype) {
  static const std::vector<DataType> double_type = {DT_DOUBLE, DT_COMPLEX128};
  static const std::vector<std::string> float_prim = {"HFFT",   "HFFT2", "HFFTN", "IRFFT", "IRFFT2",
                                                      "IRFFTN", "DCT",   "IDCT",  "DCTN",  "IDCTN"};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&x_dtype](const DataType &type_id) { return x_dtype == type_id; });
  bool is_float_prim = std::find(float_prim.begin(), float_prim.end(), op_name) != float_prim.end();
  DataType y_dtype;
  if (is_double_type && is_float_prim) {
    y_dtype = DT_DOUBLE;
  }
  if (is_double_type && !is_float_prim) {
    y_dtype = DT_COMPLEX128;
  }
  if (!is_double_type && is_float_prim) {
    y_dtype = DT_FLOAT;
  }
  if (!is_double_type && !is_float_prim) {
    y_dtype = DT_COMPLEX64;
  }
  return y_dtype;
}

DataType DCTGetType(DataType x_dtype) {
  static const std::vector<DataType> double_type = {DT_DOUBLE};
  static const std::vector<DataType> complex_type = {DT_COMPLEX64, DT_COMPLEX128};
  bool is_double_type = std::any_of(double_type.begin(), double_type.end(),
                                    [&x_dtype](const DataType &type_id) { return x_dtype == type_id; });
  bool is_complex_type = std::any_of(complex_type.begin(), complex_type.end(),
                                     [&x_dtype](const DataType &type_id) { return x_dtype == type_id; });
  DataType y_dtype;
  if (is_double_type) {
    y_dtype = DT_DOUBLE;
  } else if (is_complex_type) {
    y_dtype = x_dtype;
  } else {
    y_dtype = DT_FLOAT;
  }

  return y_dtype;
}

void FFTNGetAttr(const std::vector<int64_t> input_shape, size_t x_rank, std::vector<int64_t> *s_vec,
                 std::vector<int64_t> *dim_vec) {
  std::vector<int64_t> s = *s_vec;
  std::vector<int64_t> dim = *dim_vec;
  if (dim.empty() && !s.empty()) {
    for (size_t i = 0; i < s.size(); i++) {
      (void)dim.emplace_back(x_rank - s.size() + i);
    }
  }
  if (s.empty() && !dim.empty()) {
    for (size_t i = 0; i < dim.size(); i++) {
      (void)s.emplace_back(input_shape[dim[i]]);
    }
  }
  if (s.empty() && dim.empty()) {
    for (size_t i = 0; i < x_rank; i++) {
      (void)dim.emplace_back(i);
      (void)s.emplace_back(input_shape[i]);
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(FFTBaseInferShape) {
  auto input_desc = op.GetInputDescByName("input");
  auto out_desc = op.GetOutputDescByName("y");
  auto op_name = GetOpName(op.GetOpType());

  DataType x_dtype = input_desc.GetDataType();
  DataType y_dtype = FFTGetType(op_name, x_dtype);
  out_desc.SetDataType(y_dtype);

  bool unknown_rank_shape = IsUnknownRankShape(input_desc.GetShape());
  if (unknown_rank_shape) {
    out_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
    op.UpdateOutputDesc("y", out_desc);
    return GRAPH_SUCCESS;
  }

  size_t x_rank = input_desc.GetShape().GetDimNum();
  auto input_shape = input_desc.GetShape().GetDims();
  vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
  const vector<string> depend_names = {"n", "dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // infer output shape based on 'n' and 'dim'
  Tensor dim_tensor;
  std::vector<int64_t> dim_vec;
  if (op.GetInputConstData("dim", dim_tensor) == GRAPH_SUCCESS) {
    DataType dim_dtype = op.GetInputDescByName("dim").GetDataType();
    GetConstValue(op, dim_tensor, dim_dtype, dim_vec);
    for (size_t i = 0; i < dim_vec.size(); i++) {
      dim_vec[i] = dim_vec[i] < 0 ? static_cast<int64_t>(x_rank) + dim_vec[i] : dim_vec[i];
    }
  }

  Tensor s_tensor;
  std::vector<int64_t> s_vec;
  bool s_is_none{true};
  if (op.GetInputConstData("n", s_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("n").GetDataType();
    GetConstValue(op, s_tensor, dtype, s_vec);
    s_is_none = false;
  }

  FFTNGetAttr(output_shape, x_rank, &s_vec, &dim_vec);
  int64_t dim = dim_vec[0];
  if (!s_is_none) {
    int64_t n = s_vec[0];
    output_shape[dim] = n;
    if (op_name == "IHFFT" || op_name == "RFFT") {
      output_shape[dim] = n / 2 + 1;
    }
  } else {
    if (op_name == "HFFT") {
      output_shape[dim] = (output_shape[dim] - 1) * 2;
    } else if (op_name == "IHFFT" || op_name == "RFFT") {
      output_shape[dim] = output_shape[dim] / 2 + 1;
    }
  }

  out_desc.SetShape(ge::Shape(output_shape));
  op.UpdateOutputDesc("y", out_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DCTInferShape) {
  auto input_desc = op.GetInputDescByName("x");
  auto out_desc = op.GetOutputDescByName("y");
  auto op_name = GetOpName(op.GetOpType());

  DataType x_dtype = input_desc.GetDataType();
  DataType y_dtype = DCTGetType(x_dtype);
  out_desc.SetDataType(y_dtype);

  bool unknown_rank_shape = IsUnknownRankShape(input_desc.GetShape());
  if (unknown_rank_shape) {
    out_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
    op.UpdateOutputDesc("y", out_desc);
    return GRAPH_SUCCESS;
  }

  size_t x_rank = input_desc.GetShape().GetDimNum();
  auto input_shape = input_desc.GetShape().GetDims();
  vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
  const vector<string> depend_names = {"n", "axis"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  // infer output shape based on 'n' and 'dim'
  Tensor dim_tensor;
  std::vector<int64_t> dim_vec;
  if (op.GetInputConstData("axis", dim_tensor) == GRAPH_SUCCESS) {
    DataType dim_dtype = op.GetInputDescByName("axis").GetDataType();
    GetConstValue(op, dim_tensor, dim_dtype, dim_vec);
    for (size_t i = 0; i < dim_vec.size(); i++) {
      dim_vec[i] = dim_vec[i] < 0 ? static_cast<int64_t>(x_rank) + dim_vec[i] : dim_vec[i];
    }
  }

  Tensor s_tensor;
  std::vector<int64_t> s_vec;
  bool s_is_none{true};
  if (op.GetInputConstData("n", s_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("n").GetDataType();
    GetConstValue(op, s_tensor, dtype, s_vec);
    s_is_none = false;
  }

  FFTNGetAttr(output_shape, x_rank, &s_vec, &dim_vec);
  int64_t dim = dim_vec[0];
  if (!s_is_none) {
    int64_t n = s_vec[0];
    output_shape[dim] = n;
  }
  out_desc.SetShape(ge::Shape(output_shape));
  op.UpdateOutputDesc("y", out_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FFTNBaseInferShape) {
  auto input_desc = op.GetInputDescByName("input");
  auto out_desc = op.GetOutputDescByName("y");
  auto op_name = GetOpName(op.GetOpType());
  DataType input_dtype = input_desc.GetDataType();
  DataType output_dtype = FFTGetType(op_name, input_dtype);
  out_desc.SetDataType(output_dtype);

  bool unknown_rank_shape = IsUnknownRankShape(input_desc.GetShape());
  if (unknown_rank_shape) {
    out_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
    op.UpdateOutputDesc("y", out_desc);
    return GRAPH_SUCCESS;
  }
  const vector<string> depend_names = {"s", "dim"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  std::vector<int64_t> s_vec;
  std::vector<int64_t> dim_vec;
  size_t x_rank = input_desc.GetShape().GetDimNum();
  auto input_shape = input_desc.GetShape().GetDims();
  vector<int64_t> output_shape(input_shape.begin(), input_shape.end());

  // infer output shape based on 's' and 'dim'
  Tensor s_tensor;
  bool s_is_none{true};
  if (op.GetInputConstData("s", s_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("s").GetDataType();
    GetConstValue(op, s_tensor, dtype, s_vec);
    s_is_none = false;
  }

  Tensor dim_tensor;
  if (op.GetInputConstData("dim", dim_tensor) == GRAPH_SUCCESS) {
    DataType dim_dtype = op.GetInputDescByName("dim").GetDataType();
    GetConstValue(op, dim_tensor, dim_dtype, dim_vec);
    for (size_t i = 0; i < dim_vec.size(); i++) {
      dim_vec[i] = dim_vec[i] < 0 ? static_cast<int64_t>(x_rank) + dim_vec[i] : dim_vec[i];
    }
  }

  FFTNGetAttr(output_shape, x_rank, &s_vec, &dim_vec);

  static const std::vector<std::string> half_shape_prim = {"IHFFT", "IHFFT2", "IHFFTN", "RFFT", "RFFT2", "RFFTN"};
  static const std::vector<std::string> double_shape_prim = {"HFFT", "HFFT2", "HFFTN", "IRFFT", "IRFFT2", "IRFFTN"};
  bool is_half_shape_prim = std::find(half_shape_prim.begin(), half_shape_prim.end(), op_name) != half_shape_prim.end();
  bool is_double_shape_prim =
    std::find(double_shape_prim.begin(), double_shape_prim.end(), op_name) != double_shape_prim.end();

  for (size_t i = 0; i < s_vec.size(); i++) {
    output_shape[dim_vec[i]] = s_vec[i];
  }

  if (is_double_shape_prim && s_is_none) {
    output_shape[dim_vec.back()] = (output_shape[dim_vec.back()] - 1) * 2;
  }
  if (is_half_shape_prim && s_is_none) {
    output_shape[dim_vec.back()] = output_shape[dim_vec.back()] / 2 + 1;
  }
  if (is_half_shape_prim && !s_is_none) {
    output_shape[dim_vec.back()] = s_vec.back() / 2 + 1;
  }

  out_desc.SetShape(ge::Shape(output_shape));
  op.UpdateOutputDesc("y", out_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DCTNInferShape) {
  auto input_desc = op.GetInputDescByName("x");
  auto out_desc = op.GetOutputDescByName("y");
  auto op_name = GetOpName(op.GetOpType());
  DataType input_dtype = input_desc.GetDataType();
  DataType output_dtype = DCTGetType(input_dtype);
  out_desc.SetDataType(output_dtype);

  bool unknown_rank_shape = IsUnknownRankShape(input_desc.GetShape());
  if (unknown_rank_shape) {
    out_desc.SetShape(ge::Shape(UNKNOWN_RANK));
    OP_LOGD(TbeGetName(op).c_str(), "output shape:%s", to_string(out_desc.GetShape()).c_str());
    op.UpdateOutputDesc("y", out_desc);
    return GRAPH_SUCCESS;
  }
  const vector<string> depend_names = {"s", "axes"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  std::vector<int64_t> s_vec;
  std::vector<int64_t> dim_vec;
  size_t x_rank = input_desc.GetShape().GetDimNum();
  auto input_shape = input_desc.GetShape().GetDims();
  vector<int64_t> output_shape(input_shape.begin(), input_shape.end());

  // infer output shape based on 's' and 'dim'
  Tensor s_tensor;
  if (op.GetInputConstData("s", s_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("s").GetDataType();
    GetConstValue(op, s_tensor, dtype, s_vec);
  }

  Tensor dim_tensor;
  if (op.GetInputConstData("axes", dim_tensor) == GRAPH_SUCCESS) {
    DataType dim_dtype = op.GetInputDescByName("axes").GetDataType();
    GetConstValue(op, dim_tensor, dim_dtype, dim_vec);
    for (size_t i = 0; i < dim_vec.size(); i++) {
      dim_vec[i] = dim_vec[i] < 0 ? static_cast<int64_t>(x_rank) + dim_vec[i] : dim_vec[i];
    }
  }

  FFTNGetAttr(output_shape, x_rank, &s_vec, &dim_vec);

  for (size_t i = 0; i < s_vec.size(); i++) {
    output_shape[dim_vec[i]] = s_vec[i];
  }

  out_desc.SetShape(ge::Shape(output_shape));
  op.UpdateOutputDesc("y", out_desc);
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FFTShiftInferShape) {
  TensorDesc out_desc = op.GetOutputDescByName("input");
  out_desc.SetDataType(op.GetInputDescByName("input").GetDataType());
  out_desc.SetShape(op.GetInputDescByName("input").GetShape());
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FFTShapeCopyInferShape) {
  TensorDesc input_desc = op.GetInputDescByName("input");
  TensorDesc out_desc = op.GetOutputDescByName("y");

  Tensor shape_tensor;
  Shape output_shape;
  if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS) {
    MakeShapeFromShapeTensor(shape_tensor, output_shape, op);
  } else {
    output_shape = Shape({UNKNOWN_RANK});
  }
  out_desc.SetDataType(input_desc.GetDataType());
  out_desc.SetShape(output_shape);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Failed to update output desc.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(FFTFreqInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("y");
  auto op_name = GetOpName(op.GetOpType());
  // infer type
  const static std::unordered_map<int64_t, DataType> MSTypes2GETypes{
    {30, DT_BOOL},  {32, DT_INT8},   {33, DT_INT16},  {34, DT_INT32},     {35, DT_INT64},
    {37, DT_UINT8}, {38, DT_UINT16}, {39, DT_UINT32}, {40, DT_UINT64},    {42, DT_FLOAT16},
    {43, DT_FLOAT}, {44, DT_DOUBLE}, {45, DT_BF16},   {48, DT_COMPLEX64}, {49, DT_COMPLEX128}};

  ge::DataType output_dtype = DT_FLOAT;
  Tensor dtype_tensor;
  std::vector<int64_t> dtype_vec;
  if (op.GetInputConstData("dtype", dtype_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("dtype").GetDataType();
    GetConstValue(op, dtype_tensor, dtype, dtype_vec);
    output_dtype = MSTypes2GETypes.find(dtype_vec[0])->second;
  }
  output_desc.SetDataType(output_dtype);

  // infer shape
  Tensor n_tensor;
  std::vector<int64_t> n_vec;
  if (op.GetInputConstData("n", n_tensor) == GRAPH_SUCCESS) {
    DataType dtype = op.GetInputDescByName("n").GetDataType();
    GetConstValue(op, n_tensor, dtype, n_vec);
  }
  int64_t n = n_vec[0];
  if (op_name == "RFFTFreq") {
    n = n / 2 + 1;
  }
  vector<int64_t> output_shape{n};
  output_desc.SetShape(Shape(output_shape));

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Update output desc failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(FFTFreq, FFTFreqInferShape);
CUST_COMMON_INFER_FUNC_REG(RFFTFreq, FFTFreqInferShape);

CUST_COMMON_INFER_FUNC_REG(FFTShapeCopy, FFTShapeCopyInferShape);

CUST_COMMON_INFER_FUNC_REG(FFTShift, FFTShiftInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFTShift, FFTShiftInferShape);
CUST_COMMON_INFER_FUNC_REG(IRFFTDouble, FFTShiftInferShape);
CUST_COMMON_INFER_FUNC_REG(FFTOrtho, FFTShiftInferShape);

CUST_COMMON_INFER_FUNC_REG(FFT, FFTBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFT, FFTBaseInferShape);

CUST_COMMON_INFER_FUNC_REG(DCT, DCTInferShape);
CUST_COMMON_INFER_FUNC_REG(DCTN, DCTNInferShape);
CUST_COMMON_INFER_FUNC_REG(IDCT, DCTInferShape);
CUST_COMMON_INFER_FUNC_REG(IDCTN, DCTNInferShape);

CUST_COMMON_INFER_FUNC_REG(HFFT, FFTBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IHFFT, FFTBaseInferShape);

CUST_COMMON_INFER_FUNC_REG(RFFT, FFTBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IRFFT, FFTBaseInferShape);

CUST_COMMON_INFER_FUNC_REG(FFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(FFTN, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IFFTN, FFTNBaseInferShape);

CUST_COMMON_INFER_FUNC_REG(HFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(HFFTN, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IHFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IHFFTN, FFTNBaseInferShape);

CUST_COMMON_INFER_FUNC_REG(RFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(RFFTN, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IRFFT2, FFTNBaseInferShape);
CUST_COMMON_INFER_FUNC_REG(IRFFTN, FFTNBaseInferShape);
}  // namespace ge

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

#include "inc/max_unpool_3d_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
static bool CheckListEmpty(const std::string &opName, const std::vector<int64_t> &list, const std::string &attrName) {
  if (list.empty()) {
    OP_LOGE(opName.c_str(), "the %s is empty !", attrName.c_str());
    return false;
  }
  return true;
}

static std::vector<int64_t> GetAttrValue(const ge::Operator &op, const std::string &key_name) {
  std::vector<int64_t> list;
  AscendString op_name;
  CHECK(op.GetName(op_name) != GRAPH_SUCCESS, OP_LOGE("", "failed to get op_name"), return list);
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name.c_str(), list)) {
    CUBE_INNER_ERR_REPORT(op_name.GetString(), "GetOpAttr ConstValue failed!");
  }

  return list;
}
// ---------------------MaxUnpool3D---------------------
CUST_IMPLEMT_VERIFIER(MaxUnpool3D, MaxUnpool3DVerify) {
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      string expected_format_list = ConcatString("NCDHW, NDHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksize");
  std::vector<int64_t> strides;
  strides = GetAttrValue(op, "strides");
  std::vector<int64_t> pads;
  pads = GetAttrValue(op, "pads");

  if (!CheckListEmpty(TbeGetName(op).c_str(), ksize, "ksize") ||
      !CheckListEmpty(TbeGetName(op).c_str(), strides, "strides") ||
      !CheckListEmpty(TbeGetName(op).c_str(), pads, "pads")) {
    std::string err_msg = OtherErrMsg("The ksize or strides or pads is empty!");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (ksize.size() != 5 || strides.size() != 5 || pads.size() != 5) {
    string excepted_size = ConcatString("5");
    std::string err_msg =
      GetAttrSizeErrMsg("ksize.size or strides.size or pads.size", std::to_string(ksize.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NCDHW" &&
      (ksize[0] != 1 || ksize[1] != 1 || strides[0] != 1 || strides[1] != 1 || pads[0] != 1 || pads[1] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[1], "and", strides[0], "and", strides[1], "and", pads[0], "and", pads[1]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[1] and strides[0] and strides[1] and pads[0] and pads[1]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format == "NDHWC" &&
      (ksize[0] != 1 || ksize[4] != 1 || strides[0] != 1 || strides[4] != 1 || pads[0] != 1 || pads[4] != 1)) {
    string wrong_value =
      ConcatString(ksize[0], " and ", ksize[4], "and", strides[0], "and", strides[4], "and", pads[0], "and", pads[4]);
    std::string err_msg = GetAttrValueErrMsg(
      "ksize[0] and ksize[4] and strides[0] and strides[4] and pads[0] and pads[4]", wrong_value, ConcatString("1"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxUnpool3D, MaxUnpool3DInferShape) {
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto input_shape = input_tensor_desc.GetShape();
  std::string data_format;
  if (ge::GRAPH_SUCCESS == op.GetAttr("data_format", data_format)) {
    if (data_format != "NCDHW" && data_format != "NDHWC") {
      string expected_format_list = ConcatString("NCDHW, NDHWC");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }

  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  std::vector<int64_t> output_shape_attr;
  output_shape_attr = GetAttrValue(op, "output_shape");

  ge::Shape output_shape;
  std::vector<int64_t> vec_kernel;
  std::vector<int64_t> vec_strides;
  std::vector<int64_t> vec_pads;
  int H_index, W_index, N_index, C_index, D_index;
  std::vector<int64_t> dim_vector;

  if (CheckListEmpty(TbeGetName(op).c_str(), output_shape_attr, "output_shape")) {
    td.SetShape(Shape(output_shape_attr));
  } else {
    op.GetAttr("ksize", vec_kernel);
    op.GetAttr("strides", vec_strides);
    op.GetAttr("pads", vec_pads);

    if (data_format == "NCDHW") {
      N_index = 0;
      C_index = 1;
      D_index = 2;
      H_index = 3;
      W_index = 4;

    } else {
      N_index = 0;
      C_index = 4;
      D_index = 1;
      H_index = 2;
      W_index = 3;
    }
    int32_t output_N = input_shape.GetDim(N_index);
    int32_t output_C = input_shape.GetDim(C_index);
    int32_t output_D =
      (input_shape.GetDim(D_index) - 1) * vec_strides[D_index] - 2 * vec_pads[D_index] + vec_kernel[D_index];
    int32_t output_H =
      (input_shape.GetDim(H_index) - 1) * vec_strides[H_index] - 2 * vec_pads[H_index] + vec_kernel[H_index];
    int32_t output_W =
      (input_shape.GetDim(W_index) - 1) * vec_strides[W_index] - 2 * vec_pads[W_index] + vec_kernel[W_index];
    if (data_format == "NCDHW") {
      dim_vector.push_back(output_N);
      dim_vector.push_back(output_C);
      dim_vector.push_back(output_D);
      dim_vector.push_back(output_H);
      dim_vector.push_back(output_W);
    } else {
      dim_vector.push_back(output_N);
      dim_vector.push_back(output_D);
      dim_vector.push_back(output_H);
      dim_vector.push_back(output_W);
      dim_vector.push_back(output_C);
    }
    Shape output_shape(dim_vector);
    td.SetShape(output_shape);
  }

  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "UpdateOutputDesc run failed. Check whether the names of outputs are matched.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(MaxUnpool3D, MaxUnpool3DInferShape);
CUST_VERIFY_FUNC_REG(MaxUnpool3D, MaxUnpool3DVerify);
// ---------------------MaxUnpool3D---------------------
}  // namespace ge
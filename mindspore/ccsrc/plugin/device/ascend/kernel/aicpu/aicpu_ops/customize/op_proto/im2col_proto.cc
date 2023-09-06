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

#include "inc/ops/transformation_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "graph/common_error_codes.h"
namespace ge {
const std::string ATTR_NAME_DATA_SLICE = "_data_slice";

static bool CheckListEmptyAndValue(const std::string &op_name, const std::vector<int64_t> &list,
                                   const std::string &attr_name) {
  if (list.size() < 1) {
    OP_LOGE(op_name.c_str(), "The %s dose not have enough elements(%lu)!", attr_name.c_str(), list.size());
    return false;
  }
  return true;
}

static std::vector<int64_t> GetAttrValue(const Operator &op, const std::string &key_name) {
  std::vector<int64_t> list;
  if (op.GetAttr(key_name.c_str(), list) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ConstValue failed!");
  }
  return list;
}

// -----------------Im2col Op-------------------------
IMPLEMT_VERIFIER(Im2col, Im2colVerify) {
  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksizes");
  if (ksize.size() < 2) {
    OP_LOGE(TbeGetName(op).c_str(), "The ksizes dose not have enough elements(%lu)!", ksize.size());
    return GRAPH_FAILED;
  }

  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  if (!CheckListEmptyAndValue(TbeGetName(op), stride, "strides")) {
    return GRAPH_FAILED;
  }

  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  if (!CheckListEmptyAndValue(TbeGetName(op), dilation, "dilations")) {
    return GRAPH_FAILED;
  }

  std::string padding_mode;
  if (op.GetAttr("padding_mode", padding_mode) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get padding_mode failed!");
    return GRAPH_FAILED;
  }
  if (padding_mode != "CALCULATED" && padding_mode != "SAME" && padding_mode != "VALID") {
    OP_LOGE(TbeGetName(op).c_str(), "padding_mode only support CALCULATED, SAME and VALID!");
    return GRAPH_FAILED;
  }

  std::vector<int64_t> pad;
  pad = GetAttrValue(op, "pads");
  if (!CheckListEmptyAndValue(TbeGetName(op), pad, "pads")) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(Im2colInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), "Enter op_proto inferfunction!");

  std::vector<int64_t> ksize;
  ksize = GetAttrValue(op, "ksizes");
  std::vector<int64_t> stride;
  stride = GetAttrValue(op, "strides");
  std::vector<int64_t> dilation;
  dilation = GetAttrValue(op, "dilations");
  std::string padding_mode;
  if (op.GetAttr("padding_mode", padding_mode) == GRAPH_FAILED) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ConstValue padding_mode failed!");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> pad;
  pad = GetAttrValue(op, "pads");

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr desc_in_ptr = op_desc->MutableInputDesc("x");
  GeTensorDescPtr desc_out_ptr = op_desc->MutableOutputDesc("y");
  auto dtype = desc_in_ptr->GetDataType();
  auto shape_in = desc_in_ptr->GetShape();
  auto x_format = desc_in_ptr->GetOriginFormat();
  if (x_format != FORMAT_NHWC && x_format != FORMAT_NCHW) {
    OP_LOGE(TbeGetName(op).c_str(), "Attr x_format only support NHWC, NCHW.");
    return GRAPH_FAILED;
  }

  std::map<char, int> idx_map{{'N', 0}, {'H', 1}, {'W', 2}, {'C', 3}};
  if (x_format == FORMAT_NCHW) {
    idx_map = {{'N', 0}, {'C', 1}, {'H', 2}, {'W', 3}};
  }

  int64_t in_n = shape_in.GetDim(idx_map['N']);
  int64_t in_h = shape_in.GetDim(idx_map['H']);
  int64_t in_w = shape_in.GetDim(idx_map['W']);
  int64_t in_c = shape_in.GetDim(idx_map['C']);

  if (ksize.size() != 2) {
    OP_LOGE(TbeGetName(op).c_str(), "The size of ksizes must be 2 when x_format only support NHWC, NCHW.");
    return GRAPH_FAILED;
  }
  int64_t filter_h = ksize[0];
  int64_t filter_w = ksize[1];

  int64_t stride_h = stride[0];
  int64_t stride_w = stride[0];
  if (stride.size() == 2) {
    stride_h = stride[0];
    stride_w = stride[1];
  } else if (stride.size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "The size of strides must be 1 or 2 when x_format only support NHWC, NCHW.");
    return GRAPH_FAILED;
  }
  if (stride_h == 0 || stride_w == 0) {
    OP_LOGE(TbeGetName(op).c_str(), "The stride_h or stride_w should not 0");
    return GRAPH_FAILED;
  }

  int64_t dilation_h = dilation[0];
  int64_t dilation_w = dilation[0];
  if (dilation.size() == 2) {
    dilation_h = dilation[0];
    dilation_w = dilation[1];
  } else if (dilation.size() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "The size of dilations must be 1 or 2 when x_format only support NHWC, NCHW.");
    return GRAPH_FAILED;
  }

  int64_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int64_t effective_filter_w = (filter_w - 1) * dilation_w + 1;
  int64_t out_h{0};
  int64_t out_w{0};
  int64_t out_c{0};
  if (padding_mode == "VALID") {
    out_h = (in_h - effective_filter_h + stride_h) / stride_h;
    out_w = (in_w - effective_filter_w + stride_w) / stride_w;
  } else if (padding_mode == "SAME") {
    out_h = (in_h + stride_h - 1) / stride_h;
    out_w = (in_w + stride_w - 1) / stride_w;
  } else if (padding_mode == "CALCULATED") {
    int64_t pad_h_top;
    int64_t pad_h_bottom;
    int64_t pad_w_before;
    int64_t pad_w_after;
    if (pad.size() == 1) {
      pad_h_top = pad[0];
      pad_h_bottom = pad[0];
      pad_w_before = pad[0];
      pad_w_after = pad[0];
    } else if (pad.size() == 4) {
      pad_h_top = pad[0];
      pad_h_bottom = pad[1];
      pad_w_before = pad[2];
      pad_w_after = pad[3];
    } else {
      OP_LOGE(TbeGetName(op).c_str(), "The size of pads must be 1 or 4 when x_format only support NHWC, NCHW.");
      return GRAPH_FAILED;
    }
    out_h = (in_h + pad_h_top + pad_h_bottom - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
    out_w = (in_w + pad_w_before + pad_w_after - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;
  } else {
    OP_LOGE(TbeGetName(op).c_str(), "The padding_mode only support VALID, SAME and CALCULATED.");
    return GRAPH_FAILED;
  }
  out_c = in_c * filter_h * filter_w;

  std::vector<int64_t> out_dim{in_n, out_h, out_w, out_c};
  if (x_format == FORMAT_NCHW) {
    out_dim = {in_n, out_c, out_h, out_w};
  }

  desc_out_ptr->SetShape(ge::GeShape(out_dim));
  desc_out_ptr->SetDataType(dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Im2col, Im2colInferShape);
VERIFY_FUNC_REG(Im2col, Im2colVerify);
// -----------------Im2col END-------------------------
}  // namespace ge
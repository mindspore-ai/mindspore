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

#include "inc/batch_norm_grad_grad_op.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"

namespace ge {
CUST_IMPLEMT_INFERFUNC(BatchNormGradGrad, BatchNormGradGradInferShape) {
  // check attr
  float epsilon;
  if (op.GetAttr("epsilon", epsilon) == GRAPH_SUCCESS) {
    if (epsilon <= 0) {
      OP_LOGE(TbeGetName(op).c_str(), "'epsilon' must be greater than 0");
      return GRAPH_FAILED;
    }
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
    if (data_format != "NHWC" && data_format != "NCHW") {
      string expected_format_list = ConcatString("NHWC, NCHW");
      std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  ge::Format format;
  if (data_format == "NCHW") {
    format = FORMAT_NCHW;
  } else {
    format = FORMAT_NHWC;
  }

  // check dtype
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType dy_dtype = op.GetInputDescByName("dy").GetDataType();
  DataType ddx_dtype = op.GetInputDescByName("ddx").GetDataType();

  DataType scale_dtype = op.GetInputDescByName("scale").GetDataType();
  DataType reserve_space_1_dtype = op.GetInputDescByName("reserve_space_1").GetDataType();
  DataType reserve_space_2_dtype = op.GetInputDescByName("reserve_space_2").GetDataType();
  DataType ddscale_dtype = op.GetInputDescByName("ddscale").GetDataType();
  DataType ddoffset_dtype = op.GetInputDescByName("ddoffset").GetDataType();

  if (x_dtype != DT_FLOAT16 && x_dtype != DT_FLOAT) {
    OP_LOGE(TbeGetName(op).c_str(), "'x' should have datatype fp16 or fp32");
    return GRAPH_FAILED;
  }

  if (x_dtype != dy_dtype || x_dtype != ddx_dtype) {
    OP_LOGE(TbeGetName(op).c_str(), "'x' 'dy' 'ddx' should have the same datatype");
    return GRAPH_FAILED;
  }

  if (scale_dtype != DT_FLOAT || reserve_space_1_dtype != DT_FLOAT || reserve_space_2_dtype != DT_FLOAT ||
      ddscale_dtype != DT_FLOAT || ddoffset_dtype != DT_FLOAT) {
    OP_LOGE(TbeGetName(op).c_str(),
            "'scale' 'reserve_space_1' 'reserve_space_2' 'ddscale' 'ddoffset' must have datatype fp32");
    return GRAPH_FAILED;
  }

  // check shape
  ge::Shape x_shape = op.GetInputDescByName("x").GetShape();
  ge::Shape dy_shape = op.GetInputDescByName("dy").GetShape();
  ge::Shape ddx_shape = op.GetInputDescByName("ddx").GetShape();

  ge::Shape scale_shape = op.GetInputDescByName("scale").GetShape();
  ge::Shape reserve_space_1_shape = op.GetInputDescByName("reserve_space_1").GetShape();
  ge::Shape reserve_space_2_shape = op.GetInputDescByName("reserve_space_2").GetShape();
  ge::Shape ddscale_shape = op.GetInputDescByName("ddscale").GetShape();
  ge::Shape ddoffset_shape = op.GetInputDescByName("ddoffset").GetShape();

  if (x_shape.GetDimNum() != 4) {
    OP_LOGE(TbeGetName(op).c_str(), "'x' must be a 4D tensor");
    return GRAPH_FAILED;
  }

  if (x_shape.GetDims() != dy_shape.GetDims() || x_shape.GetDims() != ddx_shape.GetDims()) {
    OP_LOGE(TbeGetName(op).c_str(), "'x' 'dy' 'ddx' must have the same shape");
    return GRAPH_FAILED;
  }

  if (scale_shape.GetDimNum() != 1) {
    OP_LOGE(TbeGetName(op).c_str(), "'scale' must be a 1D tensor");
    return GRAPH_FAILED;
  }

  if (scale_shape.GetDims() != reserve_space_1_shape.GetDims() ||
      scale_shape.GetDims() != reserve_space_2_shape.GetDims() || scale_shape.GetDims() != ddscale_shape.GetDims() ||
      scale_shape.GetDims() != ddoffset_shape.GetDims()) {
    OP_LOGE(TbeGetName(op).c_str(),
            "'scale' 'reserve_space_1' 'reserve_space_2' 'ddscale' 'ddoffset' must have the same shape");
    return GRAPH_FAILED;
  }

  if ((format == FORMAT_NHWC && x_shape.GetDim(3) != scale_shape.GetShapeSize()) ||
      (format == FORMAT_NCHW && x_shape.GetDim(1) != scale_shape.GetShapeSize())) {
    OP_LOGE(TbeGetName(op).c_str(), "the size of 1D tensor should be equal to the size of C dim of 'x'");
    return GRAPH_FAILED;
  }

  // infer dtype and format and shape
  TensorDesc dx_desc = op.GetOutputDescByName("dx");
  dx_desc.SetDataType(x_dtype);
  dx_desc.SetFormat(format);
  dx_desc.SetShape(x_shape);
  (void)op.UpdateOutputDesc("dx", dx_desc);

  TensorDesc ddy_desc = op.GetOutputDescByName("ddy");
  ddy_desc.SetDataType(dy_dtype);
  ddy_desc.SetFormat(format);
  ddy_desc.SetShape(dy_shape);
  (void)op.UpdateOutputDesc("ddy", ddy_desc);

  TensorDesc dscale_desc = op.GetOutputDescByName("dscale");
  dscale_desc.SetDataType(scale_dtype);
  dscale_desc.SetShape(scale_shape);
  (void)op.UpdateOutputDesc("dscale", dscale_desc);

  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(BatchNormGradGrad, BatchNormGradGradInferShape);
}  // namespace ge
/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/nn_pooling_ops.h"
#include "custom_op_proto/cust_nn_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------AdaptiveAvgPool2D-------------------
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape) {
  OP_LOGI(TbeGetName(op).c_str(), " AdaptiveAvgPool2d inferShape begin!");
  const size_t DIM_SIZE2 = 2;
  auto input_tensor_desc = op.GetInputDescByName("x");
  auto shape = input_tensor_desc.GetShape();
  // get output_size
  std::vector<int64_t> ouput_size_list;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", ouput_size_list)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr ouput_size_list failed!");
    return GRAPH_FAILED;
  }
  // check output size
  if (ouput_size_list.size() != DIM_SIZE2) {
    OP_LOGE(TbeGetName(op).c_str(), "length of output_size must be 2");
    return GRAPH_FAILED;
  }
  std::vector<int64_t> dims_input = shape.GetDims();
  // set output shape
  std::vector<int64_t> dim_vector;
  for (size_t i = 0; i < dims_input.size(); i++) {
    int64_t dims = dims_input[i];
    dim_vector.push_back(dims);
  }
  size_t index0 = dims_input.size() - 2;
  size_t index1 = dims_input.size() - 1;
  if (ouput_size_list[0] > 0) {
    dim_vector[index0] = ouput_size_list[0];
  }
  if (ouput_size_list[1] > 0) {
    dim_vector[index1] = ouput_size_list[1];
  }
  TensorDesc td = op.GetOutputDescByName("y");
  DataType input_dtype = input_tensor_desc.GetDataType();
  Shape output_shape(dim_vector);
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveAvgPool2D, AdaptiveAvgPool2dVerify);
// ---------------AdaptiveAvgPool2D End---------------

// ---------------AdaptiveAvgPool2DGrad-------------------
CUST_IMPLEMT_INFERFUNC(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape) {
  std::vector<std::string> input_infer_depends = {"orig_input_shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  DataType input_dtype = op.GetInputDescByName("input_grad").GetDataType();
  Shape output_shape;
  Tensor orig_input_shape_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_shape_tensor) != GRAPH_SUCCESS) {
    auto output_desc = op.GetOutputDescByName("output_grad");
    output_desc.SetDataType(input_dtype);
    output_desc.SetShape(Shape(ge::UNKNOWN_RANK));
    return op.UpdateOutputDesc("output_grad", output_desc);
  }
  MakeShapeFromShapeTensor(orig_input_shape_tensor, output_shape, op);
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  output_grad.SetShape(output_shape);
  output_grad.SetDataType(input_dtype);
  return op.UpdateOutputDesc("output_grad", output_grad);
}

CUST_INFER_FUNC_REG(AdaptiveAvgPool2DGrad, AdaptiveAvgPool2dGradInferShape);
// ---------------AdaptiveAvgPool2DGrad END-------------------

// --------- AdaptiveAvgPool3d ---------------
IMPLEMT_COMMON_INFERFUNC(AdaptiveAvgPool3dInferShape) {
  // verify the dim of output_size
  std::vector<int64_t> output_size;
  if (GRAPH_SUCCESS != op.GetAttr("output_size", output_size)) {
    OP_LOGE(TbeGetName(op).c_str(), "GetOpAttr output_size failed!");
    return GRAPH_PARAM_INVALID;
  }
  ge::AscendString op_name;
  (void)op.GetName(op_name);
  auto input_desc = op.GetInputDescByName("x");
  TensorDesc out_desc = op.GetOutputDescByName("y");

  // update data type
  DataType input_type = input_desc.GetDataType();
  out_desc.SetDataType(input_type);

  std::vector<int64_t> input_size_shape = input_desc.GetShape().GetDims();
  auto input_size_dim_num = input_size_shape.size();
  std::vector<int64_t> output_shape(input_size_shape.begin(), input_size_shape.end());
  auto output_size_num = output_size.size();
  if (output_size_num == 1) {
    for (uint64_t i = input_size_dim_num - 3; i < input_size_dim_num; ++i) {
      if (output_size[0] < 0) {
        continue;
      }
      output_shape[i] = output_size[0];
    }
  } else if (output_size_num == 3) {
    for (uint64_t i = input_size_dim_num - 3; i < input_size_dim_num; ++i) {
      auto data = output_size[i - input_size_dim_num + 3];
      if (data < 0) {
        continue;
      }
      output_shape[i] = data;
    }
  } else {
    OP_LOGE("AdaptiveAvgPool3d", "Shape of output_size is invalid");
    return GRAPH_FAILED;
  }

  out_desc.SetShape(Shape(output_shape));
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3d", "failed to update output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(AdaptiveAvgPool3d, AdaptiveAvgPool3dInferShape);
// --------- AdaptiveAvgPool3d end---------------

// --------- AdaptiveAvgPool3dGrad ---------------
CUST_IMPLEMT_VERIFIER(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradVerify) {
  auto input_grad_desc = op.GetInputDescByName("input_grad");
  auto orig_input_shape_desc = op.GetInputDescByName("orig_input_shape");
  ge::AscendString op_name;
  (void)op.GetName(op_name);

  auto orig_input_shape_dim = orig_input_shape_desc.GetShape().GetDimNum();
  if (orig_input_shape_dim != 1) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "Num Dim of orig_input_shape is invalid");
    return GRAPH_PARAM_INVALID;
  }

  auto orig_input_dim_num = orig_input_shape_desc.GetShape().GetShapeSize();
  auto input_grad_dim_num = input_grad_desc.GetShape().GetDimNum();

  if (orig_input_dim_num != static_cast<int64_t>(input_grad_dim_num)) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "Num Dim of orig_input and input_grad should be the same");
    return GRAPH_PARAM_INVALID;
  }

  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(AdaptiveAvgPool3dGradInferShape) {
  auto input_desc = op.GetInputDescByName("input_grad");
  auto orig_input_shape_desc = op.GetInputDescByName("orig_input_shape");
  TensorDesc out_desc = op.GetOutputDescByName("output_grad");
  ge::AscendString op_name;
  (void)op.GetName(op_name);

  // update data type
  DataType input_type = input_desc.GetDataType();
  out_desc.SetDataType(input_type);

  // infer shape
  Tensor orig_input_size_tensor;
  if (op.GetInputConstData("orig_input_shape", orig_input_size_tensor) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "failed to get tensor from output_size");
    return GRAPH_FAILED;
  }

  int32_t *orig_input_size_data = reinterpret_cast<int32_t *>(orig_input_size_tensor.GetData());
  if (orig_input_size_data == nullptr) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "output_size data is invalid");
    return GRAPH_PARAM_INVALID;
  }

  auto input_size_dim_num = input_desc.GetShape().GetDimNum();
  std::vector<int64_t> output_shape(input_size_dim_num);

  for (uint64_t i = 0; i < input_size_dim_num; ++i) {
    output_shape[i] = orig_input_size_data[i];
  }

  out_desc.SetShape(Shape(output_shape));
  if (op.UpdateOutputDesc("output_grad", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE("AdaptiveAvgPool3dGrad", "failed to update output desc");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradVerify);
// --------- AdaptiveAvgPool3dGrad end---------------

// --------- AdaptiveMaxPool3d---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape) {
  TensorDesc input = op.GetInputDesc(0);
  TensorDesc output_size = op.GetInputDesc(1);
  TensorDesc output = op.GetOutputDesc(0);
  TensorDesc argmax = op.GetOutputDesc(1);

  const size_t input_num_dims = input.GetShape().GetDimNum();
  const std::vector<int64_t> output_size_shape = output_size.GetShape().GetDims();
  if ((input_num_dims == 4 || input_num_dims == 5) == false) {
    OP_LOGE(TbeGetName(op), "Input dimensions must be equal to 4 or 5.");
    return GRAPH_FAILED;
  }
  if (output_size_shape.size() != 1) {
    OP_LOGE(TbeGetName(op), "output_size dim should be equal to 1.");
    return GRAPH_FAILED;
  }
  if (output_size_shape[0] != 3) {
    OP_LOGE(TbeGetName(op), "output_size shape[0] should be equal to 3.");
    return GRAPH_FAILED;
  }

  DataType input_dtype = input.GetDataType();
  Shape output_shape(UNKNOWN_SHAPE);
  output.SetDataType(input_dtype);
  output.SetShape(output_shape);
  argmax.SetDataType(DT_INT32);
  argmax.SetShape(output_shape);
  (void)op.UpdateOutputDesc("y", output);
  (void)op.UpdateOutputDesc("argmax", argmax);
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3d, AdaptiveMaxPool3dVerify);
// --------- AdaptiveMaxPool3d END---------------

// --------- AdaptiveMaxPool2dGrad---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradInferShape) {
  TensorDesc input_grad = op.GetOutputDescByName("x_grad");
  TensorDesc input = op.GetInputDescByName("x");
  DataType input_dtype = input.GetDataType();
  Shape input_shape = input.GetShape();
  input_grad.SetShape(input_shape);
  input_grad.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("x_grad", input_grad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool2dGrad, AdaptiveMaxPool2dGradVerify);
// --------- AdaptiveMaxPool2dGrad END---------------

// --------- AdaptiveMaxPool3dGrad---------------
CUST_IMPLEMT_INFERFUNC(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape) {
  TensorDesc output_grad = op.GetOutputDescByName("output_grad");
  TensorDesc input = op.GetInputDescByName("x");
  DataType input_dtype = input.GetDataType();
  Shape input_shape = input.GetShape();
  output_grad.SetShape(input_shape);
  output_grad.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("output_grad", output_grad) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_VERIFIER(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify) { return GRAPH_SUCCESS; }

CUST_INFER_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradInferShape);
CUST_VERIFY_FUNC_REG(AdaptiveMaxPool3dGrad, AdaptiveMaxPool3dGradVerify);
// --------- AdaptiveMaxPool3dGrad END---------------

// -------------------DataFormatVecPermute---------------------
IMPLEMT_INFERFUNC(DataFormatVecPermute, DataFormatVecPermuteInfer) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc(0);

  std::vector<std::pair<int64_t, int64_t>> range;
  if (x_desc->GetShapeRange(range) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  DataType y_type = x_desc->GetDataType();

  auto y_desc = op_desc->MutableOutputDesc(0);
  y_desc->SetShape(x_desc->GetShape());
  y_desc->SetShapeRange(range);
  y_desc->SetDataType(y_type);

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(DataFormatVecPermute, DataFormatVecPermuteInfer);
// -------------------DataFormatVecPermute End---------------------

// -------------------MaxPool3DWithArgmax---------------------
IMPLEMT_INFERFUNC(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape) {
  TensorDesc inputDesc = op.GetInputDescByName("x");
  auto inputShape = inputDesc.GetShape().GetDims();
  DataType inputDtype = inputDesc.GetDataType();
  TensorDesc argmaxDesc = op.GetOutputDescByName("argmax");
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  std::vector<int64_t> stridesList;
  op.GetAttr("strides", stridesList);
  std::vector<int64_t> kernelList;
  op.GetAttr("ksize", kernelList);
  int64_t dOut = (inputShape[1] - kernelList[2]) / stridesList[2] + 1;
  int64_t hOut = (inputShape[3] - kernelList[3]) / stridesList[3] + 1;
  int64_t wOut = (inputShape[4] - kernelList[4]) / stridesList[4] + 1;
  int64_t alignedBmLine;
  alignedBmLine = (wOut * hOut % 16 == 0) ? (wOut * hOut) : (((int64_t)(wOut * hOut / 16) + 1) * 16);
  std::vector<int64_t> argShapeVec;
  argShapeVec.push_back(inputShape[0]);
  argShapeVec.push_back(dOut);
  argShapeVec.push_back(inputShape[2] * kernelList[2] * kernelList[3] * kernelList[4]);
  argShapeVec.push_back((int64_t)(alignedBmLine / 16));
  argShapeVec.push_back(inputShape[5]);
  Shape argmaxShape(argShapeVec);
  argmaxDesc.SetShape(argmaxShape);
  argmaxDesc.SetDataType(DT_UINT16);
  (void)op.UpdateOutputDesc("argmax", argmaxDesc);
  std::vector<int64_t> outShapeVec{inputShape[0], dOut, inputShape[2], hOut, wOut, inputShape[5]};
  Shape outputShape(outShapeVec);
  outputDesc.SetShape(outputShape);
  outputDesc.SetDataType(inputDtype);
  (void)op.UpdateOutputDesc("y", outputDesc);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify) {
  // verify in infer func
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxInferShape);
VERIFY_FUNC_REG(MaxPool3DWithArgmax, MaxPool3DWithArgmaxVerify);
//-------------------MaxPool3DWithArgmax---------------------

//-------------------FractionalMaxPool---------------------
IMPLEMT_INFERFUNC(FractionalMaxPool, FractionalMaxPoolInfer) {
  auto tensor = op.get_input_desc_x();
  Shape input_value;

  if (WithRank(tensor, 4, input_value, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op),
      ConcatString("call WithRank failed, ", GetShapeErrMsg(0, DebugString(tensor.GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }
  std::vector<float> pooling_ratio;
  pooling_ratio = op.get_attr_pooling_ratio();
  if (pooling_ratio.size() != 4) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(
      TbeGetName(op), GetAttrSizeErrMsg("pooling_ratio", DebugString(tensor.GetShape().GetDims()), "4D"));
    return GRAPH_FAILED;
  }

  std::vector<int64_t> output_dims;
  for (int i = 0; i < 4; ++i) {
    int64_t dim = input_value.GetDim(i);
    if (dim != UNKNOWN_DIM) {
      auto real_dim = static_cast<int64_t>(dim / pooling_ratio[i]);
      if (real_dim < 0) {
        string err_msg = ConcatString("size computed for ", i, "th dim of output[y] is ", real_dim);
        AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
        return GRAPH_FAILED;
      }
      output_dims.push_back(real_dim);
    } else {
      output_dims.push_back(UNKNOWN_DIM);
    }
  }

  TensorDesc y_desc = op.GetOutputDescByName("y");
  y_desc.SetShape(Shape(output_dims));
  y_desc.SetDataType(op.GetInputDescByName("x").GetDataType());
  if (op.UpdateOutputDesc("y", y_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[y] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc row_pooling_desc = op.GetOutputDescByName("row_pooling_sequence");
  row_pooling_desc.SetShape(Shape({output_dims[1] + 1}));
  row_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("row_pooling_sequence", row_pooling_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[row_pooling_sequence] desc failed."));
    return GRAPH_FAILED;
  }

  TensorDesc col_pooling_desc = op.GetOutputDescByName("col_pooling_sequence");
  col_pooling_desc.SetShape(Shape({output_dims[2] + 1}));
  col_pooling_desc.SetDataType(DT_INT64);
  if (op.UpdateOutputDesc("col_pooling_sequence", col_pooling_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[col_pooling_sequence] desc failed."));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPool, FractionalMaxPoolInfer);
//-------------------FractionalMaxPool END---------------------

//-------------------FractionalMaxPoolGrad---------------------
IMPLEMT_INFERFUNC(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer) {
  Shape input_shape;
  if (WithRank(op.GetInputDesc(0), 4, input_shape, op) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), ConcatString("call WithRank failed, ",
                                   GetShapeErrMsg(0, DebugString(op.GetInputDesc(0).GetShape().GetDims()), "4D")));
    return GRAPH_FAILED;
  }

  auto type = op.GetInputDescByName("orig_input").GetDataType();
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(Shape(input_shape));
  output_desc.SetDataType(type);
  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("update output[y] desc failed"));
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(FractionalMaxPoolGrad, FractionalMaxPoolGradInfer);
//-------------------FractionalMaxPoolGrad END---------------------

//-------------------MaxPool3DGradWithArgMax---------------------
CUST_IMPLEMT_VERIFIER(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxVerify) {
  const size_t DIM_SIZE1 = 1;
  const size_t DIM_SIZE3 = 3;
  const size_t DIM_SIZE5 = 5;

  std::vector<int32_t> ksizeList;
  if (GRAPH_SUCCESS != op.GetAttr("ksize", ksizeList)) {
    std::string err_msg = GetInputInvalidErrMsg("ksize");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((ksizeList.size() != DIM_SIZE1) && (ksizeList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("ksizeList", ConcatString(ksizeList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> stridesList;
  if (GRAPH_SUCCESS != op.GetAttr("strides", stridesList)) {
    std::string err_msg = GetInputInvalidErrMsg("strides");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((stridesList.size() != DIM_SIZE1) && (stridesList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("stridesList", ConcatString(stridesList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> padsList;
  if (GRAPH_SUCCESS != op.GetAttr("pads", padsList)) {
    std::string err_msg = GetInputInvalidErrMsg("pads");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((padsList.size() != DIM_SIZE1) && (padsList.size() != DIM_SIZE3)) {
    string excepted_size = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3);
    std::string err_msg = GetAttrSizeErrMsg("padsList", ConcatString(padsList.size()), excepted_size);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::vector<int32_t> dilationList;
  if (GRAPH_SUCCESS != op.GetAttr("dilation", dilationList)) {
    std::string err_msg = GetInputInvalidErrMsg("dilation");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  if ((dilationList.size() != DIM_SIZE1) && (dilationList.size() != DIM_SIZE3) && (dilationList.size() != DIM_SIZE5)) {
    string excepted_value = ConcatString(DIM_SIZE1, " or ", DIM_SIZE3, " or ", DIM_SIZE5);
    std::string err_msg = GetAttrSizeErrMsg("dilationList", ConcatString(dilationList.size()), excepted_value);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  bool ceilMode = false;
  if (GRAPH_SUCCESS != op.GetAttr("ceil_mode", ceilMode)) {
    std::string err_msg = GetInputInvalidErrMsg("ceil_mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr data_format failed.");
    return GRAPH_FAILED;
  }
  if (data_format != "NCDHW") {
    OP_LOGE(TbeGetName(op).c_str(), "Attr data_format(%s) only support NCDHW.", data_format.c_str());
    return GRAPH_FAILED;
  }

  int dtype = 0;
  if (GRAPH_SUCCESS != op.GetAttr("dtype", dtype)) {
    std::string err_msg = GetInputInvalidErrMsg("dtype");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }

  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  CHECK_PTR_NULL(op_desc, "op desc", return GRAPH_FAILED);
  auto grads_desc = op_desc->MutableInputDesc("grads");
  CHECK_PTR_NULL(grads_desc, "grads desc", return GRAPH_FAILED);
  vector<int64_t> grads_shape = grads_desc->MutableShape().GetDims();
  if (grads_shape.size() != DIM_SIZE5 && !IsUnknownRankShape(grads_shape)) {
    OP_LOGE(TbeGetName(op).c_str(), "grads_shape's dim expect: %lu, but real: %lu.", DIM_SIZE5, grads_shape.size());
    return GRAPH_FAILED;
  }

  TensorDesc inputDesc = op.GetInputDescByName("x");
  vector<int64_t> inputShape = inputDesc.GetShape().GetDims();
  if (inputShape.size() != DIM_SIZE5) {
    OP_LOGE(TbeGetName(op).c_str(), "input x's dim expect: %lu, but real: %lu.", DIM_SIZE5, inputShape.size());
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

CUST_IMPLEMT_INFERFUNC(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxInferShape) {
  auto shape = op.GetInputDescByName("x").GetShape();
  auto shape_dims = shape.GetDims();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(shape);
  td.SetDataType(op.GetInputDescByName("x").GetDataType());
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxInferShape);
CUST_VERIFY_FUNC_REG(MaxPool3DGradWithArgmax, MaxPool3DGradWithArgmaxVerify);
//-------------------MaxPool3DGradWithArgMax---------------------

//-------------------NthElement---------------------
IMPLEMT_INFERFUNC(NthElement, NthElementInfer) {
  std::vector<std::string> input_infer_depends = {"n"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);
  Shape x_shape;
  auto x_tensor = op.get_input_desc_x();
  if (WithRankAtLeast(x_tensor, 1, x_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtLeast function, ", "input[x] rank must be at least 1D, but got rank[",
                   op.get_input_desc_x().GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Tensor n_tensor;
  int64_t n_dim = 0;
  if (op.GetInputConstData("n", n_tensor) != GRAPH_SUCCESS) {
    n_dim = ge::UNKNOWN_DIM;
  } else {
    if (MakeDimForScalarInput(n_tensor, n_dim, op) != GRAPH_SUCCESS) {
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), std::string("failed to call MakeDimForScalarInput function, "
                                                                    "get input n shape failed"));
      return GRAPH_FAILED;
    }
  }

  int64_t existing = x_shape.GetDimNum();
  int64_t last_input_dim = x_shape.GetDim(existing - 1);
  if ((last_input_dim != ge::UNKNOWN_DIM) && (n_dim != ge::UNKNOWN_DIM) && (last_input_dim <= n_dim)) {
    std::string err_msg =
      ConcatString("input[x] last dim value[", last_input_dim, "] must be greater than [", n_dim, "]");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  Shape output_shape;
  if (SubShape(x_shape, 0, -1, 1, output_shape, op) != GRAPH_SUCCESS) {
    std::string err_msg = ConcatString("failed to call SubShape function, input[x] shape[",
                                       DebugString(x_shape.GetDims()), "], start[0], end[-1], stride[1]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  TensorDesc y_tensor = op.GetOutputDescByName("y");
  y_tensor.SetDataType(x_tensor.GetDataType());
  y_tensor.SetShape(output_shape);
  return op.UpdateOutputDesc("y", y_tensor);
}

INFER_FUNC_REG(NthElement, NthElementInfer);
//-------------------NthElement END---------------------
}  // namespace ge
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

}  // namespace ge
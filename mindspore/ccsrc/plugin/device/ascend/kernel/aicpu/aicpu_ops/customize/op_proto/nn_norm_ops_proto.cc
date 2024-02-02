/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_proto/inc/nn_norm_ops.h"
#include "custom_op_proto/cust_nn_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ----------------KlDivLossGrad Begin-------------------
bool InferShapeAndTypeKlDivLossGrad(Operator &op, const string &input_name, const string &output_name) {
  TensorDesc output_desc = op.GetOutputDescByName(output_name.c_str());
  DataType input_dtype = op.GetInputDescByName(input_name.c_str()).GetDataType();
  Format input_format =
    static_cast<ge::Format>(ge::GetPrimaryFormat(op.GetInputDescByName(input_name.c_str()).GetFormat()));
  ge::Shape input_shape = op.GetInputDescByName(input_name.c_str()).GetShape();

  output_desc.SetShape(input_shape);
  output_desc.SetDataType(input_dtype);
  output_desc.SetFormat(input_format);
  op.UpdateOutputDesc(output_name.c_str(), output_desc);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(KlDivLossGradInferShape) {
  if (InferShapeAndTypeKlDivLossGrad(op, "input", "y")) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(TbeGetName(op).c_str(), "KL_DIV_LOSS_GRAD Infershape Failed");
  return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(KlDivLossGrad, KlDivLossGradVerify) {
  if (op.GetInputDescByName("grad").GetDataType() != op.GetInputDescByName("input").GetDataType() ||
      op.GetInputDescByName("input").GetDataType() != op.GetInputDescByName("target").GetDataType()) {
    OP_LOGE(TbeGetName(op).c_str(), "grad type is not same with input or target");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(KlDivLossGrad, KlDivLossGradInferShape);
VERIFY_FUNC_REG(KlDivLossGrad, KlDivLossGradVerify);
// ----------------KlDivLossGrad END---------------------

// ----------------MultiMarginLoss Begin-------------------
CUST_IMPLEMT_VERIFIER(MultiMarginLoss, MultiMarginLossVerify) {
  Shape shape_x = op.GetInputDescByName("x").GetShape();
  Shape shape_target = op.GetInputDescByName("target").GetShape();
  TensorDesc tensordesc_weight;
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType target_dtype = op.GetInputDescByName("target").GetDataType();
  if (x_dtype != DT_DOUBLE && x_dtype != DT_FLOAT && x_dtype != DT_FLOAT16) {
    string err_msg1 = ConcatString("dtype of input x must be double, float or float16.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (target_dtype != DT_INT64) {
    string err_msg1 = ConcatString("dtype of input target must be int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.TryGetInputDesc("weight", tensordesc_weight) == GRAPH_SUCCESS) {
    Shape shape_w = op.GetInputDescByName("weight").GetShape();
    DataType weight_dtype = op.GetInputDescByName("weight").GetDataType();
    if (weight_dtype != x_dtype) {
      string err_msg1 = ConcatString("weight should have the same dtype with x.");
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
    if (shape_w.GetDimNum() != 1) {
      string err_msg1 = ConcatString("rank of input weight must be 1, shape_weight.GetDimNum():", shape_w.GetDimNum());
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if ((shape_x.GetDimNum() != 2) || (shape_target.GetDimNum() != 1)) {
    string err_msg2 =
      ConcatString("Rank of x must be 2, rank of target must be 1, shape_x.GetDimNum():", shape_x.GetDimNum(),
                   ", shape_target.GetDimNum():", shape_target.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (shape_x.GetDim(0) != (shape_target.GetDim(0))) {
    string err_msg3 = ConcatString(
      "shape[0] of x and shape[0] of target must be "
      "the same, shape_x.GetDim(0):",
      shape_x.GetDim(0), ", shape_target.GetDim(0):", shape_target.GetDim(0));
    std::string err_msg = OtherErrMsg(err_msg3);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    OP_LOGE(TbeGetName(op).c_str(), "The val of reduction is invalid.");
    return GRAPH_FAILED;
  }
  int64_t p;
  op.GetAttr("p", p);
  if ((p != 1) && (p != 2)) {
    string err_msg4 = ConcatString("The value of p must be 1 or 2, p:", p);
    std::string err_msg = OtherErrMsg(err_msg4);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(MultiMarginLossInferShape) {
  auto shape_x = op.GetInputDescByName("x").GetShape().GetDims();
  auto shape_target = op.GetInputDescByName("target").GetShape().GetDims();
  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  Shape y_shape = Shape(shape_target);
  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction == "mean") || (reduction == "sum")) {
    Shape scalar_shape;
    Scalar(scalar_shape);
    tensordesc_output.SetShape(scalar_shape);
  }
  if (reduction == "none") {
    tensordesc_output.SetShape(y_shape);
  }
  TensorDesc input_desc = op.GetInputDescByName("x");
  tensordesc_output.SetDataType(input_desc.GetDataType());
  tensordesc_output.SetFormat(FORMAT_ND);
  op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MultiMarginLoss, MultiMarginLossInferShape);
CUST_VERIFY_FUNC_REG(MultiMarginLoss, MultiMarginLossVerify);
// ----------------MultiMarginLoss END---------------------

// ----------------------MultiMarginLossGrad------------------------
CUST_IMPLEMT_VERIFIER(MultiMarginLossGrad, MultiMarginLossGradVerify) { return GRAPH_SUCCESS; }

CUST_VERIFY_FUNC_REG(MultiMarginLossGrad, MultiMarginLossGradVerify);

IMPLEMT_COMMON_INFERFUNC(MultiMarginLossGradInferShape) {
  Shape shape_x = op.GetInputDescByName("x").GetShape();
  Shape shape_target = op.GetInputDescByName("target").GetShape();
  TensorDesc tensordesc_weight;
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_grad_dtype = op.GetInputDescByName("y_grad").GetDataType();
  DataType target_dtype = op.GetInputDescByName("target").GetDataType();
  if (y_grad_dtype != x_dtype) {
    string err_msg1 = ConcatString("dtype of input x must be the same as y_grad.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (x_dtype != DT_DOUBLE && x_dtype != DT_FLOAT && x_dtype != DT_FLOAT16) {
    string err_msg1 = ConcatString("dtype of input x must be double, float or float16");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (target_dtype != DT_INT64) {
    string err_msg1 = ConcatString("dtype of input target must be int64.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (op.TryGetInputDesc("weight", tensordesc_weight) == GRAPH_SUCCESS) {
    Shape shape_w = op.GetInputDescByName("weight").GetShape();
    DataType weight_dtype = op.GetInputDescByName("weight").GetDataType();
    if (weight_dtype != x_dtype) {
      string err_msg1 = ConcatString("weight should have the same dtype with x.");
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
    if (shape_w.GetDimNum() != 1) {
      string err_msg1 = ConcatString("rank of weight must be 1, shape_weight.GetDimNum():", shape_w.GetDimNum());
      std::string err_msg = OtherErrMsg(err_msg1);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
      return GRAPH_FAILED;
    }
  }
  if ((shape_x.GetDimNum() != 2) || (shape_target.GetDimNum() != 1)) {
    string err_msg2 =
      ConcatString("Rank of x must be 2, rank of target must be 1, shape_x.GetDimNum():", shape_x.GetDimNum(),
                   ", shape_target.GetDimNum():", shape_target.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  if (shape_x.GetDim(0) != (shape_target.GetDim(0))) {
    string err_msg3 = ConcatString(
      "shape[0] of x and shape[0] of target must be "
      "the same, shape_x.GetDim(0):",
      shape_x.GetDim(0), ", shape_target.GetDim(0):", shape_target.GetDim(0));
    std::string err_msg = OtherErrMsg(err_msg3);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  std::string reduction;
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    string expected_reduction_list = ConcatString("mean, sum, none");
    std::string err_msg = GetInputFormatNotSupportErrMsg("reduction", expected_reduction_list, reduction);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  int64_t p;
  op.GetAttr("p", p);
  if ((p != 1) && (p != 2)) {
    string err_msg4 = ConcatString("The value of p must be 1 or 2, p:", p);
    std::string err_msg = OtherErrMsg(err_msg4);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op).c_str(), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDescByName("x_grad");
  Shape x_grad_shape = Shape(shape_x);
  tensordesc_output.SetShape(x_grad_shape);
  TensorDesc input_desc = op.GetInputDescByName("x");
  tensordesc_output.SetDataType(input_desc.GetDataType());
  op.UpdateOutputDesc("x_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MultiMarginLossGrad, MultiMarginLossGradInferShape);
// ----------------------MultiMarginLossGrad END------------------------

// ----------------SparseSoftmaxCrossEntropyWithLogits Begin-------------------
IMPLEMT_VERIFIER(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsInfer) {
  auto logits_desc = op.GetInputDescByName("features");
  auto labels_desc = op.GetInputDescByName("labels");
  auto loss_desc = op.GetOutputDescByName("loss");
  auto backprop_desc = op.GetOutputDescByName("backprop");
  loss_desc.SetShape(labels_desc.GetShape());
  loss_desc.SetDataType(logits_desc.GetDataType());
  backprop_desc.SetShape(logits_desc.GetShape());
  backprop_desc.SetDataType(logits_desc.GetDataType());
  RETURN_IF_FAILURE(op.UpdateOutputDesc("loss", loss_desc));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("backprop", backprop_desc));
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(SparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsInfer);
// ----------------SparseSoftmaxCrossEntropyWithLogits End-------------------

// ----------------BatchNormGradGrad-------------------
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
// ----------------BatchNormGradGrad END-------------------

// ----------------MultilabelMarginLossGrad-------------------
IMPLEMT_COMMON_INFERFUNC(MultilabelMarginLossGradInferShape) {
  Shape shape_x = op.GetInputDescByName("x").GetShape();
  Shape shape_target = op.GetInputDescByName("target").GetShape();
  Shape shape_is_target = op.GetInputDescByName("is_target").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  DataType y_grad_dtype = op.GetInputDescByName("y_grad").GetDataType();
  DataType target_dtype = op.GetInputDescByName("target").GetDataType();
  DataType is_target_dtype = op.GetInputDescByName("is_target").GetDataType();
  size_t dims = shape_x.GetDims().size();
  if (y_grad_dtype != x_dtype) {
    string err_msg1 = ConcatString("Dtype of input x must be the same as y_grad.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (x_dtype != DT_FLOAT && x_dtype != DT_FLOAT16) {
    string err_msg1 = ConcatString("Dtype of input x must be float or float16.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (target_dtype != DT_INT32 || is_target_dtype != DT_INT32) {
    string err_msg1 = ConcatString("Dtype of input target and is_target must be int32.");
    std::string err_msg = OtherErrMsg(err_msg1);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if ((shape_x.GetDimNum() != 2) && (shape_x.GetDimNum() != 1)) {
    string err_msg2 = ConcatString("Rank of x must be 1 or 2, shape_x.GetDimNum():", shape_x.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (shape_x.GetDimNum() != shape_target.GetDimNum()) {
    string err_msg2 = ConcatString("Rank of target must be the same as x, shape_x.GetDimNum():", shape_x.GetDimNum(),
                                   ", shape_target.GetDimNum():", shape_target.GetDimNum());
    std::string err_msg = OtherErrMsg(err_msg2);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  for (size_t i = 0; i < dims; i++) {
    if (shape_x.GetDim(i) != shape_target.GetDim(i) || shape_target.GetDim(i) != shape_is_target.GetDim(i)) {
      string err_msg2 = ConcatString("Shape of x, target, is_target must be the same.");
      std::string err_msg = OtherErrMsg(err_msg2);
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  std::string reduction = "mean";
  op.GetAttr("reduction", reduction);
  if ((reduction != "mean") && (reduction != "sum") && (reduction != "none")) {
    string expected_reduction_list = ConcatString("mean, sum, none");
    std::string err_msg = GetInputFormatNotSupportErrMsg("reduction", expected_reduction_list, reduction);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  TensorDesc tensordesc_output = op.GetOutputDescByName("x_grad");
  Shape x_grad_shape = Shape(shape_x);
  tensordesc_output.SetShape(x_grad_shape);
  TensorDesc input_desc = op.GetInputDescByName("x");
  tensordesc_output.SetDataType(input_desc.GetDataType());
  op.UpdateOutputDesc("x_grad", tensordesc_output);
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(MultilabelMarginLossGrad, MultilabelMarginLossGradInferShape);
// ----------------MultilabelMarginLossGrad END-------------------
}  // namespace ge
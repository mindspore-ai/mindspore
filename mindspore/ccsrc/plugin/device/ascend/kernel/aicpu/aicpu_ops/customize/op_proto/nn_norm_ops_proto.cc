/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/nn_norm_ops.h"
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
}  // namespace ge
/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_proto/inc/matrix_calculation_ops.h"
#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ---------------Eye----------------------------
static bool CheckRows(const Operator &op, const string &attr_num_rows) {
  int64_t num_rows;
  op.GetAttr(attr_num_rows.c_str(), num_rows);
  if (num_rows <= 0) {
    return false;
  }
  return true;
}

static bool CheckBatchShape(const Operator &op, const string &attr_batch_shape) {
  const std::string opName = TbeGetName(op);
  std::vector<int64_t> batch_shape;
  op.GetAttr(attr_batch_shape.c_str(), batch_shape);
  for (size_t i = 0; i < batch_shape.size(); ++i) {
    if (batch_shape[i] <= 0) {
      OP_LOGE(opName, "the value of batch_shape less than 0.");
      return false;
    }
  }
  return true;
}

IMPLEMT_COMMON_INFERFUNC(EyeInferShape) {
  TensorDesc td = op.GetOutputDescByName("y");
  int64_t num_rows, num_columns;
  std::vector<int64_t> batch_shape;
  op.GetAttr("num_rows", num_rows);
  op.GetAttr("num_columns", num_columns);
  op.GetAttr("batch_shape", batch_shape);

  if (!CheckRows(op, "num_rows") || !CheckBatchShape(op, "batch_shape")) {
    return GRAPH_FAILED;
  }
  if (num_columns <= 0) {
    num_columns = num_rows;
  }
  std::vector<int64_t> dim_vec;
  for (size_t i = 0; i < batch_shape.size(); ++i) {
    dim_vec.push_back(batch_shape[i]);
  }
  dim_vec.push_back(num_rows);
  dim_vec.push_back(num_columns);
  td.SetShape(ge::Shape(dim_vec));
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Eye, EyeVerify) { return GRAPH_SUCCESS; }

COMMON_INFER_FUNC_REG(Eye, EyeInferShape);

VERIFY_FUNC_REG(Eye, EyeVerify);
// --------------Eye END-------------------------------

// ----------------FillDiagonal-------------------
IMPLEMT_COMMON_INFERFUNC(FillDiagonalInferShape) {
  Shape x_shape = op.GetInputDescByName("x").GetShape();
  DataType x_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(x_shape));
  td.SetDataType(x_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(FillDiagonal, FillDiagonalInferShape);
// ----------------FillDiagonal END-------------------

// ----------------MatrixLogarithm--------------------
IMPLEMT_COMMON_INFERFUNC(MatrixLogarithmInferShaper) {
  auto x_shape = op.GetInputDescByName("x").GetShape().GetDims();
  Shape input_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  int64_t size_num = static_cast<int64_t>(op.GetInputDescByName("x").GetShape().GetDimNum());
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  if (op.UpdateOutputDesc("y", td) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  if (size_num < 2) {
    string err_msg = ConcatString("the input[x] should be greater than 2, but get ", size_num, ".");
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
// 注册函数
CUST_COMMON_INFER_FUNC_REG(MatrixLogarithm, MatrixLogarithmInferShaper);
// ----------------MatrixLogarithm END-------------------

// ----------------MatrixExp-------------------
CUST_ONE_IN_ONE_OUT_INFER(MatrixExp, x, y);
// ----------------MatrixExp END-------------------

// ----------------TraceGrad Begin------------------------
IMPLEMT_COMMON_INFERFUNC(TraceGradInferShape) {
  Shape shape = op.GetInputDescByName("y_grad").GetShape();
  DataType input_dtype = op.GetInputDescByName("y_grad").GetDataType();
  std::vector<std::string> input_infer_depends = {"x_shape"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  Tensor tensor_input;
  Shape output_shape;
  if (op.GetInputConstData("x_shape", tensor_input) == GRAPH_SUCCESS) {
    MakeShapeFromShapeTensor(tensor_input, output_shape, op);
  } else {
    output_shape = Shape({UNKNOWN_RANK});
  }
  TensorDesc td = op.GetOutputDescByName("x_grad");
  td.SetShape(output_shape);
  td.SetDataType(input_dtype);
  td.SetFormat(FORMAT_ND);
  (void)op.UpdateOutputDesc("x_grad", td);
  return GRAPH_SUCCESS;
}
CUST_IMPLEMT_VERIFIER(TraceGrad, TraceGradVerify) {
  DataType x_shape_dtype = op.GetInputDescByName("x_shape").GetDataType();
  if ((x_shape_dtype != DT_INT32) && (x_shape_dtype != DT_INT64)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(TraceGrad, TraceGradInferShape);
CUST_VERIFY_FUNC_REG(TraceGrad, TraceGradVerify);
// ---------------TraceGrad END-------------------------------

// ---------------Tril--------------
IMPLEMT_COMMON_INFERFUNC(TrilInferShape) {
  Shape input_shape = op.GetInputDesc(0).GetShape();
  DataType input_dtype = op.GetInputDesc(0).GetDataType();
  TensorDesc td = op.GetOutputDesc(0);
  td.SetShape(ge::Shape(input_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(Tril, TrilVerify) { return GRAPH_SUCCESS; }

INFER_FUNC_REG(Tril, TrilInferShape);
VERIFY_FUNC_REG(Tril, TrilVerify);
// ----------------Tril END----------------

// -----------------ScatterNdUpdate-----------------
IMPLEMT_VERIFIER(ScatterNdUpdate, ScatterNdUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "var", "updates")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(ScatterNdUpdateInferShape) {
  // main part of shape infer
  ge::Shape var_shape = op.GetInputDesc("var").GetShape();
  std::vector<std::pair<int64_t, int64_t>> var_shape_range;
  op.GetInputDesc("var").GetShapeRange(var_shape_range);
  DataType input_dtype = op.GetInputDesc("var").GetDataType();
  TensorDesc td = op.GetOutputDesc("var");
  td.SetShape(var_shape);
  td.SetDataType(input_dtype);
  td.SetShapeRange(var_shape_range);
  op.UpdateOutputDesc("var", td);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateInferShape);
VERIFY_FUNC_REG(ScatterNdUpdate, ScatterNdUpdateVerify);
// -------------------ScatterNdUpdate END----------------

// -----------------TensorScatterUpdate-----------------
IMPLEMT_VERIFIER(TensorScatterUpdate, TensorScatterUpdateVerify) {
  if (!CheckTwoInputDtypeSame(op, "x", "updates")) {
    return GRAPH_FAILED;
  }
  DataType indices_dtype = op.GetInputDescByName("indices").GetDataType();
  if ((indices_dtype != DT_INT32) && (indices_dtype != DT_INT64)) {
    OP_LOGE("tensor_scatter_update", "The indices type is not int32 or int64, please check!");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(TensorScatterUpdate, TensorScatterUpdateInferShape) {
  Shape var_shape = op.GetInputDescByName("x").GetShape();
  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  TensorDesc td = op.GetOutputDescByName("y");
  td.SetShape(ge::Shape(var_shape));
  td.SetDataType(input_dtype);
  (void)op.UpdateOutputDesc("y", td);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateInferShape);
VERIFY_FUNC_REG(TensorScatterUpdate, TensorScatterUpdateVerify);
// -------------------TensorScatterUpdate END----------------

// -------------------Orgqr----------------
CUST_ONE_IN_ONE_OUT_INFER(Orgqr, x, y);
// -------------------Orgqr END----------------

// -----------------------Trace-----------------------
static bool InferShapeAndTypeTrace(Operator &op, const std::string &inputName, const std::string outputName) {
  TensorDesc vOutputDesc = op.GetOutputDescByName(outputName.c_str());
  DataType inputDtype = op.GetInputDescByName(inputName.c_str()).GetDataType();
  ge::Shape inputShape = op.GetInputDescByName(inputName.c_str()).GetShape();

  // set output tensor dim
  std::vector<int64_t> dimVec;
  ge::Shape outputShape = ge::Shape(dimVec);
  vOutputDesc.SetShape(outputShape);
  const std::vector<DataType> unchange_dtype = {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,
                                                DT_FLOAT16,    DT_INT64,     DT_UINT64};
  if (CheckInputDataType(op, "x", unchange_dtype) == true) {
    vOutputDesc.SetDataType(inputDtype);
  } else {
    vOutputDesc.SetDataType(ge::DT_INT64);
  }
  op.UpdateOutputDesc(outputName.c_str(), vOutputDesc);
  return true;
}

IMPLEMT_VERIFIER(Trace, TraceVerify) {
  AscendString op_name;
  CHECK((op.GetName(op_name) != GRAPH_SUCCESS), OP_LOGE("", "GetName failed."), return GRAPH_FAILED);
  ge::Shape shapeX = op.GetInputDescByName("x").GetShape();
  DataType dtypeX = op.GetInputDescByName("x").GetDataType();
  constexpr int64_t shapeDimsLimit = 2;
  if (shapeX.GetDims() != UNKNOWN_RANK && shapeX.GetDimNum() != shapeDimsLimit) {
    OP_LOGE(op_name.GetString(), "the input shape must be 2-D matrix.\n");
    return GRAPH_FAILED;
  }
  const std::vector<DataType> support_list = {DT_COMPLEX128, DT_COMPLEX64, DT_DOUBLE, DT_FLOAT,  DT_FLOAT16,
                                              DT_INT8,       DT_UINT8,     DT_INT16,  DT_UINT16, DT_INT32,
                                              DT_UINT32,     DT_INT64,     DT_UINT64};
  if (CheckInputDataType(op, "x", support_list) == false) {
    OP_LOGE(TbeGetName(op).c_str(), "dataType [%s] is not supported in Trace.", DTypeStr(dtypeX).c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(TraceInferShape) {
  if (InferShapeAndTypeTrace(op, "x", "y")) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(Trace, TraceInferShape);
VERIFY_FUNC_REG(Trace, TraceVerify);
// ---------------------Trace END----------------------

// ----------------IndexPut-------------------
ONE_IN_ONE_OUT_INFER(IndexPut, x1, y);
// ----------------IndexPut END-------------------
}  // namespace ge
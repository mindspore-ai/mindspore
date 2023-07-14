/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/matrix_calculation_ops.h"
#include "custom_op_proto/cust_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(OneInOneOutCommonInferShape) {
  static const int64_t input_x_idx = 0;
  static const int64_t output_y_idx = 0;
  if (OneInOneOutDynamicInfer(op, input_x_idx, {output_y_idx})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

// ----------------DiagPart-------------------
IMPLEMT_COMMON_INFERFUNC(DiagPartInferShape) {
  ge::OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  CHECK(op_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DiagPart", GetInputInvalidErrMsg("op_desc")),
        return GRAPH_FAILED);
  ge::ConstGeTensorDescPtr input_x_desc = op_desc->GetInputDescPtr(0);
  CHECK(input_x_desc == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT("DiagPart", GetInputInvalidErrMsg("x")),
        return GRAPH_FAILED);
  const GeShape &input_shape = input_x_desc->GetShape();
  const size_t input_to_output_dims_times = 2;
  size_t output_shape_len = input_shape.GetDimNum() / input_to_output_dims_times;
  ge::GeTensorDescPtr output_desc = op_desc->MutableOutputDesc(0);
  GeShape &output_shape = output_desc->MutableShape();
  DataType input_dtype = input_x_desc->GetDataType();

  if (input_shape.IsUnknownDimNum()) {
    output_desc->SetShape(input_shape);
  } else {
    output_shape.SetDimNum(output_shape_len);
    for (size_t i = 0; i < output_shape_len; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
  }
  if (input_shape.IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> shape_range;
    input_x_desc->GetShapeRange(shape_range);
    for (unsigned i = 0; i < shape_range.size(); i++) {
      if (shape_range[i].first > 0) {
        shape_range[i].first = shape_range[i].first;
      }
      if (shape_range[i].second > 0) {
        shape_range[i].second = shape_range[i].second;
      }
    }
    output_desc->SetShapeRange(shape_range);
  }
  output_desc->SetShape(output_shape);
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DiagPart, DiagPartInferShape);
// ----------------DiagPart END-------------------

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
  int64_t size_num = op.GetInputDescByName("x").GetShape().GetDimNum();
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
CUST_COMMON_INFER_FUNC_REG(MatirxExp, OneInOneOutCommonInferShape);
// ----------------MatrixExp END-------------------
}  // namespace ge
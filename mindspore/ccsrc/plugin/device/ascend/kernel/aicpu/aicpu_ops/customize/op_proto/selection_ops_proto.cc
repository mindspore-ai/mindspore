/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/selection_ops.h"
#include "custom_op_proto/cust_array_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// ----------------CumulativeLogsumexp-------------------
IMPLEMT_COMMON_INFERFUNC(CumulativeLogsumexpInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("y");
  output_desc.SetShape(op.GetInputDescByName("x").GetShape());
  output_desc.SetDataType(op.GetInputDescByName("x").GetDataType());
  op.UpdateOutputDesc("y", output_desc);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(CumulativeLogsumexp, CumulativeLogsumexpInferShape);
// ----------------CumulativeLogsumexp END-------------------

// ----------------GatherNd-------------------
bool CheckGatherNdInputIndicesSize(const Operator &op, const string &input_name) {
  auto indices_shape = OpDescUtils::GetOpDescFromOperator(op)->MutableInputDesc("indices")->GetShape();
  auto indices_shape_size = indices_shape.GetDimNum();
  int indices_last_element = indices_shape.GetDim(indices_shape_size - 1);
  int64_t indices_part{1};
  for (int i = 0; i < indices_last_element - 1; ++i) {
    indices_part *= static_cast<int64_t>(indices_shape.GetDim(i));
  }
  if (indices_part > std::numeric_limits<int>::max()) {
    OP_LOGE(TbeGetName(op).c_str(), "Indices has too many elements for int indexing");
    return false;
  }
  return true;
}

bool CheckGatherNdParamsSize(const Operator &op, int last_dim, int shape_size) {
  if (last_dim > shape_size) {
    OP_LOGE(TbeGetName(op).c_str(), "The last dim(%d) of indices must be <= params.rank(%d).", last_dim, shape_size);
    return false;
  }
  return true;
}

IMPLEMT_VERIFIER(GatherNd, GatherNdVerify) {
  if (!CheckGatherNdInputIndicesSize(op, "indices")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(GatherNdInferShape) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr output_tensor_desc = op_desc->MutableOutputDesc("y");
  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc("x")->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_indices;
  op_desc->MutableInputDesc("indices")->GetShapeRange(shape_range_indices);
  std::vector<std::pair<int64_t, int64_t>> out_range;
  auto input_params = op_desc->MutableInputDesc("x");
  auto input_indices = op_desc->MutableInputDesc("indices");
  auto params_shape = input_params->GetShape();
  auto indices_shape = input_indices->GetShape();
  auto params_shape_size = params_shape.GetDimNum();
  int indices_shape_size = indices_shape.GetDimNum();
  vector<int64_t> dim_vec;
  vector<int64_t> params_shape_vec = params_shape.GetDims();
  vector<int64_t> indices_shape_vec = indices_shape.GetDims();
  MakeUpShapeRange(params_shape_vec, shape_range_x);
  MakeUpShapeRange(indices_shape_vec, shape_range_indices);
  int indices_last_element{-2};
  if (!IsUnknownRankShape(indices_shape_vec)) {
    indices_last_element = indices_shape.GetDim(indices_shape_size - 1);
  }
  DataType params_type = input_params->GetDataType();
  if (indices_last_element == -1 || indices_last_element == -2 || IsUnknownRankShape(params_shape_vec)) {
    dim_vec.push_back(-2);
  } else if (!CheckGatherNdParamsSize(op, indices_last_element, (int)params_shape_size)) {
    return GRAPH_FAILED;
  } else {
    for (int i = 0; i < indices_shape_size - 1; ++i) {
      dim_vec.push_back(indices_shape.GetDim(i));
      if ((size_t)i < shape_range_indices.size()) {
        out_range.push_back(shape_range_indices[i]);
      }
    }
    for (size_t i = indices_last_element; i < params_shape_size; ++i) {
      dim_vec.push_back(params_shape.GetDim(i));
      if (i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }
  ge::GeShape output_shape = ge::GeShape(dim_vec);
  DataType output_dtype = params_type;
  output_tensor_desc->SetShape(output_shape);
  output_tensor_desc->SetDataType(output_dtype);
  TensorUtils::SetRealDimCnt(*output_tensor_desc, dim_vec.size());
  if (!IsUnknownRankShape(dim_vec)) {
    output_tensor_desc->SetShapeRange(out_range);
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(GatherNd, GatherNdInferShape);
VERIFY_FUNC_REG(GatherNd, GatherNdVerify);
// ----------------GatherNd End-------------------

// ----------------MaskedSelect Begin-------------------
bool InferShapeAndTypeMaskedSelect(Operator &op) {
  OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr x_input = op_desc->MutableInputDesc(0);
  GeShape x_shape = x_input->GetShape();
  GeTensorDescPtr y_desc = op_desc->MutableOutputDesc(0);
  DataType input_dtype = x_input->GetDataType();
  y_desc->SetDataType(input_dtype);
  std::vector<std::pair<int64_t, int64_t>> range;
  y_desc->SetShape(GeShape({UNKNOWN_DIM}));
  y_desc->SetOriginShape(GeShape({UNKNOWN_DIM}));
  range.emplace_back(std::make_pair(1, x_shape.GetShapeSize()));
  y_desc->SetShapeRange(range);
  return true;
}

// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(MaskedSelectInferShape) {
  if (InferShapeAndTypeMaskedSelect(op)) {
    return GRAPH_SUCCESS;
  }
  OP_LOGE(TbeGetName(op).c_str(), "The shape of output y does not match that of x1 x2.");
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(MaskedSelect, MaskedSelectInferShape);
// ----------------MaskedSelect END---------------------

// ----------------IndexFill-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(IndexFillInferShape) {
  TensorDesc v_output_desc = op.GetOutputDescByName("y");

  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  Format input_format = op.GetInputDescByName("x").GetFormat();
  // shape of output y is the same as input x
  ge::Shape shape_input = op.GetInputDescByName("x").GetShape();

  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("y", v_output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// Registered inferfunction
CUST_COMMON_INFER_FUNC_REG(IndexFill, IndexFillInferShape);
// ----------------IndexFill END-------------------
}  // namespace ge

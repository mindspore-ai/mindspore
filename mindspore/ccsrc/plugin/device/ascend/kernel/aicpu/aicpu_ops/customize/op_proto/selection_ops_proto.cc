/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/selection_ops.h"
#include "custom_op_proto/cust_array_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"
#include "utils/common_shape_fns.h"
#include "utils/vector_proto_profiling.h"

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

// ----------------SegmentSum-------------------
static bool SegmentSumShapeVerify(const Operator &op, const std::string &input_name,
                                  const std::string &segment_ids_name) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_shape_dims = op_info->MutableInputDesc("x")->MutableShape().GetDims();
  auto segment_ids_shape_dims = op_info->MutableInputDesc("segment_ids")->MutableShape().GetDims();

  return true;
}

IMPLEMT_VERIFIER(SegmentSum, SegmentSumInferShapeVerifier) {
  if (!SegmentSumShapeVerify(op, "x", "segment_ids")) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SegmentSumInferShape) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_x_desc = op_info->MutableInputDesc("x");
  auto output_desc = op_info->MutableOutputDesc("y");
  auto shape_x = input_x_desc->MutableShape().GetDims();
  auto output_shape_dims = input_x_desc->MutableShape().GetDims();
  if (output_shape_dims.empty()) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), std::string("the input[x]'s shape should not be empty."));
    return GRAPH_FAILED;
  }
  const vector<string> depend_name = {"segment_ids"};
  PREPARE_DYNAMIC_SHAPE(depend_name);
  const std::string segment_ids_name = "segment_ids";
  Tensor segment_ids;
  int64_t first_axis_dims;
  int64_t out_range_first_dims;
  if (GRAPH_SUCCESS != op.GetInputConstData(segment_ids_name.c_str(), segment_ids)) {
    OP_LOGI("segment_max", "GetInputConstData %s failed.", segment_ids_name.c_str());
    first_axis_dims = -1;
    out_range_first_dims = 0;
  } else {
    auto data_type = op.GetInputDescByName(segment_ids_name.c_str()).GetDataType();
    std::vector<int64_t> const_data;
    if (!GetConstIntData(segment_ids, data_type, const_data)) {
      std::string err_msg =
        ConcatString("failed to call GetConstIntData function ",
                     "due to invalid data type of input[segment_ids]. data_type is ", DTypeStr(data_type));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    first_axis_dims = (*std::max_element(const_data.begin(), const_data.end())) + 1;
    out_range_first_dims = first_axis_dims;
  }

  if (IsUnknownRankShape(shape_x)) {
    output_desc->SetShape(GeShape(shape_x));
  } else {
    output_shape_dims[0] = first_axis_dims;
    GeShape output_shape(output_shape_dims);
    output_desc->SetShape(GeShape(output_shape_dims));
    if (output_shape.IsUnknownShape()) {
      std::vector<std::pair<int64_t, int64_t>> shape_range_x;
      std::vector<std::pair<int64_t, int64_t>> output_shape_range;
      output_shape_range.push_back(std::pair<int64_t, int64_t>(out_range_first_dims, first_axis_dims));
      input_x_desc->GetShapeRange(shape_range_x);
      MakeUpShapeRange(output_shape_dims, shape_range_x);
      for (size_t i = 1; i < output_shape_dims.size(); i++) {
        output_shape_range.push_back(shape_range_x[i]);
      }
      output_desc->SetShapeRange(output_shape_range);
    }
  }
  DataType input_dtype = input_x_desc->GetDataType();
  output_desc->SetDataType(input_dtype);
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(SegmentSum, SegmentSumInferShape);
VERIFY_FUNC_REG(SegmentSum, SegmentSumInferShapeVerifier);
// ----------------SegmentSum END-------------------

// ----------------Select----------------------
IMPLEMT_VERIFIER(Select, SelectVerify) {
  if (!CheckTwoInputDtypeSame(op, "x1", "x2")) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op),
      string("call function CheckTwoInputDtypeSame failed, data type of input[x1] is not same as input[x2]"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(SelectInferShape) {
  if (!TwoInOneOutDynamicInferNoBroadcast(op, "x1", "x2", {"y"})) {
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(
      TbeGetName(op), string("call function TwoInOneOutDynamicInferNoBroadcast failed, update output[y] desc failed"));
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(Select, SelectInferShape);
VERIFY_FUNC_REG(Select, SelectVerify);
// ---------------Select END-----------------------

// ----------------ReverseV2 Op Begin-----------------
IMPLEMT_COMMON_INFERFUNC(ReverseV2InferShape) {
  const vector<string> depend_names = {"axis"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  if (OneInOneOutDynamicInfer(op, "x", {"y"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}
COMMON_INFER_FUNC_REG(ReverseV2, ReverseV2InferShape);
// ----------------ReverseV2 Op End-------------------

// ----------------ScatterNd-------------------
IMPLEMT_COMMON_INFERFUNC(ScatterNdInferShape) {
  vector<string> input_infer_depends = {"shape"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  auto output_desc = op_desc->MutableOutputDesc("y");
  auto shape_desc = op_desc->MutableInputDesc("shape");
  std::vector<int64_t> shape_shape = shape_desc->MutableShape().GetDims();
  std::vector<std::pair<int64_t, int64_t>> out_range;
  Tensor shape;
  std::vector<int64_t> const_data;
  if (GRAPH_SUCCESS != op.GetInputConstData("shape", shape)) {
    const_data = {-2};
  } else {
    auto data_type = shape_desc->GetDataType();
    if (!GetConstIntData(shape, data_type, const_data)) {
      USER_GE_LOGE("Invalid data type of shape, data_type is %d.", (int)data_type);
      return GRAPH_FAILED;
    }
  }

  vector<int64_t> shape_dims;
  if (shape_shape.size() == 1 && shape_shape[0] > 0 && IsUnknownRankShape(const_data)) {
    for (int64_t i = 0; i < shape_shape[0]; i++) {
      shape_dims.push_back(-1);
    }
  } else {
    for (size_t i = 0; i < (uint32_t)const_data.size(); ++i) {
      shape_dims.push_back(const_data[i]);
    }
  }

  if (IsUnknownRankShape(shape_dims)) {
    out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
  } else if (IsUnknownVec(shape_dims)) {
    for (size_t i = 0; i < shape_dims.size(); i++) {
      if (shape_dims[i] == -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
      } else {
        out_range.push_back(std::pair<int64_t, int64_t>(shape_dims[i], shape_dims[i]));
      }
    }
  }

  GeShape output_shape(shape_dims);
  output_desc->SetShape(output_shape);
  output_desc->SetShapeRange(out_range);
  output_desc->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(ScatterNd, ScatterNdInferShape);
// ----------------ScatterNd End-------------------

// ----------------OneHot---------------------------
IMPLEMT_COMMON_INFERFUNC(OneHotInferShape) {
  const vector<string> depend_names = {"depth"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  // get attr axis
  int32_t axis = -1;
  if (ge::GRAPH_SUCCESS != op.GetAttr("axis", axis)) {
    std::string err_msg = GetInputInvalidErrMsg("Get const axis failed from op of 'OneHot'!\n");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (axis < -1) {
    string correct_size = ConcatString("attr axis(", axis, ") must be >= -1");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), correct_size);
    return GRAPH_FAILED;
  }

  // get all Desc info
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  static const int64_t input_x_idx = 0;
  auto input_desc = op_info->MutableInputDesc(input_x_idx);
  const ge::GeShape &input_shape = input_desc->MutableShape();

  static const int64_t input_on_value_idx = 2;
  auto value_desc = op_info->MutableInputDesc(input_on_value_idx);
  DataType value_dtype = value_desc->GetDataType();

  // output desc and set dtype
  static const int64_t output_y_idx = 0;
  auto output_desc = op_info->MutableOutputDesc(output_y_idx);
  output_desc->SetDataType(value_dtype);

  if (input_shape.IsUnknownDimNum()) {
    // input is UnknownRank, set output UnknownRank
    OP_LOGW("OneHot", "input shape is UnknownRank, set output UnknownRank");
    output_desc->SetShape(input_shape);
    return GRAPH_SUCCESS;
  }
  // update axis to positive number
  int32_t dimnum = input_shape.GetDimNum();
  if (axis > dimnum) {
    string correct_size = ConcatString("attr axis(", axis, ") must be < ", input_shape.GetDimNum());
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), correct_size);
    return GRAPH_FAILED;
  }

  // get depth const value, depth index is 1
  int64_t depth_value = -1;
  static const int64_t input_depth_idx = 1;
  if (!ops::GetConstInt(op, input_depth_idx, depth_value)) {
    OP_LOGW("OneHot", "Get depth const tensor failed, set depth -1");
  }

  // update output shape
  ge::GeShape &output_shape = output_desc->MutableShape();
  output_shape.SetDimNum(dimnum + 1);
  if (-1 == axis) {
    for (int32_t i = 0; i < dimnum; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
    output_shape.SetDim(dimnum, depth_value);
  } else {
    while (dimnum > axis) {
      output_shape.SetDim(dimnum, input_shape.GetDim(dimnum - 1));
      dimnum--;
    }
    output_shape.SetDim(axis, depth_value);
    for (int32_t i = 0; i < axis; i++) {
      output_shape.SetDim(i, input_shape.GetDim(i));
    }
  }

  // if output shape is dynamic update output range
  if (output_shape.IsUnknownShape()) {
    output_desc->SetOriginShape(output_shape);
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);
    std::pair<int64_t, int64_t> depth_range =
      depth_value == -1 ? std::pair<int64_t, int64_t>(1, -1) : std::pair<int64_t, int64_t>(depth_value, depth_value);
    if (-1 == axis) {
      input_range.insert(input_range.end(), depth_range);
    } else {
      input_range.insert(input_range.begin() + axis, depth_range);
    }
    output_desc->SetShapeRange(input_range);
  }

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(OneHot, OneHotInferShape);
// ----------------OneHot END----------------------

// ----------------UnsortedSegmentSum-------------------
static void GetUnsortedSegmentSumConstValue(const Tensor &const_tensor, const DataType &dtype, int64_t &const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    const_data = *const_data_ptr;
  } else {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    const_data = *const_data_ptr;
  }
}

static void GetRealRange(ge::GeShape shape, std::vector<std::pair<int64_t, int64_t>> &range) {
  if (shape.IsUnknownDimNum()) {
    return;
  }
  if (range.empty()) {
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
      int64_t dim = shape.GetDim(i);
      if (dim == -1) {
        range.push_back(std::pair<int64_t, int64_t>(0, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(dim, dim));
      }
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(UnsortedSegmentSumInferShape) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  vector<string> input_infer_depends = {"num_segments"};
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends(input_infer_depends);

  Tensor input_num_segments_tensor;
  int64_t input_num_segments;
  DataType input_num_segments_dtype = op_desc->GetInputDescPtr(2)->GetDataType();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->GetInputDescPtr(0)->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_seg_id;
  op_desc->GetInputDescPtr(1)->GetShapeRange(shape_range_seg_id);

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (GRAPH_SUCCESS != op.GetInputConstData("num_segments", input_num_segments_tensor)) {
    input_num_segments = -1;
    out_range.push_back(std::pair<int64_t, int64_t>(0, -1));
  } else {
    GetUnsortedSegmentSumConstValue(input_num_segments_tensor, input_num_segments_dtype, input_num_segments);
    out_range.push_back(std::pair<int64_t, int64_t>(input_num_segments, input_num_segments));
  }

  ge::GeShape shape = op_desc->GetInputDescPtr(0)->GetShape();
  ge::GeShape shape_id = op_desc->GetInputDescPtr(1)->GetShape();

  auto output_desc = op_desc->MutableOutputDesc(0);
  ge::GeShape output_shape = output_desc->MutableShape();
  GetRealRange(shape, shape_range_x);
  GetRealRange(shape_id, shape_range_seg_id);

  int64_t dim_idsize_input = shape_id.GetDimNum();
  int64_t dim_size_input = shape.GetDimNum();
  DataType input_dtype = op_desc->GetInputDescPtr(0)->GetDataType();
  PROFILING_PROTO_AFTER_GET_SHAPE_REG();
  if (shape.IsUnknownDimNum() || shape_id.IsUnknownDimNum()) {
    if (shape.IsUnknownDimNum()) {
      output_desc->SetShape(shape);
      output_desc->SetDataType(input_dtype);
    } else {
      output_desc->SetShape(shape_id);
      output_desc->SetDataType(input_dtype);
    }
    return GRAPH_SUCCESS;
  } else if (dim_idsize_input > 1) {
    size_t rank = dim_size_input - dim_idsize_input + 1;
    size_t idx = 1;
    output_shape.SetDimNum(rank);
    output_shape.SetDim(0, input_num_segments);

    for (int64_t i = dim_idsize_input; i < dim_size_input; i++) {
      int64_t x_dim = shape.GetDim(i);
      output_shape.SetDim(idx, x_dim);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
      idx++;
    }
  } else {
    size_t rank = shape.GetDimNum();
    output_shape.SetDimNum(rank);
    output_shape.SetDim(0, input_num_segments);

    for (size_t i = 1; i < rank; i++) {
      int64_t x_dim = shape.GetDim(i);
      output_shape.SetDim(i, x_dim);
      if ((size_t)i < shape_range_x.size()) {
        out_range.push_back(shape_range_x[i]);
      }
    }
  }

  PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
  output_desc->SetShape(output_shape);
  output_desc->SetDataType(input_dtype);
  output_desc->SetShapeRange(out_range);
  PROFILING_PROTO_END();
  return GRAPH_SUCCESS;
}
COMMON_INFER_FUNC_REG(UnsortedSegmentSum, UnsortedSegmentSumInferShape);
// ----------------UnsortedSegmentSum END----------------

// ----------------Slice Op Begin ----------------------
static void GetSliceConstValue(const Tensor &const_tensor, const DataType &dtype, std::vector<int64_t> &const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t *const_data_ptr = (int32_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
    }
  } else {
    int64_t *const_data_ptr = (int64_t *)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
    }
  }
}

IMPLEMT_COMMON_INFERFUNC(SliceInferShape) {
  const vector<string> depend_names = {"offsets", "size"};
  PREPARE_DYNAMIC_SHAPE(depend_names);

  Tensor input_begin_tensor;
  Tensor input_size_tensor;
  auto input_desc = op.GetInputDescByName("x");
  const Shape shape = input_desc.GetShape();
  DataType input_dtype = input_desc.GetDataType();
  std::vector<int64_t> input_begin;
  std::vector<int64_t> input_size;

  bool has_offsets = true;
  if (op.GetInputConstData("offsets", input_begin_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get offsets failed.");
    has_offsets = false;
  } else {
    DataType input_begin_dtype = op.GetInputDescByName("offsets").GetDataType();
    GetSliceConstValue(input_begin_tensor, input_begin_dtype, input_begin);
  }

  bool has_size = true;
  if (op.GetInputConstData("size", input_size_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get size failed.");
    has_size = false;
  } else {
    DataType input_size_dtype = op.GetInputDescByName("size").GetDataType();
    GetSliceConstValue(input_size_tensor, input_size_dtype, input_size);
  }

  bool is_unknown_rank = !has_size && !has_offsets && shape.GetDims() == UNKNOWN_RANK;
  if (is_unknown_rank) {
    TensorDesc output_desc = op.GetOutputDescByName("y");
    output_desc.SetDataType(input_dtype);
    Shape outputShape(UNKNOWN_RANK);
    output_desc.SetShape(outputShape);
    OP_LOGD(TbeGetName(op).c_str(), "output_shape:%s", to_string(output_desc.GetShape()).c_str());
    (void)op.UpdateOutputDesc("y", output_desc);
    return GRAPH_SUCCESS;
  }

  auto shape_dims = shape.GetDims();
  if (shape.GetDims() == UNKNOWN_RANK) {
    shape_dims.assign(std::max(input_begin.size(), input_size.size()), -1);
  }

  size_t dimNum = shape_dims.size();
  std::vector<int64_t> outputList;

  vector<pair<int64_t, int64_t>> ranges;
  input_desc.GetShapeRange(ranges);
  if (ranges.empty()) {
    MakeUpShapeRange(shape_dims, ranges);
  }

  if (ranges.size() < dimNum) {
    OP_LOGE(TbeGetName(op).c_str(), "ranges.size is:%ld, smaller than dimNum, dimNum is %ld.", ranges.size(), dimNum);
    return GRAPH_FAILED;
  }

  if (!has_size && !has_offsets) {
    for (size_t i = 0; i < dimNum; ++i) {
      outputList.push_back(-1);
      ranges[i].first = 0;
    }
  } else if (!has_offsets && has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      if (input_size[i] == -1) {
        outputList.push_back(-1);
        ranges[i].first = 0;
      } else {
        outputList.push_back(input_size[i]);
        ranges[i].first = input_size[i];
        ranges[i].second = input_size[i];
      }
    }
  } else if (has_offsets && !has_size) {
    for (size_t i = 0; i < dimNum; ++i) {
      outputList.push_back(-1);
      ranges[i].first = 0;
      if (ranges[i].second != -1) {
        if (shape_dims[i] != -1) {
          ranges[i].second = std::min(ranges[i].second, shape_dims[i]);
        }
        ranges[i].second -= input_begin[i];
      }
    }
  } else {
    for (size_t i = 0; i < dimNum; ++i) {
      if (input_size[i] == -1) {
        if (shape_dims[i] == -1) {
          outputList.push_back(-1);
        } else {
          outputList.push_back(shape_dims[i] - input_begin[i]);
        }

        ranges[i].first = 0;
      } else {
        outputList.push_back(input_size[i]);
        ranges[i].first = input_size[i];
        ranges[i].second = input_size[i];
      }
    }
  }

  TensorDesc tensordesc_output = op.GetOutputDescByName("y");
  tensordesc_output.SetDataType(input_dtype);
  if (IsUnKnownShape(outputList)) {
    tensordesc_output.SetShapeRange(ranges);
    OP_LOGD(TbeGetName(op).c_str(), "output_ranges:%s", to_string(ranges).c_str());
  }

  Shape outputShape(outputList);
  tensordesc_output.SetShape(outputShape);
  OP_LOGD(TbeGetName(op).c_str(), "output_ranges:%s", to_string(ranges).c_str());
  OP_LOGD(TbeGetName(op).c_str(), "offset:%s", to_string(input_begin).c_str());
  OP_LOGD(TbeGetName(op).c_str(), "size:%s", to_string(input_size).c_str());
  OP_LOGD(TbeGetName(op).c_str(), "output_shape:%s", to_string(tensordesc_output.GetShape()).c_str());
  (void)op.UpdateOutputDesc("y", tensordesc_output);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Slice, SliceInferShape);
// ----------------Slice Op END ----------------------
}  // namespace ge

/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/transformation_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"
#include "utils/common_shape_fns.h"
#include "utils/vector_proto_profiling.h"

namespace ge {
// ------------------DepthToSpace------------------
static bool VerifyDepthToSpaceInputShape(const Operator &op, const int64_t &block_size,
                                         const std::vector<int64_t> &input_dims, const std::string &data_format) {
  bool check_format = (data_format == "NCHW" || data_format == "NHWC");
  if (check_format && !IsUnknown(input_dims)) {
    int64_t c_dim = 3;
    c_dim = data_format == "NHWC" ? 3 : 1;
    auto mod_res = input_dims[c_dim] % (block_size * block_size);
    if (mod_res != 0) {
      OP_LOGE(TbeGetName(op),
              "Depth size must be divisible by block_size * block_size,"
              "but got depth[%ld], block_size[%ld], data_format[%s]",
              input_dims[c_dim], block_size, data_format.c_str());
      return false;
    }
  }
  return true;
}

IMPLEMT_VERIFIER(DepthToSpace, DepthToSpaceVerify) {
  // verify input shape size
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    std::string err_msg = GetAttrValueErrMsg("input_dims", std::to_string(input_dims.size()), ConcatString(">=4"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // verify block size
  int64_t block_size;
  if (op.GetAttr("block_size", block_size) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (block_size < 2) {
    std::string err_msg = GetAttrValueErrMsg("block_size", std::to_string(block_size), ConcatString("=<2"));
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // verify mode
  std::string mode;
  if (op.GetAttr("mode", mode) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("mode");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (mode != "DCR" && mode != "CRD") {
    string expected_format_list = ConcatString("DCR, CRD");
    std::string err_msg = GetAttrValueErrMsg("mode", mode, expected_format_list);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // verify data_format
  std::string data_format;
  if (op.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
    std::string err_msg = GetInputInvalidErrMsg("data_format");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (data_format != "NHWC" && data_format != "NCHW" && data_format != "NC1HWC0") {
    string expected_format_list = ConcatString("NHWC, NCHW, NC1HWC0");
    std::string err_msg = GetInputFormatNotSupportErrMsg("data_format", expected_format_list, data_format);
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  // verify input shape
  bool check_input_shape = true;
  check_input_shape = VerifyDepthToSpaceInputShape(op, block_size, input_dims, data_format);
  if (!check_input_shape) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

IMPLEMT_COMMON_INFERFUNC(DepthToSpaceInfer) {
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc("x");
  auto input_dims = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc->GetFormat()));

  auto output_desc = op_info->MutableOutputDesc("y");
  output_desc->SetDataType(input_dtype);

  // get attr block_size
  int64_t block_size;
  if (GRAPH_SUCCESS != op.GetAttr("block_size", block_size)) {
    std::string err_msg = GetInputInvalidErrMsg("block_size");
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  // not dynamic case, only set shape
  if (!IsUnknown(input_dims)) {
    std::vector<int64_t> output_dims;
    output_dims.push_back(input_dims[0]);
    if (input_format == FORMAT_NCHW) {
      output_dims.push_back(input_dims[1] / block_size / block_size);
      output_dims.push_back(input_dims[2] * block_size);
      output_dims.push_back(input_dims[3] * block_size);
    } else {  // without NCHW all other format set as NHWC
      output_dims.push_back(input_dims[1] * block_size);
      output_dims.push_back(input_dims[2] * block_size);
      output_dims.push_back(input_dims[3] / block_size / block_size);
    }
    output_desc->SetShape(GeShape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc->SetShape(GeShape(input_dims));
    OP_LOGW(TbeGetName(op).c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_dims, input_range);

  // infer output shape and range
  std::vector<int64_t> output_dims;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  output_dims.push_back(input_dims[0]);
  output_range.push_back(input_range[0]);
  int64_t dim;
  int64_t range_min;
  int64_t range_max;
  if (input_format == FORMAT_NCHW) {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] / block_size / block_size;
    range_min = input_range[1].first / block_size / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second / block_size / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size;
    range_min = input_range[2].first * block_size;
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] * block_size;
    range_min = input_range[3].first * block_size;
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  } else {
    dim = input_dims[1] == -1 ? -1 : input_dims[1] * block_size;
    range_min = input_range[1].first * block_size;
    range_max = input_range[1].second == -1 ? -1 : input_range[1].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[2] == -1 ? -1 : input_dims[2] * block_size;
    range_min = input_range[2].first * block_size;
    range_max = input_range[2].second == -1 ? -1 : input_range[2].second * block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
    dim = input_dims[3] == -1 ? -1 : input_dims[3] / block_size / block_size;
    range_min = input_range[3].first / block_size / block_size;
    range_min = std::max(int64_t(range_min), int64_t(1));
    range_max = input_range[3].second == -1 ? -1 : input_range[3].second / block_size / block_size;
    output_dims.push_back(dim);
    output_range.push_back(std::pair<int64_t, int64_t>(range_min, range_max));
  }

  output_desc->SetShape(GeShape(output_dims));
  output_desc->SetShapeRange(output_range);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DepthToSpace, DepthToSpaceInfer);
VERIFY_FUNC_REG(DepthToSpace, DepthToSpaceVerify);
// -------------------DepthToSpace END-----------------

// -------------------Transpose-----------------
static graphStatus TransposeCommonInferShape(const std::vector<int64_t> &perm_list, Operator &op) {
  PROFILING_PROTO_INIT(TbeGetName(op).c_str());
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  const int64_t input_x_idx = 0;
  auto input_desc = op_info->MutableInputDesc(input_x_idx);
  const int64_t output_y_idx = 0;
  auto output_desc = op_info->MutableOutputDesc(output_y_idx);

  auto input_dtype = input_desc->GetDataType();
  const GeShape &input_ge_shape = input_desc->MutableShape();

  int64_t input_shape_len = input_ge_shape.GetDimNum();

  PROFILING_PROTO_AFTER_GET_SHAPE_REG();

  if (IsUnknownRankShape(input_ge_shape)) {
    // UnknownRankShape, set shape is -1, -1, -1....
    std::vector<int64_t> out_vec(perm_list.size(), -1);
    output_desc->SetShape(GeShape(out_vec));
    output_desc->SetDataType(input_dtype);
    return GRAPH_SUCCESS;
  }

  // infer the shape
  GeShape &output_ge_shape = output_desc->MutableShape();
  output_ge_shape.SetDimNum(input_shape_len);
  for (size_t i = 0; i < perm_list.size(); ++i) {
    // verify perm_list begin
    int64_t perm_value = perm_list[i] < 0 ? perm_list[i] + input_shape_len : perm_list[i];
    if (perm_value >= input_shape_len) {
      std::string err_msg = GetAttrValueErrMsg("perm", ConcatString(perm_value),
                                               ConcatString("less than input shape size[", input_shape_len, "]"));
      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
    // verify perm_list end

    // set the output shape
    output_ge_shape.SetDim(i, input_ge_shape.GetDim(perm_value));
  }
  PROFILING_PROTO_AFTER_INFER_SHAPE_REG();
  // set output dtype as the same with input x
  output_desc->SetDataType(input_dtype);

  // infer the range, when need
  if (output_ge_shape.IsUnknownShape()) {
    output_desc->SetOriginShape(output_ge_shape);
    std::vector<std::pair<int64_t, int64_t>> input_range;
    std::vector<std::pair<int64_t, int64_t>> output_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_ge_shape, input_range);
    for (size_t i = 0; i < perm_list.size(); ++i) {
      output_range.push_back(input_range[perm_list[i]]);
    }
    output_desc->SetShapeRange(output_range);
    return GRAPH_SUCCESS;
  }
  PROFILING_PROTO_END();
  return GRAPH_SUCCESS;
}
IMPLEMT_COMMON_INFERFUNC(TransposeInferShape) {
  const vector<string> depend_names = {"perm"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);

  bool perm_done = true;
  std::vector<int64_t> perm_list;
  static const int64_t perm_input_index = 1;
  if (!(ops::GetConstIntData(op, perm_input_index, perm_list))) {
    perm_done = false;
    OP_LOGW(TbeGetName(op), "Get Const perm value failed ");
  }

  // perm is const node , will do infer use function TransposeCommonInferShape
  if (perm_done) {
    if (GRAPH_SUCCESS != TransposeCommonInferShape(perm_list, op)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // perm is not const node, infer for aicpu
  static const int64_t x_input_index = 0;
  static const int64_t y_output_index = 0;
  auto input_desc = op_desc->MutableInputDesc(x_input_index);
  auto input_shape = input_desc->MutableShape().GetDims();
  auto input_dtype = input_desc->GetDataType();
  auto output_desc = op_desc->MutableOutputDesc(y_output_index);

  // set output dtype as the same with input x
  output_desc->SetDataType(input_dtype);

  if (IsUnknownRankShape(input_shape)) {
    auto perm_desc = op_desc->MutableInputDesc("perm");
    auto perm_shape = perm_desc->MutableShape().GetDims();
    if (IsUnknown(perm_shape)) {
      // set output is -2 UnknownRank
      OP_LOGW(TbeGetName(op), "the output will be set to -2");
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetOriginShape(GeShape(input_shape));
      return GRAPH_SUCCESS;
    }

    // pert is not dynamic shape, will update the input shape
    if (perm_shape.empty()) {
      perm_shape.push_back(1);
    }
    input_shape.clear();
    for (auto i = 0; i < perm_shape[0]; ++i) {
      input_shape.push_back(-1);
    }
  }

  // begin to infer shape and range
  std::vector<std::pair<int64_t, int64_t>> input_range;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  vector<int64_t> out_vec;
  input_desc->GetShapeRange(input_range);
  MakeUpShapeRange(input_shape, input_range);

  int64_t range_first = input_range[0].first;
  int64_t range_second = input_range[0].second;

  for (size_t i = 0; i < input_range.size(); ++i) {
    // all range is the same and get the shape range
    range_first = std::min(range_first, input_range[i].first);
    range_second =
      (range_second == -1 || input_range[i].second == -1) ? -1 : std::max(range_second, input_range[i].second);
  }

  for (size_t i = 0; i < input_range.size(); ++i) {
    out_vec.push_back(-1);
    output_range.push_back(std::pair<int64_t, int64_t>(range_first, range_second));
  }
  output_desc->SetShape(GeShape(out_vec));
  output_desc->SetOriginShape(GeShape(out_vec));
  output_desc->SetShapeRange(output_range);

  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(Transpose, TransposeInferShape);
// -------------------Transpose END-----------------
}  // namespace ge
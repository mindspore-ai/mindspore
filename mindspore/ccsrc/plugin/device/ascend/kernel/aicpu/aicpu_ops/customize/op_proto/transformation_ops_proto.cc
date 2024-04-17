/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <numeric>

#include "op_proto/inc/transformation_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/op_const.h"
#include "utils/common_shape_fns.h"
#include "utils/vector_proto_profiling.h"

namespace ge {
namespace {
constexpr int kChannelDim = 3;
}
// ------------------DepthToSpace------------------
static bool VerifyDepthToSpaceInputShape(const Operator &op, const int64_t &block_size,
                                         const std::vector<int64_t> &input_dims, const std::string &data_format) {
  bool check_format = (data_format == "NCHW" || data_format == "NHWC");
  if (check_format && !IsUnknown(input_dims)) {
    int64_t c_dim = 3;
    c_dim = data_format == "NHWC" ? kChannelDim : 1;
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
  auto input_desc = op.GetInputDesc("x");
  auto input_dims = input_desc.GetShape().GetDims();
  if (!IsUnknownRankShape(input_dims) && (input_dims.size() < 4)) {
    std::string err_msg = GetAttrValueErrMsg("input_dims", std::to_string(input_dims.size()), ConcatString(">=4"));
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
  auto input_desc = op.GetInputDesc("x");
  auto input_dims = input_desc.GetShape().GetDims();
  auto input_dtype = input_desc.GetDataType();
  auto input_format = static_cast<ge::Format>(ge::GetPrimaryFormat(input_desc.GetFormat()));

  auto output_desc = op.GetOutputDesc("y");
  output_desc.SetDataType(input_dtype);

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
    output_desc.SetShape(Shape(output_dims));
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -2, output is -2
  if (IsUnknownRankShape(input_dims)) {
    output_desc.SetShape(Shape(input_dims));
    OP_LOGW(TbeGetName(op).c_str(), "input shape is UnknownRank, set output is UnknownRank.");
    return GRAPH_SUCCESS;
  }

  // dynamic case, input shape is -1, output is -1
  std::vector<std::pair<int64_t, int64_t>> input_range;
  input_desc.GetShapeRange(input_range);
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

  output_desc.SetShape(Shape(output_dims));
  output_desc.SetShapeRange(output_range);
  op.UpdateOutputDesc("y", output_desc);
  return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DepthToSpace, DepthToSpaceInfer);
VERIFY_FUNC_REG(DepthToSpace, DepthToSpaceVerify);
// -------------------DepthToSpace END-----------------

// ----------------Flatten-----------------------
IMPLEMT_INFERFUNC(Flatten, FlattenInfer) {
  auto input_desc = op.GetInputDescByName("x");
  auto input_shape = input_desc.GetShape().GetDims();
  auto input_type = input_desc.GetDataType();
  Shape output_shape({UNKNOWN_DIM, UNKNOWN_DIM});
  if (!IsUnknown(input_shape)) {
    auto batchsize = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int64_t>());
    output_shape.SetDim(0, input_shape[0]);
    output_shape.SetDim(1, batchsize);
  }
  auto out_desc = op.GetOutputDescByName("y");
  out_desc.SetShape(output_shape);
  out_desc.SetDataType(input_type);
  return op.UpdateOutputDesc("y", out_desc);
}
INFER_FUNC_REG(Flatten, FlattenInfer);
// ----------------Flatten END-----------------------
}  // namespace ge

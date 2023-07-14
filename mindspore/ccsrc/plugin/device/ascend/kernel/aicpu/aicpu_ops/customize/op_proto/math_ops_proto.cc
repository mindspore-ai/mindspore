/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inc/ops/math_ops.h"
#include "inc/ops/ragged_math_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"
#include "utils/reduce_infer_util.h"

namespace ge {
// ----------------ComplexAbs-------------------
IMPLEMT_INFERFUNC(ComplexAbs, ComplexAbsInfer) {
  TensorDesc x_desc = op.GetInputDescByName("x");
  DataType x_type = x_desc.GetDataType();
  DataType out_type;
  switch (x_type) {
    case DT_COMPLEX64:
      out_type = DT_FLOAT;
      break;
    case DT_COMPLEX128:
      out_type = DT_DOUBLE;
      break;
    default:
      OP_LOGE("ComplexAbs", "Invalid input dtype: %s", DTypeStr(x_type).c_str());
      return GRAPH_FAILED;
  }
  x_desc.SetDataType(out_type);
  return op.UpdateOutputDesc("y", x_desc);
}

INFER_FUNC_REG(ComplexAbs, ComplexAbsInfer);
// ----------------ComplexAbs End-------------------

// ----------------ComplexAbs-------------------
IMPLEMT_INFERFUNC(Complex, ComplexInfer) {
  bool is_dynamic_output = true;
  if (!InferShapeAndTypeTwoInOneOutBroadcast(op, 0, 1, 0, is_dynamic_output)) {
    return GRAPH_FAILED;
  }
  TensorDesc x_desc = op.GetInputDescByName("real");
  DataType x_type = x_desc.GetDataType();
  DataType out_type;
  switch (x_type) {
    case DT_FLOAT:
      out_type = DT_COMPLEX64;
      break;
    case DT_DOUBLE:
      out_type = DT_COMPLEX128;
      break;
    default:
      OP_LOGE("Complex", "Invalid input dtype: %s", DTypeStr(x_type).c_str());
      return GRAPH_FAILED;
  }
  TensorDesc out_desc = op.GetOutputDescByName("out");
  out_desc.SetDataType(out_type);
  return op.UpdateOutputDesc("out", out_desc);
}
INFER_FUNC_REG(Complex, ComplexInfer);
// ----------------ComplexAbs-------------------

// ----------------IsNan-------------------
IMPLEMT_INFERFUNC(IsNan, IsNanInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsNan, IsNanInfer);
// ----------------IsNan End-------------------

// ----------------NextAfter-------------------
IMPLEMT_INFERFUNC(NextAfter, NextAfterInfer) {
  Shape x_shape = op.GetInputDescByName("x1").GetShape();
  Shape y_shape = op.GetInputDescByName("x2").GetShape();
  TensorDesc out_desc = op.GetOutputDescByName("output");
  DataType x_type = op.GetInputDescByName("x1").GetDataType();
  DataType y_type = op.GetInputDescByName("x2").GetDataType();
  if (x_type != y_type) {
    OP_LOGE(TbeGetName(op).c_str(), "the type of x1 is different from that of x2!");
    return GRAPH_FAILED;
  }

  out_desc.SetDataType(x_type);
  if ((!RankKnown(x_shape)) || (!RankKnown(y_shape))) {
    Shape out_shape(UNKNOWN_SHAPE);
    out_desc.SetShape(out_shape);
    if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "update output failed");
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  const size_t rank_x = x_shape.GetDimNum();
  const size_t rank_y = y_shape.GetDimNum();
  const size_t rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, zip together x_shape and y_shape
  // and pad with 1 to make them the same length.
  std::vector<int64_t> dims;
  int64_t dim_one = 1;
  if (rank_x != rank_y) {
    OP_LOGI(TbeGetName(op).c_str(), "x1 shape is not equal to x2 shape!");
    dim_one = 1;
  }
  for (size_t i = 0; i < rank_out; ++i) {
    int64_t dim_x;
    if (i < (rank_out - rank_x)) {
      dim_x = dim_one;
    } else {
      // rank_out = rank_x or i >= rank_y - rank_x.
      for (size_t j = 0; j < x_shape.GetDimNum(); ++j) {
        if (x_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_x = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_x)) < 0) {
        dim_x = x_shape.GetDim(rank_x + i - (rank_out - rank_x));
      } else {
        dim_x = x_shape.GetDim(i - (rank_out - rank_x));
      }
    }

    const bool dim_y_is_one = (i < (rank_out - rank_y));
    int64_t dim_y;
    if (dim_y_is_one) {
      dim_y = dim_one;
    } else {
      // rank_out = rank_y or i >= rank_x - rank_y.
      for (size_t j = 0; j < y_shape.GetDimNum(); ++j) {
        if (y_shape.GetDim(j) == UNKNOWN_DIM) {
          dim_y = UNKNOWN_DIM;
          break;
        }
      }
      if ((i - (rank_out - rank_y)) < 0) {
        dim_y = y_shape.GetDim(rank_y + i - (rank_out - rank_y));
      } else {
        dim_y = y_shape.GetDim(i - (rank_out - rank_y));
      }
    }

    if ((dim_x == UNKNOWN_DIM) || (dim_y == UNKNOWN_DIM)) {
      /* One or both dimensions is unknown.
       * If either dimension is greater than 1, assume that the program is
       * correct, and the other dimension will be broadcast to match it.
       * For shape inference, if eliminate the shape checks
       * in this code, assert that the unknown dim is either 1
       * or the same as the known dim.
       * If either dimension is 1, the other dimension is the output.
       */
      if (dim_x > 1) {
        dims.push_back(dim_x);
      } else if (dim_y > 1) {
        dims.push_back(dim_y);
      } else if (dim_x == 1) {
        dims.push_back(dim_y);
      } else if (dim_y == 1) {
        dims.push_back(dim_x);
      } else if (dim_x == dim_y) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(UNKNOWN_DIM);
      }
    } else if ((dim_x == 1) || (dim_y == 1)) {
      // dim_x is dim_one or dim_y is dim_one.
      if ((dim_x == 1) && (!dim_y_is_one)) {
        // broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        if (dim_y == 1) {
          // broadcast dim_y to dim_x.
          dims.push_back(dim_x);
        }
      }
    } else {
      int64_t dim;
      if (Merge(dim_x, dim_y, dim) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
      dims.push_back(dim);
    }
  }
  Shape out_shape(dims);
  out_desc.SetShape(out_shape);
  if (op.UpdateOutputDesc("output", out_desc) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "update output failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
INFER_FUNC_REG(NextAfter, NextAfterInfer);
// ----------------NextAfter End-------------------

// ----------------IsInf------------------------
IMPLEMT_INFERFUNC(IsInf, IsInfInfer) {
  TensorDesc out_desc = op.GetOutputDescByName("y");
  out_desc.SetDataType(DT_BOOL);
  if (op.UpdateOutputDesc("y", out_desc) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("update output[y] failed."));
    return GRAPH_FAILED;
  }
  return UnchangedShape(op, "x", "y");
}

INFER_FUNC_REG(IsInf, IsInfInfer);
// ----------------IsInf END------------------------

// ----------------ReduceOp-------------------
static bool InferReduceShapeProcess(const Operator &op, const int64_t input_x_idx, const int64_t output_y_idx,
                                    const int64_t input_axes_idx) {
  bool keep_dims = false;
  op.GetAttr("keep_dims", keep_dims);
  reduce_ops::CommonReduceInferWithInputAxes(op, input_x_idx, output_y_idx, input_axes_idx, keep_dims);
  return true;
}

IMPLEMT_COMMON_INFERFUNC(TypicalReduceInferShape) {
  OP_LOGD(TbeGetName(op), "Enter %s InferShape", TbeGetOpType(op).c_str());
  const int64_t input_x_idx = 0;
  const int64_t output_y_idx = 0;
  const int64_t input_axes_idx = 1;
  if (InferReduceShapeProcess(op, input_x_idx, output_y_idx, input_axes_idx)) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(ReduceSum, TypicalReduceInferShape);
// ----------------ReduceOp END-------------------

// ----------------RaggedRange-------------------
IMPLEMT_INFERFUNC(RaggedRange, RaggedRangeInfer) {
  Shape starts;
  Shape limits;
  Shape deltas;
  if (WithRankAtMost(op.GetInputDesc(0), 1, starts, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, ", "input[starts] rank must be at most 1D, got rank[",
                   op.GetInputDesc(0).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(1), 1, limits, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, ", "input[limits] rank must be at most 1D, got rank[",
                   op.GetInputDesc(1).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  if (WithRankAtMost(op.GetInputDesc(2), 1, deltas, op) != GRAPH_SUCCESS) {
    std::string err_msg =
      ConcatString("failed to call WithRankAtMost function, input[deltas] ", "rank must be at most 1D, got rank[",
                   op.GetInputDesc(2).GetShape().GetDimNum(), "]");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }

  int64_t dim = ge::UNKNOWN_DIM;
  int64_t starts_dim = starts.GetDim(0);
  int64_t limits_dim = limits.GetDim(0);
  int64_t deltas_dim = deltas.GetDim(0);
  if (op.GetInputDesc(0).GetShape().GetDimNum() == 1) {
    if (Merge(starts_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", starts_dim,
                                         "] of input[starts] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(1).GetShape().GetDimNum() == 1) {
    if (Merge(limits_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", limits_dim,
                                         "] of input[limits] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }
  if (op.GetInputDesc(2).GetShape().GetDimNum() == 1) {
    if (Merge(deltas_dim, dim, dim) != GRAPH_SUCCESS) {
      std::string err_msg = ConcatString("failed to call Merge function, the 0th dim[", deltas_dim,
                                         "] of input[deltas] not equal UNKNOWN_DIM");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }
  }

  int64_t rt_nested_splits_dim = ge::UNKNOWN_DIM;
  if (dim != ge::UNKNOWN_DIM) {
    rt_nested_splits_dim = dim + 1;
  } else if (op.GetInputDesc(0).GetShape().GetDimNum() == 0 && op.GetInputDesc(1).GetShape().GetDimNum() == 0 &&
             op.GetInputDesc(2).GetShape().GetDimNum() == 0) {
    rt_nested_splits_dim = 2;
  }

  DataType Tsplits_type;
  if (op.GetAttr("Tsplits", Tsplits_type) != GRAPH_SUCCESS) {
    AICPU_INFER_SHAPE_INNER_ERR_REPORT(TbeGetName(op), string("get attr[Tsplits] failed"));
    return GRAPH_FAILED;
  }
  TensorDesc rt_nested_desc = op.GetOutputDescByName("rt_nested_splits");
  rt_nested_desc.SetShape(Shape({rt_nested_splits_dim}));
  rt_nested_desc.SetDataType(Tsplits_type);
  (void)op.UpdateOutputDesc("rt_nested_splits", rt_nested_desc);

  DataType T_type = op.GetInputDescByName("starts").GetDataType();
  std::vector<int64_t> unknow_dim_vec(1, UNKNOWN_DIM);
  TensorDesc dense_desc = op.GetOutputDescByName("rt_dense_values");
  dense_desc.SetShape(Shape(unknow_dim_vec));
  dense_desc.SetDataType(T_type);
  (void)op.UpdateOutputDesc("rt_dense_values", dense_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(RaggedRange, RaggedRangeInfer);
// ----------------RaggedRange END-------------------
}  // namespace ge

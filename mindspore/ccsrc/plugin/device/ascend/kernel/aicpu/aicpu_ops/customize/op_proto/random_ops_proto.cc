/**
 * Copyright (c) 2023 Huawei Technologies Co., Ltd.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "op_proto/inc/random_ops.h"
#include "op_proto/inc/math_ops.h"
#include "op_proto/inc/stateful_random_ops.h"
#include "custom_op_proto/cust_random_ops.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
namespace {
template <typename T>
static void RandomOpCalcDims(const Tensor &data, std::vector<int64_t> &vec_dim) {
  int32_t size = data.GetSize() / static_cast<int64_t>(sizeof(T));
  for (int32_t i = 0; i < size; i++) {
    T dim = *(reinterpret_cast<const T *>(data.GetData()) + i);
    vec_dim.push_back(dim);
  }
}
}  // namespace

ONE_IN_ONE_OUT_INFER(ShuffleChannel, x, y);

IMPLEMT_COMMON_INFERFUNC(InputShapeAttrDtypeInfer) {
  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape_tensor error.");
    return GRAPH_FAILED;
  }
  if (MakeShapeFromShapeTensor(shape_tensor, shape, op) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get shape error.");
    return GRAPH_FAILED;
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  outputDesc.SetDataType(dtype);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

COMMON_INFER_FUNC_REG(NonDeterministicInts, InputShapeAttrDtypeInfer);
CUST_COMMON_INFER_FUNC_REG(StandardLaplace, InputShapeAttrDtypeInfer);

// ----------------LogNormalReverse-------------------
// Obtains the processing function of the output tensor description.
IMPLEMT_COMMON_INFERFUNC(LogNormalReverseInferShape) {
  TensorDesc v_output_desc = op.GetOutputDescByName("output");

  DataType input_dtype = op.GetInputDescByName("input").GetDataType();
  Format input_format = op.GetInputDescByName("input").GetFormat();
  ge::Shape shape_input = op.GetInputDescByName("input").GetShape();

  v_output_desc.SetShape(shape_input);
  v_output_desc.SetDataType(input_dtype);
  v_output_desc.SetFormat(input_format);

  if (op.UpdateOutputDesc("output", v_output_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(LogNormalReverse, LogNormalReverseInferShape);
// ----------------LogNormalReverse END-------------------

// ----------------Randperm-------------------
CUST_IMPLEMT_INFERFUNC(Randperm, RandpermInfer) {
  TensorDesc output_desc = op.GetOutputDescByName("output");
  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  int64_t max_length;
  if (op.GetAttr("max_length", max_length) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr 'max_length' error.");
    return GRAPH_FAILED;
  }
  auto output_shape = Shape({max_length});
  output_desc.SetDataType(dtype);
  output_desc.SetShape(output_shape);
  return op.UpdateOutputDesc("output", output_desc);
}
CUST_INFER_FUNC_REG(Randperm, RandpermInfer);
// ----------------Randperm END-------------------

// ----------------Dropout2D-------------------
// ----------------Dropout3D-------------------
IMPLEMT_COMMON_INFERFUNC(DropoutNDInferShape) {
  TensorDesc output_desc = op.GetOutputDescByName("y");
  TensorDesc mask_desc = op.GetOutputDescByName("mask");

  DataType input_dtype = op.GetInputDescByName("x").GetDataType();
  ge::Shape shape_input = op.GetInputDescByName("x").GetShape();

  output_desc.SetShape(shape_input);
  output_desc.SetDataType(input_dtype);
  mask_desc.SetShape(shape_input);
  mask_desc.SetDataType(DT_BOOL);

  if (op.UpdateOutputDesc("y", output_desc) != GRAPH_SUCCESS ||
      op.UpdateOutputDesc("mask", mask_desc) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

CUST_COMMON_INFER_FUNC_REG(Dropout2D, DropoutNDInferShape);
CUST_COMMON_INFER_FUNC_REG(Dropout3D, DropoutNDInferShape);
// ----------------Dropout2D END-------------------

// ----------------Gamma-------------------
CUST_IMPLEMT_INFERFUNC(Gamma, GammaInfer) {
  const std::vector<std::string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor shape_data;
  if (op.GetInputConstData("shape", shape_data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get const value failed of [shape]");
    auto shape_desc = op.GetInputDesc("shape");
    std::vector<int64_t> shapedims = shape_desc.GetShape().GetDims();
    size_t dim_num = shapedims.size();

    if (dim_num > 1) {
      std::string err_msg = ConcatString("the rank[", dim_num, "] of input[shape] should not be more than 1");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector(dim_num, -1);
    std::vector<std::pair<int64_t, int64_t>> range_vector(dim_num, std::make_pair(1, -1));

    auto output_desc = op.GetOutputDesc("output");
    output_desc.SetShape(Shape(shape_vector));
    output_desc.SetShapeRange(range_vector);
    output_desc.SetDataType(DT_FLOAT);
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  DataType shape_dtype = shape_data.GetTensorDesc().GetDataType();
  std::vector<int64_t> shape_dims;
  if (shape_dtype == DT_INT32) {
    RandomOpCalcDims<int32_t>(shape_data, shape_dims);
  } else if (shape_dtype == DT_INT64) {
    RandomOpCalcDims<int64_t>(shape_data, shape_dims);
  } else {
    std::string err_msg =
      ConcatString("dtype of input[shape] must be INT32 or INT64, but got [", DataTypeToStringDesc(shape_dtype), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  Tensor alpha_data;
  ge::Shape alpha_shape;
  if (op.GetInputConstData("alpha", alpha_data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get const value failed of [alpha]");
    auto alpha_desc = op.GetInputDesc("alpha");
    alpha_shape = alpha_desc.GetShape();
  } else {
    alpha_shape = alpha_data.GetTensorDesc().GetShape();
  }

  Tensor beta_data;
  ge::Shape beta_shape;
  if (op.GetInputConstData("beta", beta_data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get const value failed of [beta]");
    auto beta_desc = op.GetInputDesc("beta");
    beta_shape = beta_desc.GetShape();
  } else {
    beta_shape = beta_data.GetTensorDesc().GetShape();
  }
  std::string op_name = TbeGetName(op);
  ge::Shape broadcast_shape;
  ge::Shape output_shape;
  InferBroadcastshapeForStatic(alpha_shape, beta_shape, broadcast_shape);
  InferBroadcastshapeForStatic(Shape(shape_dims), broadcast_shape, output_shape);
  OP_LOGI(op_name.c_str(), "infer output_shape: %s", to_string(output_shape).c_str());

  auto output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(DT_FLOAT);
  op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Gamma, GammaInfer);
// ----------------Gamma END-------------------

// ----------------LogUniformCandidateSampler-------------------
CUST_IMPLEMT_INFERFUNC(LogUniformCandidateSampler, LogUniformCandidateSamplerInfer) {
  // Infer shape
  int64_t num_sampled;
  if (op.GetAttr("num_sampled", num_sampled) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr 'num_sampled' failed.");
    return GRAPH_FAILED;
  }
  auto true_classes_desc = op.GetInputDescByName("true_classes");
  ge::Shape true_classes_shape = true_classes_desc.GetShape();
  const size_t valid_true_classes_shape_rank = 2;
  const size_t true_classes_shape_rank = true_classes_shape.GetDims().size();
  if (!IsUnknownRankShape(true_classes_shape) && true_classes_shape_rank != valid_true_classes_shape_rank) {
    std::string err_msg = ConcatString(
      "For 'LogUniformCandidateSampler', the rank of 'true_classes' must be 2, but got ", true_classes_shape_rank, ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  int64_t num_true;
  if (op.GetAttr("num_true", num_true) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr 'num_true' failed.");
    return GRAPH_FAILED;
  }
  if (!IsUnknown(true_classes_shape.GetDims()) && num_true != true_classes_shape.GetDim(1)) {
    std::string err_msg = ConcatString(
      "For 'LogUniformCandidateSampler', dim[1] of 'true_classes' must be equal to 'num_true', but got "
      "true_classes[1]: ",
      true_classes_shape.GetDim(1), ", 'num_true': ", num_true, ".");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_FAILED;
  }
  auto sampled_candidates_desc = op.GetOutputDesc("sampled_candidates");
  auto true_expected_count_desc = op.GetOutputDesc("true_expected_count");
  auto sampled_expected_count_desc = op.GetOutputDesc("sampled_expected_count");
  ge::Shape sampled_candidates_shape({num_sampled});
  ge::Shape sampled_expected_count_shape({num_sampled});
  sampled_candidates_desc.SetShape(sampled_candidates_shape);
  true_expected_count_desc.SetShape(true_classes_shape);
  sampled_expected_count_desc.SetShape(sampled_expected_count_shape);
  // Infer type
  auto true_classes_type = op.GetInputDescByName("true_classes").GetDataType();
  if (true_classes_type != DT_INT64) {
    OP_LOGE(TbeGetName(op).c_str(), "Dtype of 'true_classes' must be DT_INT64.");
    return GRAPH_FAILED;
  }
  sampled_candidates_desc.SetDataType(DT_INT64);
  true_expected_count_desc.SetDataType(DT_FLOAT);
  sampled_expected_count_desc.SetDataType(DT_FLOAT);
  return op.UpdateOutputDesc("sampled_candidates", sampled_candidates_desc) &&
         op.UpdateOutputDesc("true_expected_count", true_expected_count_desc) &&
         op.UpdateOutputDesc("sampled_expected_count", sampled_expected_count_desc);
}

CUST_INFER_FUNC_REG(LogUniformCandidateSampler, LogUniformCandidateSamplerInfer);
// ----------------LogUniformCandidateSampler END-------------------

// ----------------Multinomial-------------------
IMPLEMT_COMMON_INFERFUNC(MultinomialInfer) {
  auto logits_desc = op.GetInputDescByName("logits");
  auto num_samples_desc = op.GetInputDescByName("num_samples");
  auto output_desc = op.GetOutputDescByName("y");

  DataType logits_dtype = logits_desc.GetDataType();
  ge::Shape logits_shape = logits_desc.GetShape();
  int64_t num_samples = UNKNOWN_DIM;

  std::vector<std::string> input_infer_depends = {"num_samples"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  Tensor num_samples_tensor;
  if (op.GetInputConstData("num_samples", num_samples_tensor) == GRAPH_SUCCESS) {
    if (MakeDimForScalarInput(num_samples_tensor, num_samples, op) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
  }

  DataType output_dtype;
  if (op.GetAttr("dtype", output_dtype) != GRAPH_SUCCESS) {
    output_dtype = logits_dtype;
  }
  if (logits_shape.GetDims().size() == 2) {
    output_desc.SetShape(ge::Shape({logits_shape.GetDim(0), num_samples}));
  } else {
    output_desc.SetShape(ge::Shape({num_samples}));
  }
  output_desc.SetDataType(output_dtype);

  return op.UpdateOutputDesc("y", output_desc);
}

CUST_COMMON_INFER_FUNC_REG(Multinomial, MultinomialInfer);
// ----------------Multinomial END-------------------

IMPLEMT_COMMON_INFERFUNC(BatchSizeAndNumSampleInferShape) {
  auto logits_desc = op.GetInputDescByName("logits");
  auto num_samples_desc = op.GetInputDescByName("num_samples");
  auto output_desc = op.GetOutputDescByName("y");

  DataType logits_dtype = logits_desc.GetDataType();
  ge::Shape logits_shape = logits_desc.GetShape();
  ge::Shape output_shape({logits_shape.GetDim(0), UNKNOWN_DIM});

  std::vector<std::string> input_infer_depends = {"num_samples"};
  PREPARE_DYNAMIC_SHAPE(input_infer_depends);
  Tensor num_samples_tensor;
  if (op.GetInputConstData("num_samples", num_samples_tensor) == GRAPH_SUCCESS) {
    int64_t num_samples;
    if (MakeDimForScalarInput(num_samples_tensor, num_samples, op) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
    output_shape.SetDim(1, num_samples);
  }

  DataType output_dtype;
  if (op.GetAttr("dtype", output_dtype) != GRAPH_SUCCESS) {
    output_dtype = logits_dtype;
  }
  output_desc.SetShape(output_shape);
  output_desc.SetDataType(output_dtype);

  return op.UpdateOutputDesc("y", output_desc);
}

CUST_COMMON_INFER_FUNC_REG(RandomCategorical, BatchSizeAndNumSampleInferShape);
CUST_ONE_IN_ONE_OUT_INFER(RandomShuffle, x, y);

// ----------------RandomPoisson-------------------
CUST_IMPLEMT_INFERFUNC(RandomPoisson, RandomPoissonInfer) {
  std::vector<std::string> depend_input{"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_input);

  auto output_desc = op.GetOutputDesc("y");
  Shape output_shape(UNKNOWN_RANK);

  auto rate_desc = op.GetInputDescByName("rate");
  auto rate_shape = rate_desc.GetShape();

  Shape shape;
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS && !IsUnknownDimNum(rate_shape)) {
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Get shape error.");
      return GRAPH_FAILED;
    }
    if (Concatenate(shape, rate_shape, output_shape) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Concatenate shape error.");
      return GRAPH_FAILED;
    }
  }

  DataType dtype;
  if (op.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr dtype error.");
    return GRAPH_FAILED;
  }
  TensorDesc outputDesc = op.GetOutputDescByName("y");
  outputDesc.SetDataType(dtype);
  outputDesc.SetShape(output_shape);
  return op.UpdateOutputDesc("y", outputDesc);
}

CUST_INFER_FUNC_REG(RandomPoisson, RandomPoissonInfer);
// ----------------RandomPoisson End-------------------

// ----------------RandomChoiceWithMask-------------------
CUST_IMPLEMT_INFERFUNC(RandomChoiceWithMask, RandomChoiceWithMaskInfer) {
  auto x_desc = op.GetInputDescByName("x");
  auto x_shape = x_desc.GetShape();
  int64_t x_rank = IsUnknownRankShape(x_shape) ? UNKNOWN_DIM : x_shape.GetDims().size();

  int64_t count;
  if (op.GetAttr("count", count) != GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "Get attr 'count' failed.");
    return GRAPH_FAILED;
  }

  auto index_desc = op.GetOutputDesc("index");
  auto mask_desc = op.GetOutputDesc("mask");
  Shape index_shape({count, x_rank});
  Shape mask_shape({count});

  index_desc.SetDataType(DT_INT32);
  mask_desc.SetDataType(DT_BOOL);
  return op.UpdateOutputDesc("index", index_desc) && op.UpdateOutputDesc("mask", mask_desc);
}
CUST_INFER_FUNC_REG(RandomChoiceWithMask, RandomChoiceWithMaskInfer);
// ----------------RandomChoiceWithMask End-------------------

// ----------------RandomUniformInt-------------------
CUST_IMPLEMT_INFERFUNC(RandomUniformInt, RandomUniformIntInfer) {
  const vector<string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Shape shape{ge::UNKNOWN_RANK};
  Tensor shape_tensor;
  if (op.GetInputConstData("shape", shape_tensor) == GRAPH_SUCCESS) {
    if (MakeShapeFromShapeTensor(shape_tensor, shape, op) != GRAPH_SUCCESS) {
      OP_LOGE(TbeGetName(op).c_str(), "Get shape error.");
      return GRAPH_FAILED;
    }
  }

  TensorDesc outputDesc = op.GetOutputDescByName("y");
  outputDesc.SetDataType(DT_INT32);
  outputDesc.SetShape(shape);
  return op.UpdateOutputDesc("y", outputDesc);
}
CUST_INFER_FUNC_REG(RandomUniformInt, RandomUniformIntInfer);
// ----------------RandomUniformInt End-------------------

// ----------------Igamma-------------------

#define IGAMMA_INFER()                                          \
  do {                                                          \
    DataType a_type = op.GetInputDescByName("a").GetDataType(); \
    TensorDesc z_desc = op.GetOutputDescByName("z");            \
    z_desc.SetDataType(a_type);                                 \
    if (op.UpdateOutputDesc("z", z_desc) != GRAPH_SUCCESS) {    \
      return GRAPH_FAILED;                                      \
    }                                                           \
    auto a_desc = op.GetInputDescByName("a");                   \
    auto a_shape = a_desc.GetShape();                           \
    if (IsUnknownRankShape(a_shape)) {                          \
      std::vector<int64_t> out_dim{ge::UNKNOWN_DIM_NUM};        \
      z_desc.SetShape(ge::Shape(out_dim));                      \
      return op.UpdateOutputDesc("z", z_desc);                  \
    }                                                           \
    if (IsUnknown(a_shape.GetDims())) {                         \
      int64_t x_rank = a_shape.GetDims().size();                \
      std::vector<int64_t> out_dim(x_rank);                     \
      for (int64_t di = 0; di < x_rank; di++) {                 \
        out_dim[di] = UNKNOWN_DIM;                              \
      }                                                         \
      z_desc.SetShape(ge::Shape(out_dim));                      \
      return op.UpdateOutputDesc("z", z_desc);                  \
    }                                                           \
    return BROADCAST_INFER("a", "x", "z")(op);                  \
  } while (false)

CUST_IMPLEMT_INFERFUNC(Igamma, IgammaInfer) { IGAMMA_INFER(); }
CUST_INFER_FUNC_REG(Igamma, IgammaInfer);

IMPLEMT_INFERFUNC(Igammac, IgammacInfer) { IGAMMA_INFER(); }
INFER_FUNC_REG(Igammac, IgammacInfer);

IMPLEMT_INFERFUNC(IgammaGradA, IgammaGradAInfer) { IGAMMA_INFER(); }
INFER_FUNC_REG(IgammaGradA, IgammaGradAInfer);
// ----------------Igamma END-------------------

// ----------------Poisson-------------------
CUST_IMPLEMT_INFERFUNC(Poisson, PoissonInfer) {
  const std::vector<std::string> depend_names = {"shape"};
  PREPARE_DYNAMIC_SHAPE(depend_names);
  Tensor shape_data;
  if (op.GetInputConstData("shape", shape_data) != GRAPH_SUCCESS) {
    OP_LOGI(TbeGetName(op).c_str(), "Get const value failed of [shape]");
    auto shape_desc = op.GetInputDesc("shape");
    std::vector<int64_t> shapedims = shape_desc.GetShape().GetDims();
    size_t dim_num = shapedims.size();

    if (dim_num > 1) {
      std::string err_msg = ConcatString("the rank[", dim_num, "] of input[shape] should not be more than 1");
      AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
      return GRAPH_FAILED;
    }

    std::vector<int64_t> shape_vector(dim_num, -1);
    std::vector<std::pair<int64_t, int64_t>> range_vector(dim_num, std::make_pair(1, -1));

    auto output_desc = op.GetOutputDesc("output");
    output_desc.SetShape(Shape(shape_vector));
    output_desc.SetShapeRange(range_vector);
    output_desc.SetDataType(DT_INT32);
    op.UpdateOutputDesc("output", output_desc);
    return GRAPH_SUCCESS;
  }

  DataType shape_dtype = shape_data.GetTensorDesc().GetDataType();
  std::vector<int64_t> shape_dims;
  if (shape_dtype == DT_INT32) {
    RandomOpCalcDims<int32_t>(shape_data, shape_dims);
  } else if (shape_dtype == DT_INT64) {
    RandomOpCalcDims<int64_t>(shape_data, shape_dims);
  } else {
    std::string err_msg =
      ConcatString("dtype of input[shape] must be INT32 or INT64, but got [", DataTypeToStringDesc(shape_dtype), "].");
    AICPU_INFER_SHAPE_CALL_ERR_REPORT(TbeGetName(op), err_msg);
    return GRAPH_PARAM_INVALID;
  }

  auto output_desc = op.GetOutputDesc("output");
  output_desc.SetShape(Shape(shape_dims));
  output_desc.SetDataType(DT_INT32);
  op.UpdateOutputDesc("output", output_desc);
  return GRAPH_SUCCESS;
}

CUST_INFER_FUNC_REG(Poisson, PoissonInfer);
// ----------------Poisson END-------------------
}  // namespace ge
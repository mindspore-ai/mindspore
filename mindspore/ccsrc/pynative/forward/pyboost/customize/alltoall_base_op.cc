/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "include/pynative/forward/pyboost/converter.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/utils/pynative/dispatch/ms_dispatch.h"
#include "include/utils/tensor_py.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "pynative/forward/pyboost/auto_generate/pyboost_api.h"
#include "pynative/forward/pyboost/customize/pyboost_custom_reg.h"

namespace mindspore::pynative {
namespace {
constexpr char kErrorPrefix[] = "For '_all_to_all_single_v2', ";

struct AllToAllTensorInfo {
  ShapeVector shape;
  TypeId dtype;
  bool contiguous;
};

struct AllToAllTensors {
  AllToAllTensorInfo input;
  AllToAllTensorInfo output;
  // Null when Python fallback handles the tensors.
  tensor::TensorPtr output_tensor;
};

AllToAllTensorInfo ReadTensorInfo(const py::object &obj, const tensor::TensorPtr &native_tensor,
                                  bool use_python_metadata) {
  if (use_python_metadata) {
    return {py::cast<ShapeVector>(obj.attr("shape")), py::cast<TypePtr>(obj.attr("dtype"))->type_id(),
            py::cast<bool>(obj.attr("is_contiguous")())};
  }
  return {native_tensor->shape(), native_tensor->data_type(), native_tensor->is_contiguous()};
}

AllToAllTensors ReadAllToAllTensors(const py::object &input, const py::object &output) {
  // Choose the same metadata source for both tensors.
  bool use_python_metadata =
    fallback_enabled() && (PyObjectHasFallbackAttr(output.ptr()) || PyObjectHasFallbackAttr(input.ptr()));
  tensor::TensorPtr input_tensor;
  tensor::TensorPtr output_tensor;
  if (!use_python_metadata) {
    input_tensor = tensor::ConvertToTensor(input);
    output_tensor = tensor::ConvertToTensor(output);
    use_python_metadata = ShouldDispatch(input_tensor, output_tensor);
  }
  return {ReadTensorInfo(input, input_tensor, use_python_metadata),
          ReadTensorInfo(output, output_tensor, use_python_metadata), output_tensor};
}

void ValidateTensorMetadata(const AllToAllTensorInfo &input, const AllToAllTensorInfo &output) {
  if (input.shape.empty() || output.shape.empty()) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "input and output must have rank >= 1.";
  }
  const auto is_negative = [](int64_t dim) { return dim < 0; };
  if (std::any_of(input.shape.begin(), input.shape.end(), is_negative) ||
      std::any_of(output.shape.begin(), output.shape.end(), is_negative)) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "input and output dimensions must be non-negative.";
  }
  if (input.dtype != output.dtype) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "input and output must have the same dtype.";
  }
  if (!input.contiguous || !output.contiguous) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "input and output must be contiguous.";
  }
}

std::vector<int64_t> ParseSplitSizes(const py::object &splits, const char *name) {
  if (splits.is_none()) {
    return {};
  }
  if (!PyList_Check(splits.ptr()) && !PyTuple_Check(splits.ptr())) {
    MS_EXCEPTION(TypeError) << kErrorPrefix << name << " must be a list or tuple of integers.";
  }
  std::vector<int64_t> values;
  values.reserve(py::len(splits));
  for (const auto &item : splits) {
    if (!PyIndex_Check(item.ptr())) {
      MS_EXCEPTION(TypeError) << kErrorPrefix << name << " must contain integers.";
    }
    auto index = py::reinterpret_steal<py::object>(PyNumber_Index(item.ptr()));
    if (!index) {
      throw py::error_already_set();
    }
    int overflow = 0;
    const auto value = PyLong_AsLongLongAndOverflow(index.ptr(), &overflow);
    if (overflow != 0) {
      MS_EXCEPTION(ValueError) << kErrorPrefix << name << " contains an integer out of range.";
    }
    if (value < 0) {
      MS_EXCEPTION(ValueError) << kErrorPrefix << name << " must contain non-negative sizes.";
    }
    values.push_back(value);
  }
  return values;
}

int64_t MultiplyElementCount(int64_t count, int64_t size) {
  // Tensor dimensions and split sizes have already been validated as non-negative.
  if (size != 0 && count > std::numeric_limits<int64_t>::max() / size) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "tensor element count is out of range.";
  }
  return count * size;
}

int64_t Numel(const ShapeVector &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, MultiplyElementCount);
}

// Complex values are exchanged as two elements of their real component type.
std::pair<TypeId, int64_t> GetCommunicationType(TypeId dtype) {
  constexpr int64_t kComplexComponents = 2;
  if (dtype == kNumberTypeComplex64) {
    return {kNumberTypeFloat32, kComplexComponents};
  }
  if (dtype == kNumberTypeComplex128) {
    return {kNumberTypeFloat64, kComplexComponents};
  }
  return {dtype, 1};
}

// Validates explicit row splits or infers equal ones, then converts each peer's rows to communication element counts.
py::tuple BuildPeerCounts(const ShapeVector &shape, std::vector<int64_t> splits, uint32_t world_size,
                          int64_t elements_per_value, const char *name) {
  if (splits.empty()) {
    if (shape[0] % world_size != 0) {
      MS_EXCEPTION(ValueError) << kErrorPrefix << name << " dim 0 must be divisible by world_size, but got " << shape[0]
                               << " and " << world_size;
    }
    splits.assign(world_size, shape[0] / world_size);
  } else {
    if (splits.size() != world_size) {
      MS_EXCEPTION(ValueError) << kErrorPrefix << name << " split sizes length must equal world_size.";
    }
    // Subtract instead of summing so that oversized splits cannot overflow.
    int64_t remaining = shape[0];
    for (auto split : splits) {
      if (split > remaining) {
        MS_EXCEPTION(ValueError) << kErrorPrefix << name << " split sizes must sum to dim 0.";
      }
      remaining -= split;
    }
    if (remaining != 0) {
      MS_EXCEPTION(ValueError) << kErrorPrefix << name << " split sizes must sum to dim 0.";
    }
  }
  const auto row_size = std::accumulate(shape.begin() + 1, shape.end(), elements_per_value, MultiplyElementCount);
  py::tuple counts(splits.size());
  for (size_t i = 0; i < splits.size(); ++i) {
    counts[i] = py::int_(MultiplyElementCount(splits[i], row_size));
  }
  return counts;
}

py::object PrepareCommunicationTensor(const py::object &obj, TypeId dtype, TypeId comm_type) {
  if (dtype == comm_type) {
    return obj;
  }
  auto args = py::list(py::make_tuple(obj, static_cast<int64_t>(comm_type)));
  return py::reinterpret_steal<py::object>(PyboostViewDtypeBase(prim::kPrimViewDtype, args.ptr()));
}
}  // namespace

py::tuple ValidateAllToAllSingleAndBuildCommArgs(const py::object &output, const py::object &input,
                                                 const py::object &output_split_sizes,
                                                 const py::object &input_split_sizes, const py::object &group,
                                                 const py::object &async_op) {
  // Validate the Python arguments before reading tensor metadata.
  if (!tensor::IsTensorPy(input) || !tensor::IsTensorPy(output)) {
    MS_EXCEPTION(TypeError) << kErrorPrefix << "input and output must be Tensor.";
  }
  if (!PyUnicode_Check(group.ptr())) {
    MS_EXCEPTION(TypeError) << kErrorPrefix << "group must be a string.";
  }
  if (!PyBool_Check(async_op.ptr())) {
    MS_EXCEPTION(TypeError) << kErrorPrefix << "async_op must be a bool.";
  }
  const auto world_size =
    ::mindspore::distributed::collective::CollectiveManager::instance()->GetGroupSize(py::cast<std::string>(group));
  if (world_size == 0) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "world_size must be positive.";
  }
  const auto [input_info, output_info, output_tensor] = ReadAllToAllTensors(input, output);
  ValidateTensorMetadata(input_info, output_info);

  // Parse both split sequences before inferring any omitted splits.
  auto input_splits = ParseSplitSizes(input_split_sizes, "input_split_sizes");
  auto output_splits = ParseSplitSizes(output_split_sizes, "output_split_sizes");
  const bool equal_split = input_splits.empty() && output_splits.empty();
  if (equal_split && Numel(input_info.shape) != Numel(output_info.shape)) {
    MS_EXCEPTION(ValueError) << kErrorPrefix << "equally split input and output must have equal numel.";
  }
  const auto [comm_type, elements_per_value] = GetCommunicationType(input_info.dtype);
  const auto send_counts =
    BuildPeerCounts(input_info.shape, std::move(input_splits), world_size, elements_per_value, "input");
  const auto recv_counts =
    BuildPeerCounts(output_info.shape, std::move(output_splits), world_size, elements_per_value, "output");

  // Prepare communication views only after all argument checks succeed.
  if (output_tensor != nullptr) {
    // The caller reads the original output, which may differ from the communication view.
    output_tensor->set_need_pipeline_sync(true);
  }
  const auto send = PrepareCommunicationTensor(input, input_info.dtype, comm_type);
  const auto recv = PrepareCommunicationTensor(output, output_info.dtype, comm_type);
  return py::make_tuple(recv, send, group, send_counts, recv_counts, world_size, equal_split, async_op);
}

}  // namespace mindspore::pynative

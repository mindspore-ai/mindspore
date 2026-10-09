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

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "common/mockcpp.h"
#include "pybind11/eval.h"
#include "include/utils/python_adapter.h"
#include "ir/tensor_new.h"
// Include the implementation to exercise the native entry without rebuilding the Python module.
#include "pynative/forward/pyboost/customize/alltoall_base_op.cc"

namespace mindspore::pynative {
namespace {
using distributed::collective::CollectiveManager;

uint32_t test_group_size = 2;
std::vector<TypeId> requested_view_types;

uint32_t GetTestGroupSize(const CollectiveManager *, const std::string &) { return test_group_size; }

// Record the requested view dtype without submitting a device operation.
PyObject *RecordViewDtype(const PrimitivePtr &, PyObject *args) {
  requested_view_types.push_back(static_cast<TypeId>(py::cast<int64_t>(PyList_GET_ITEM(args, 1))));
  auto result = PyList_GET_ITEM(args, 0);
  Py_INCREF(result);
  return result;
}

struct TensorInfo {
  ShapeVector shape;
  TypeId dtype;
  bool contiguous;
};

class AllToAllBaseTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    (void)python_adapter::set_python_scoped();
    (void)py::module::import("mindspore");
  }

  void SetUp() override {
    test_group_size = 2;
    requested_view_types.clear();
    MOCKER_CPP(&CollectiveManager::GetGroupSize, uint32_t(*)(const CollectiveManager *, const std::string &))
      .stubs()
      .will(invoke(GetTestGroupSize));
    MOCKER(PyboostViewDtypeBase).stubs().will(invoke(RecordViewDtype));
  }

  void TearDown() override { GlobalMockObject::verify(); }

  static py::object MakeTensor(const ShapeVector &shape, TypeId dtype = kNumberTypeFloat32) {
    return tensor::PackTensorToPyObject(tensor::from_spec(dtype, shape, device::DeviceType::kNone));
  }

  static py::object MakeMetadataTensor(const TensorInfo &info) {
    py::dict scope;
    py::exec(R"(
import numpy as np
from mindspore._c_expression import TensorPy

class MetadataTensor(TensorPy):
    __fallback__ = None

    def __init__(self, shape, dtype, contiguous):
        super().__init__(np.array([0], dtype=np.float32))
        self.metadata_shape = shape
        self.metadata_dtype = dtype
        self.metadata_contiguous = contiguous

    @property
    def shape(self):
        return self.metadata_shape

    @property
    def dtype(self):
        return self.metadata_dtype

    def is_contiguous(self):
        return self.metadata_contiguous
)",
             scope);
    return scope["MetadataTensor"](py::cast(info.shape), py::cast(TypeIdToType(info.dtype)),
                                   py::bool_(info.contiguous));
  }

  static py::tuple BuildCommArgs(const py::object &output, const py::object &input,
                                 const py::object &output_splits = py::none(),
                                 const py::object &input_splits = py::none()) {
    return ValidateAllToAllSingleAndBuildCommArgs(output, input, output_splits, input_splits, py::str("named_group"),
                                                  py::bool_(false));
  }
};

/// Feature: all_to_all_single public argument validation.
/// Description: Pass Python scalar objects instead of input and output tensors.
/// Expectation: Reject the objects with TypeError before reading tensor metadata.
TEST_F(AllToAllBaseTest, TensorTypesRejectPythonScalars) {
  EXPECT_THROW(BuildCommArgs(py::int_(0), py::int_(1)), py::type_error);
}

/// Feature: all_to_all_single TensorPy support.
/// Description: Validate native TensorPy objects that are not instances of the Python Tensor subclass.
/// Expectation: Accept both objects using the common native tensor type check.
TEST_F(AllToAllBaseTest, TensorTypesAcceptTensorPyBaseObjects) {
  const auto numpy = py::module::import("numpy");
  const auto tensor_type = py::module::import("mindspore._c_expression").attr("TensorPy");
  const auto input = tensor_type(numpy.attr("array")(py::make_tuple(1, 2)));
  const auto output = tensor_type(numpy.attr("array")(py::make_tuple(0, 0)));

  EXPECT_NO_THROW(BuildCommArgs(output, input));
}

/// Feature: all_to_all_single communication options.
/// Description: Parse a named group and an explicit asynchronous boolean flag.
/// Expectation: Preserve the group name and asynchronous mode.
TEST_F(AllToAllBaseTest, OptionsPreserveNamedGroupAndAsyncMode) {
  const auto input = MakeTensor({4});
  const auto output = MakeTensor({4});

  const auto args = ValidateAllToAllSingleAndBuildCommArgs(output, input, py::none(), py::none(),
                                                           py::str("named_group"), py::bool_(true));

  EXPECT_EQ(py::cast<std::string>(args[2]), "named_group");
  EXPECT_TRUE(py::cast<bool>(args[7]));
}

/// Feature: all_to_all_single group type validation.
/// Description: Supply a byte string that a pybind string argument could implicitly accept.
/// Expectation: Preserve the public string-only contract by raising TypeError.
TEST_F(AllToAllBaseTest, OptionsRejectByteStringGroup) {
  const auto input = MakeTensor({4});
  const auto output = MakeTensor({4});

  EXPECT_THROW(ValidateAllToAllSingleAndBuildCommArgs(output, input, py::none(), py::none(), py::bytes("named_group"),
                                                      py::bool_(false)),
               py::type_error);
}

/// Feature: all_to_all_single native group validation.
/// Description: Pass an unresolved None to the native helper instead of a group resolved by the Python wrapper.
/// Expectation: Raise TypeError without looking up a default group in Python.
TEST_F(AllToAllBaseTest, OptionsRejectUnresolvedNoneGroup) {
  const auto input = MakeTensor({4});
  const auto output = MakeTensor({4});

  EXPECT_THROW(
    ValidateAllToAllSingleAndBuildCommArgs(output, input, py::none(), py::none(), py::none(), py::bool_(false)),
    py::type_error);
}

/// Feature: all_to_all_single asynchronous flag validation.
/// Description: Supply the integer one instead of a Python boolean.
/// Expectation: Raise TypeError instead of implicitly enabling asynchronous mode.
TEST_F(AllToAllBaseTest, OptionsRejectIntegerAsyncFlag) {
  const auto input = MakeTensor({4});
  const auto output = MakeTensor({4});

  EXPECT_THROW(
    ValidateAllToAllSingleAndBuildCommArgs(output, input, py::none(), py::none(), py::str("named_group"), py::int_(1)),
    py::type_error);
}

struct SplitParseCase {
  std::string name;
  std::function<py::object()> make_splits;
  std::vector<int64_t> expected;
};

class AllToAllSplitParseTest : public AllToAllBaseTest, public testing::WithParamInterface<SplitParseCase> {};

/// Feature: all_to_all_single split parsing.
/// Description: Parse omitted splits, empty sequences and integer index values.
/// Expectation: Preserve explicit split values and infer equal per-peer counts for omitted splits.
TEST_P(AllToAllSplitParseTest, ParsesOmittedAndIntegerSplits) {
  const auto &test_case = GetParam();
  const auto splits = test_case.make_splits();
  const auto input = MakeTensor({3});
  const auto output = MakeTensor({3});
  test_group_size = 3;

  const auto args = BuildCommArgs(output, input, splits, splits);

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), test_case.expected);
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), test_case.expected);
}

INSTANTIATE_TEST_SUITE_P(
  ValidSplits, AllToAllSplitParseTest,
  testing::Values(SplitParseCase{"None", []() { return py::none(); }, {1, 1, 1}},
                  SplitParseCase{"EmptyList", []() { return py::list(); }, {1, 1, 1}},
                  SplitParseCase{"EmptyTuple", []() { return py::tuple(); }, {1, 1, 1}},
                  SplitParseCase{"ListWithZero", []() { return py::list(py::make_tuple(0, 1, 2)); }, {0, 1, 2}},
                  SplitParseCase{"IntegerTuple", []() { return py::make_tuple(1, 2, 0); }, {1, 2, 0}},
                  SplitParseCase{"NumpyInteger",
                                 []() { return py::make_tuple(py::module::import("numpy").attr("int64")(2), 1, 0); },
                                 {2, 1, 0}}),
  [](const testing::TestParamInfo<SplitParseCase> &info) { return info.param.name; });

struct InvalidSplitCase {
  std::string name;
  std::function<py::object()> make_splits;
  ExceptionType exception_type;
};

class AllToAllInvalidSplitTest : public AllToAllBaseTest, public testing::WithParamInterface<InvalidSplitCase> {};

/// Feature: all_to_all_single split validation.
/// Description: Parse an unsupported container, a non-integer element, a negative size or an oversized integer.
/// Expectation: Raise the matching TypeError or ValueError without leaving a pending Python error.
TEST_P(AllToAllInvalidSplitTest, RejectsInvalidSplitsWithTheExpectedException) {
  const auto &test_case = GetParam();
  const auto splits = test_case.make_splits();

  const auto input = MakeTensor({4});
  const auto output = MakeTensor({4});

  if (test_case.exception_type == TypeError) {
    EXPECT_THROW(BuildCommArgs(output, input, py::none(), splits), py::type_error);
  } else {
    EXPECT_THROW(BuildCommArgs(output, input, py::none(), splits), py::value_error);
  }
  EXPECT_EQ(PyErr_Occurred(), nullptr);
}

INSTANTIATE_TEST_SUITE_P(
  InvalidSplits, AllToAllInvalidSplitTest,
  testing::Values(InvalidSplitCase{"ScalarContainer", []() { return py::int_(2); }, TypeError},
                  InvalidSplitCase{"StringContainer", []() { return py::str("1,2"); }, TypeError},
                  InvalidSplitCase{"FloatElement", []() { return py::make_tuple(1.0, 2); }, TypeError},
                  InvalidSplitCase{"StringElement", []() { return py::make_tuple("1", 2); }, TypeError},
                  InvalidSplitCase{"NegativeSize", []() { return py::make_tuple(-1, 2); }, ValueError},
                  InvalidSplitCase{"Int64Overflow",
                                   []() { return py::make_tuple(py::int_(std::numeric_limits<uint64_t>::max())); },
                                   ValueError}),
  [](const testing::TestParamInfo<InvalidSplitCase> &info) { return info.param.name; });

struct CountCase {
  std::string name;
  ShapeVector shape;
  std::vector<int64_t> splits;
  int64_t world_size;
  int64_t elements_per_value;
  std::vector<int64_t> expected;
};

class AllToAllCountTest : public AllToAllBaseTest, public testing::WithParamInterface<CountCase> {};

/// Feature: all_to_all_single communication counts.
/// Description: Convert original dimension-zero splits to per-peer element counts.
/// Expectation: Count complete rows, preserve zero-sized peers and double complex element counts.
TEST_P(AllToAllCountTest, ConvertsOriginalRowsToCommunicationElements) {
  const auto &test_case = GetParam();
  const auto dtype = test_case.elements_per_value == 2 ? kNumberTypeComplex128 : kNumberTypeFloat32;
  const auto input = MakeTensor(test_case.shape, dtype);
  const auto output = MakeTensor(test_case.shape, dtype);
  const auto splits = py::cast(test_case.splits);
  test_group_size = test_case.world_size;

  const auto args = BuildCommArgs(output, input, splits, splits);

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), test_case.expected);
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), test_case.expected);
}

INSTANTIATE_TEST_SUITE_P(ValidCounts, AllToAllCountTest,
                         testing::Values(CountCase{"EqualVector", {4}, {}, 2, 1, {2, 2}},
                                         CountCase{"EqualMatrix", {4, 3}, {}, 2, 1, {6, 6}},
                                         CountCase{"UnevenMatrix", {3, 2}, {1, 2}, 2, 1, {2, 4}},
                                         CountCase{"ZeroPeer", {3, 2}, {0, 3}, 2, 1, {0, 6}},
                                         CountCase{"EmptyLeadingDimension", {0, 2}, {}, 2, 1, {0, 0}},
                                         CountCase{"EmptyTrailingDimension", {2, 0}, {}, 2, 1, {0, 0}},
                                         CountCase{"ComplexOriginalRows", {3, 2}, {1, 2}, 2, 2, {4, 8}}),
                         [](const testing::TestParamInfo<CountCase> &info) { return info.param.name; });

class AllToAllInvalidCountTest : public AllToAllBaseTest, public testing::WithParamInterface<CountCase> {};

/// Feature: all_to_all_single count validation.
/// Description: Compute counts for invalid split lengths, sums, equal divisions or overflowing dimensions.
/// Expectation: Reject invalid communication counts with ValueError before submission.
TEST_P(AllToAllInvalidCountTest, RejectsInvalidOrOverflowingCounts) {
  const auto &test_case = GetParam();
  const auto dtype = test_case.elements_per_value == 2 ? kNumberTypeComplex128 : kNumberTypeFloat32;
  const auto input = MakeTensor(test_case.shape, dtype);
  const auto output = MakeTensor({0}, dtype);
  test_group_size = test_case.world_size;

  EXPECT_THROW(BuildCommArgs(output, input, py::make_tuple(0, 0), py::cast(test_case.splits)), py::value_error);
}

INSTANTIATE_TEST_SUITE_P(
  InvalidCounts, AllToAllInvalidCountTest,
  testing::Values(CountCase{"NotDivisible", {3}, {}, 2, 1, {}}, CountCase{"WrongSplitLength", {4}, {4}, 2, 1, {}},
                  CountCase{"SplitSumTooSmall", {4}, {1, 1}, 2, 1, {}},
                  CountCase{"SplitSumTooLarge", {4}, {3, 2}, 2, 1, {}},
                  CountCase{"NegativeDimension", {2, -1}, {}, 2, 1, {}},
                  CountCase{"RowSizeOverflow", {2, std::numeric_limits<int64_t>::max(), 2}, {}, 2, 1, {}},
                  CountCase{"PeerCountOverflow", {2, std::numeric_limits<int64_t>::max()}, {2, 0}, 2, 1, {}},
                  CountCase{"ComplexCountOverflow", {2, std::numeric_limits<int64_t>::max()}, {}, 2, 2, {}}),
  [](const testing::TestParamInfo<CountCase> &info) { return info.param.name; });

/// Feature: all_to_all_single count multiplication.
/// Description: Multiply the largest valid int64 count by one or zero.
/// Expectation: Preserve the exact boundary and allow zero-sized data without overflow.
TEST_F(AllToAllBaseTest, ExplicitSplitsAllowInt64BoundaryAndZeroCounts) {
  const auto max_count = std::numeric_limits<int64_t>::max();
  const auto input = MakeTensor({2, max_count});
  const auto output = MakeTensor({2, 0});

  const auto args = BuildCommArgs(output, input, py::make_tuple(1, 1), py::make_tuple(1, 1));

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{max_count, max_count}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{0, 0}));
}

/// Feature: all_to_all_single tensor size.
/// Description: Omit splits for tensors with different shapes but the same total element count.
/// Expectation: Accept the exchange and return equal per-peer counts.
TEST_F(AllToAllBaseTest, EqualSplitsAllowDifferentShapesWithEqualNumel) {
  const auto input = MakeTensor({2, 3, 4});
  const auto output = MakeTensor({4, 6});

  const auto args = BuildCommArgs(output, input);

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{12, 12}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{12, 12}));
  EXPECT_TRUE(py::cast<bool>(args[6]));
}

/// Feature: all_to_all_single tensor size overflow.
/// Description: Count a tensor whose total number of elements exceeds int64.
/// Expectation: Raise ValueError instead of wrapping the count.
TEST_F(AllToAllBaseTest, EqualSplitsRejectElementCountOverflow) {
  const auto input = MakeTensor({std::numeric_limits<int64_t>::max(), 2});
  const auto output = MakeTensor({2});

  EXPECT_THROW(BuildCommArgs(output, input), py::value_error);
}

struct TensorValidationCase {
  std::string name;
  TensorInfo output;
  TensorInfo input;
};

class AllToAllTensorValidationTest : public AllToAllBaseTest,
                                     public testing::WithParamInterface<TensorValidationCase> {};

/// Feature: all_to_all_single tensor metadata validation.
/// Description: Validate scalar, dtype-mismatched or non-contiguous input and output tensors.
/// Expectation: Raise ValueError for each unsupported metadata combination.
TEST_P(AllToAllTensorValidationTest, RejectsUnsupportedTensorMetadata) {
  const auto &test_case = GetParam();
  const auto input = MakeMetadataTensor(test_case.input);
  const auto output = MakeMetadataTensor(test_case.output);

  EXPECT_THROW(BuildCommArgs(output, input), py::value_error);
}

INSTANTIATE_TEST_SUITE_P(
  InvalidTensors, AllToAllTensorValidationTest,
  testing::Values(
    TensorValidationCase{"ScalarInput", {{2}, kNumberTypeFloat32, true}, {{}, kNumberTypeFloat32, true}},
    TensorValidationCase{"ScalarOutput", {{}, kNumberTypeFloat32, true}, {{2}, kNumberTypeFloat32, true}},
    TensorValidationCase{"DtypeMismatch", {{2}, kNumberTypeFloat64, true}, {{2}, kNumberTypeFloat32, true}},
    TensorValidationCase{"NonContiguousInput", {{2, 2}, kNumberTypeFloat32, true}, {{2, 2}, kNumberTypeFloat32, false}},
    TensorValidationCase{
      "NonContiguousOutput", {{2, 2}, kNumberTypeFloat32, false}, {{2, 2}, kNumberTypeFloat32, true}}),
  [](const testing::TestParamInfo<TensorValidationCase> &info) { return info.param.name; });

/// Feature: all_to_all_single independent tensor shapes.
/// Description: Validate contiguous tensors with different leading dimensions.
/// Expectation: Allow differing input and output lengths for exchanges with explicit splits.
TEST_F(AllToAllBaseTest, TensorValidationAllowsDifferentLeadingDimensions) {
  const auto input = MakeTensor({3, 2});
  const auto output = MakeTensor({4, 2});

  const auto args = BuildCommArgs(output, input, py::make_tuple(2, 2), py::make_tuple(1, 2));

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{2, 4}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{4, 4}));
  EXPECT_FALSE(py::cast<bool>(args[6]));
}

/// Feature: all_to_all_single native tensor metadata.
/// Description: Read the shape, dtype and layout from a native tensor without accessing Python attributes.
/// Expectation: Preserve its original multidimensional metadata.
TEST_F(AllToAllBaseTest, NativeTensorMetadataProducesCountsAndMarksOriginalOutputForSync) {
  const auto input = MakeTensor({2, 3});
  const auto output = MakeTensor({2, 3});

  const auto args = BuildCommArgs(output, input);

  EXPECT_EQ(args[0].ptr(), output.ptr());
  EXPECT_EQ(args[1].ptr(), input.ptr());
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{3, 3}));
  EXPECT_TRUE(tensor::ConvertToTensor(output)->need_pipeline_sync());
}

/// Feature: all_to_all_single Python tensor metadata.
/// Description: Read wrapper metadata that differs from the backing tensor's shape and dtype.
/// Expectation: Honor the wrapper's shape, dtype and layout when dispatch is active.
TEST_F(AllToAllBaseTest, FallbackUsesWrapperMetadataInsteadOfBackingTensorShape) {
  const auto input = MakeMetadataTensor({{4, 3}, kNumberTypeFloat64, true});
  const auto output = MakeMetadataTensor({{4, 3}, kNumberTypeFloat64, true});

  const auto args = BuildCommArgs(output, input);

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{6, 6}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{6, 6}));
}

/// Feature: all_to_all_single parameter preparation.
/// Description: Prepare an equally split matrix exchange without submitting a communication operator.
/// Expectation: Return per-peer element counts and select the equal-split path.
TEST_F(AllToAllBaseTest, OmittedAndEmptySplitsSelectEqualMatrixExchange) {
  const auto input = MakeTensor({4, 3});
  const auto output = MakeTensor({4, 3});

  const auto args = BuildCommArgs(output, input, py::none(), py::tuple());

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{6, 6}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{6, 6}));
  EXPECT_EQ(py::cast<uint32_t>(args[5]), 2U);
  EXPECT_TRUE(py::cast<bool>(args[6]));
}

/// Feature: all_to_all_single complex parameter preparation.
/// Description: Omit input splits and explicitly split a larger complex output by its original rows.
/// Expectation: Infer input splits independently and count both components using float64 storage.
TEST_F(AllToAllBaseTest, OmittedInputSplitsCountOriginalComplexRowsIndependently) {
  const auto input = MakeTensor({4, 2}, kNumberTypeComplex128);
  const auto output = MakeTensor({6, 2}, kNumberTypeComplex128);

  const auto args = BuildCommArgs(output, input, py::make_tuple(2, 4), py::none());

  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[3]), (std::vector<int64_t>{8, 8}));
  EXPECT_EQ(py::cast<std::vector<int64_t>>(args[4]), (std::vector<int64_t>{8, 16}));
  EXPECT_FALSE(py::cast<bool>(args[6]));
  EXPECT_EQ(requested_view_types, (std::vector<TypeId>{kNumberTypeFloat64, kNumberTypeFloat64}));
}

/// Feature: all_to_all_single group validation.
/// Description: Prepare an exchange for a group with no ranks.
/// Expectation: Reject the group with ValueError before dividing or computing counts.
TEST_F(AllToAllBaseTest, ZeroWorldSizeIsRejectedBeforeCountCalculation) {
  const auto input = MakeTensor({2});
  const auto output = MakeTensor({2});
  test_group_size = 0;

  EXPECT_THROW(BuildCommArgs(output, input), py::value_error);
}

/// Feature: all_to_all_single equal-split size validation.
/// Description: Omit both split sequences for tensors with different element counts.
/// Expectation: Reject the exchange with ValueError before submission.
TEST_F(AllToAllBaseTest, EqualSplitsRejectUnequalElementCounts) {
  const auto input = MakeTensor({2, 3});
  const auto output = MakeTensor({2, 2});

  EXPECT_THROW(BuildCommArgs(output, input), py::value_error);
}

struct CommunicationTypeCase {
  std::string name;
  TypeId dtype;
  TypeId expected;
};

class AllToAllCommunicationTypeTest : public AllToAllBaseTest,
                                      public testing::WithParamInterface<CommunicationTypeCase> {};

/// Feature: all_to_all_single communication dtype.
/// Description: Select the storage component type used to communicate real or complex tensors.
/// Expectation: Map complex64/complex128 to float32/float64 and preserve other dtypes.
TEST_P(AllToAllCommunicationTypeTest, SelectsTheCommunicationStorageType) {
  const auto &test_case = GetParam();
  const auto input = MakeTensor({4}, test_case.dtype);
  const auto output = MakeTensor({4}, test_case.dtype);

  const auto args = BuildCommArgs(output, input);

  if (test_case.dtype == test_case.expected) {
    EXPECT_TRUE(requested_view_types.empty());
    EXPECT_EQ(args[0].ptr(), output.ptr());
    EXPECT_EQ(args[1].ptr(), input.ptr());
  } else {
    EXPECT_EQ(requested_view_types, (std::vector<TypeId>{test_case.expected, test_case.expected}));
  }
}

INSTANTIATE_TEST_SUITE_P(CommunicationTypes, AllToAllCommunicationTypeTest,
                         testing::Values(CommunicationTypeCase{"Complex64", kNumberTypeComplex64, kNumberTypeFloat32},
                                         CommunicationTypeCase{"Complex128", kNumberTypeComplex128, kNumberTypeFloat64},
                                         CommunicationTypeCase{"Float32", kNumberTypeFloat32, kNumberTypeFloat32},
                                         CommunicationTypeCase{"BFloat16", kNumberTypeBFloat16, kNumberTypeBFloat16},
                                         CommunicationTypeCase{"Int64", kNumberTypeInt64, kNumberTypeInt64}),
                         [](const testing::TestParamInfo<CommunicationTypeCase> &info) { return info.param.name; });
}  // namespace
}  // namespace mindspore::pynative

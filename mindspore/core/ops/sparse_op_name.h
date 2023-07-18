/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_BASE_SPARSE_OP_NAME_H_
#define MINDSPORE_CORE_BASE_SPARSE_OP_NAME_H_

namespace mindspore {
// Sparse ops
constexpr auto kSparseFillEmptyRowsGradOpName = "SparseFillEmptyRowsGrad";
constexpr auto kSparseSparseMinimumOpName = "SparseSparseMinimum";
constexpr auto kSparseCrossOpName = "SparseCross";
constexpr auto kRaggedTensorToTensorOpName = "RaggedTensorToTensor";
constexpr auto kSparseTensorDenseMatmulOpName = "SparseTensorDenseMatmul";
constexpr auto kSparseFillEmptyRowsOpName = "SparseFillEmptyRows";
constexpr auto kSparseToDenseV2OpName = "SparseToDenseV2";
constexpr auto kSparseSoftmaxOpName = "SparseSoftmax";
constexpr auto kSparseAddmmOpName = "SparseAddmm";
constexpr auto kSparseSparseMaximumOpName = "SparseSparseMaximum";
constexpr auto kCSRReduceSumOpName = "CSRReduceSum";
constexpr auto kCSRMVOpName = "CSRMV";
constexpr auto kCSRMMOpName = "CSRMM";
constexpr auto kCSRMulOpName = "CSRMul";
constexpr auto kCSRGatherOpName = "CSRGather";
constexpr auto kCSR2COOOpName = "CSR2COO";
constexpr auto kSparseDenseCwiseAddOpName = "SparseDenseCwiseAdd";
constexpr auto kSparseDenseCwiseDivOpName = "SparseDenseCwiseDiv";
constexpr auto kSparseDenseCwiseMulOpName = "SparseDenseCwiseMul";
constexpr auto kCOO2CSROpName = "COO2CSR";
constexpr auto kCSRDivOpName = "CSRDiv";
constexpr auto kDenseToDenseSetOperationOpName = "DenseToDenseSetOperation";
constexpr auto kSparseMatrixAddOpName = "SparseMatrixAdd";
constexpr auto kSparseMatrixMulOpName = "SparseMatrixMul";
constexpr auto kSparseAddOpName = "SparseAdd";
constexpr auto kSparseSegmentMeanGradOpName = "SparseSegmentMeanGrad";
constexpr auto kSparseSegmentMeanWithNumSegmentsOpName = "SparseSegmentMeanWithNumSegments";
constexpr auto kSparseConcatOpName = "SparseConcat";
constexpr auto kSparseMatrixNNZOpName = "SparseMatrixNNZ";
constexpr auto kSparseMatrixTransposeOpName = "SparseMatrixTranspose";
constexpr auto kSparseMatrixSoftmaxOpName = "SparseMatrixSoftmax";
constexpr auto kSparseMatrixMatMulOpName = "SparseMatrixMatMul";
constexpr auto kSparseMatrixSparseMatMulOpName = "SparseMatrixSparseMatMul";
constexpr auto kSparseMatrixOrderingAMDOpName = "SparseMatrixOrderingAMD";
constexpr auto kSparseSegmentSumOpName = "SparseSegmentSum";
constexpr auto kSparseSegmentSumGradOpName = "SparseSegmentSumGrad";
constexpr auto kSparseSegmentSumWithNumSegmentsOpName = "SparseSegmentSumWithNumSegments";
constexpr auto kSparseSegmentSqrtNOpName = "SparseSegmentSqrtN";
constexpr auto kSparseSegmentSqrtNGradOpName = "SparseSegmentSqrtNGrad";
constexpr auto kSparseSegmentSqrtNWithNumSegmentsOpName = "SparseSegmentSqrtNWithNumSegments";
constexpr auto kRaggedTensorToSparseOpName = "RaggedTensorToSparse";
constexpr auto kDenseToSparseSetOperationOpName = "DenseToSparseSetOperation";
constexpr auto kSparseTensorToCSRSparseMatrixOpName = "SparseTensorToCSRSparseMatrix";
constexpr auto kCSRSparseMatrixToSparseTensorOpName = "CSRSparseMatrixToSparseTensor";
constexpr auto kSparseSplitOpName = "SparseSplit";
constexpr auto kSparseReshapeOpName = "SparseReshape";
constexpr auto kSparseReorderOpName = "SparseReorder";
constexpr auto kSparseSegmentMeanOpName = "SparseSegmentMean";

// Sparse Grad ops
constexpr auto kSparseAddGradOpName = "SparseAddGrad";
constexpr auto kSparseTensorDenseAddOpName = "SparseTensorDenseAdd";
constexpr auto kSparseSliceOpName = "SparseSlice";
constexpr auto kSparseSliceGradOpName = "SparseSliceGrad";

// COOTensor
constexpr auto kMakeCOOTensorOpName = "MakeCOOTensor";
constexpr auto kCOOTensorGetValuesOpName = "COOTensorGetValues";
constexpr auto kCOOTensorGetIndicesOpName = "COOTensorGetIndices";
constexpr auto kCOOTensorGetDenseShapeOpName = "COOTensorGetDenseShape";
constexpr auto kCOOTensorDenseMatmulOpName = "COOTensorDenseMatmul";

// RowTensor
constexpr auto kMakeRowTensorOpName = "MakeRowTensor";
constexpr auto kRowTensorGetValuesOpName = "RowTensorGetValues";
constexpr auto kRowTensorGetIndicesOpName = "RowTensorGetIndices";
constexpr auto kRowTensorGetDenseShapeOpName = "RowTensorGetDenseShape";
constexpr auto kRowTensorAddOpName = "RowTensorAdd";

// CSRTensor
constexpr auto kMakeCSRTensorOpName = "MakeCSRTensor";
constexpr auto kCSRTensorGetValuesOpName = "CSRTensorGetValues";
constexpr auto kCSRTensorGetIndptrOpName = "CSRTensorGetIndptr";
constexpr auto kCSRTensorGetIndicesOpName = "CSRTensorGetIndices";
constexpr auto kCSRTensorGetDenseShapeOpName = "CSRTensorGetDenseShape";
constexpr auto kIsCSRFuncOpName = "IsCSRFunc";

// MapTensor
constexpr auto kMakeMapParameterOpName = "MakeMapParameter";
constexpr auto kMapTensorGetOpName = "MapTensorGet";
constexpr auto kMapTensorPutOpName = "MapTensorPut";
constexpr auto kMapTensorEraseOpName = "MapTensorErase";
constexpr auto kMapTensorPutWithStatusOpName = "MapTensorPutWithStatus";
constexpr auto kMapTensorGetDefaultValueOpName = "MapTensorGetDefaultValue";
constexpr auto kMapTensorGetPermitFilterValueOpName = "MapTensorGetPermitFilterValue";
constexpr auto kMapTensorGetEvictFilterValueOpName = "MapTensorGetEvictFilterValue";
constexpr auto kMapTensorGetKeysOpName = "MapTensorGetKeys";
constexpr auto kMapTensorGetValuesOpName = "MapTensorGetValues";
constexpr auto kMapTensorGetDataOpName = "MapTensorGetData";
constexpr auto kMapTensorGetGradOpName = "MapTensorGetGrad";
constexpr auto kCSRSparseMatrixToDenseOpName = "CSRSparseMatrixToDense";
constexpr auto kSspaddmmOpName = "Sspaddmm";
constexpr auto kDenseToCSRSparseMatrixOpName = "DenseToCSRSparseMatrix";
constexpr auto kSparseToDenseOpName = "SparseToDense";
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SPARSE_OP_NAME_H_

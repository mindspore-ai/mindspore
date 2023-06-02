/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_BASE_SPARSE_OPS_H_
#define MINDSPORE_CORE_BASE_SPARSE_OPS_H_

#include <iostream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Sparse ops
constexpr auto kSparseFillEmptyRowsGrad = "SparseFillEmptyRowsGrad";
constexpr auto kSparseSparseMinimum = "SparseSparseMinimum";
constexpr auto kSparseCross = "SparseCross";
constexpr auto kRaggedTensorToTensor = "RaggedTensorToTensor";
constexpr auto kSparseTensorDenseMatmul = "SparseTensorDenseMatmul";
constexpr auto kSparseFillEmptyRows = "SparseFillEmptyRows";
constexpr auto kSparseToDenseV2 = "SparseToDenseV2";
constexpr auto kSparseSoftmax = "SparseSoftmax";
constexpr auto kSparseAddmm = "SparseAddmm";
constexpr auto kSparseSparseMaximum = "SparseSparseMaximum";
constexpr auto kCSRReduceSum = "CSRReduceSum";
constexpr auto kCSRMV = "CSRMV";
constexpr auto kCSRMM = "CSRMM";
constexpr auto kCSRMul = "CSRMul";
constexpr auto kCSRGather = "CSRGather";
constexpr auto kCSR2COO = "CSR2COO";
constexpr auto kSparseDenseCwiseAdd = "SparseDenseCwiseAdd";
constexpr auto kSparseDenseCwiseDiv = "SparseDenseCwiseDiv";
constexpr auto kSparseDenseCwiseMul = "SparseDenseCwiseMul";
constexpr auto kCOO2CSR = "COO2CSR";
constexpr auto kCSRDiv = "CSRDiv";
constexpr auto kDenseToDenseSetOperation = "DenseToDenseSetOperation";
constexpr auto kSparseMatrixAdd = "SparseMatrixAdd";
constexpr auto kSparseMatrixMul = "SparseMatrixMul";
constexpr auto kSparseAdd = "SparseAdd";
constexpr auto kSparseSegmentMeanGrad = "SparseSegmentMeanGrad";
constexpr auto kSparseSegmentMeanWithNumSegments = "SparseSegmentMeanWithNumSegments";
constexpr auto kSparseConcat = "SparseConcat";
constexpr auto kSparseMatrixNNZ = "SparseMatrixNNZ";
constexpr auto kSparseMatrixTranspose = "SparseMatrixTranspose";
constexpr auto kSparseMatrixSoftmax = "SparseMatrixSoftmax";
constexpr auto kSparseMatrixMatMul = "SparseMatrixMatMul";
constexpr auto kSparseMatrixSparseMatMul = "SparseMatrixSparseMatMul";
constexpr auto kSparseMatrixOrderingAMD = "SparseMatrixOrderingAMD";
constexpr auto kSparseSegmentSum = "SparseSegmentSum";
constexpr auto kSparseSegmentSumGrad = "SparseSegmentSumGrad";
constexpr auto kSparseSegmentSumWithNumSegments = "SparseSegmentSumWithNumSegments";
constexpr auto kSparseSegmentSqrtN = "SparseSegmentSqrtN";
constexpr auto kSparseSegmentSqrtNGrad = "SparseSegmentSqrtNGrad";
constexpr auto kSparseSegmentSqrtNWithNumSegments = "SparseSegmentSqrtNWithNumSegments";
constexpr auto kRaggedTensorToSparse = "RaggedTensorToSparse";
constexpr auto kDenseToSparseSetOperation = "DenseToSparseSetOperation";
constexpr auto kSparseTensorToCSRSparseMatrix = "SparseTensorToCSRSparseMatrix";
constexpr auto kCSRSparseMatrixToSparseTensor = "CSRSparseMatrixToSparseTensor";
constexpr auto kSparseSplit = "SparseSplit";
constexpr auto kSparseReshape = "SparseReshape";
constexpr auto kSparseReorder = "SparseReorder";
constexpr auto kSparseSegmentMean = "SparseSegmentMean";

// Sparse Grad ops
constexpr auto kSparseAddGrad = "SparseAddGrad";
constexpr auto kSparseTensorDenseAdd = "SparseTensorDenseAdd";
constexpr auto kSparseSlice = "SparseSlice";
constexpr auto kSparseSliceGrad = "SparseSliceGrad";

// Sparse ops
GVAR_DEF(PrimitivePtr, kPrimSparseReorder, std::make_shared<Primitive>(kSparseReorder));
GVAR_DEF(PrimitivePtr, kPrimSparseReshape, std::make_shared<Primitive>(kSparseReshape));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMinimum, std::make_shared<Primitive>(kSparseSparseMinimum));
GVAR_DEF(PrimitivePtr, kPrimDenseToSparseSetOperation, std::make_shared<Primitive>(kDenseToSparseSetOperation));
GVAR_DEF(PrimitivePtr, kPrimSparseCross, std::make_shared<Primitive>(kSparseCross));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToTensor, std::make_shared<Primitive>(kRaggedTensorToTensor));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseMatmul, std::make_shared<Primitive>(kSparseTensorDenseMatmul));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRows, std::make_shared<Primitive>(kSparseFillEmptyRows));
GVAR_DEF(PrimitivePtr, kPrimSparseToDenseV2, std::make_shared<Primitive>(kSparseToDenseV2));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmax, std::make_shared<Primitive>(kSparseSoftmax));
GVAR_DEF(PrimitivePtr, kPrimSparseAddmm, std::make_shared<Primitive>(kSparseAddmm));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMul, std::make_shared<Primitive>(kSparseMatrixMul));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMaximum, std::make_shared<Primitive>(kSparseSparseMaximum));
GVAR_DEF(PrimitivePtr, kPrimCSRReduceSum, std::make_shared<Primitive>(kCSRReduceSum));
GVAR_DEF(PrimitivePtr, kPrimCSRMV, std::make_shared<Primitive>(kCSRMV));
GVAR_DEF(PrimitivePtr, kPrimCSRMM, std::make_shared<Primitive>(kCSRMM));
GVAR_DEF(PrimitivePtr, kPrimCSRMul, std::make_shared<Primitive>(kCSRMul));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseAdd, std::make_shared<Primitive>(kSparseDenseCwiseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseDiv, std::make_shared<Primitive>(kSparseDenseCwiseDiv));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseMul, std::make_shared<Primitive>(kSparseDenseCwiseMul));
GVAR_DEF(PrimitivePtr, kPrimCSRGather, std::make_shared<Primitive>(kCSRGather));
GVAR_DEF(PrimitivePtr, kPrimCSR2COO, std::make_shared<Primitive>(kCSR2COO));
GVAR_DEF(PrimitivePtr, kPrimCOO2CSR, std::make_shared<Primitive>(kCOO2CSR));
GVAR_DEF(PrimitivePtr, kPrimCSRDiv, std::make_shared<Primitive>(kCSRDiv));
GVAR_DEF(PrimitivePtr, kPrimSparseSplit, std::make_shared<Primitive>(kSparseSplit));
GVAR_DEF(PrimitivePtr, kPrimDenseToDenseSetOperation, std::make_shared<Primitive>(kDenseToDenseSetOperation));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixAdd, std::make_shared<Primitive>(kSparseMatrixAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseAdd, std::make_shared<Primitive>(kSparseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMean, std::make_shared<Primitive>(kSparseSegmentMean));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanGrad, std::make_shared<Primitive>("SparseSegmentMeanGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentMeanWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimDenseToCSRSparseMatrix, std::make_shared<Primitive>("DenseToCSRSparseMatrix"));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorToCSRSparseMatrix, std::make_shared<Primitive>(kSparseTensorToCSRSparseMatrix));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToSparseTensor, std::make_shared<Primitive>(kCSRSparseMatrixToSparseTensor));
GVAR_DEF(PrimitivePtr, kPrimSparseConcat, std::make_shared<Primitive>(kSparseConcat));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixNNZ, std::make_shared<Primitive>(kSparseMatrixNNZ));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSoftmax, std::make_shared<Primitive>(kSparseMatrixSoftmax));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMatMul, std::make_shared<Primitive>(kSparseMatrixMatMul));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSparseMatMul, std::make_shared<Primitive>(kSparseMatrixSparseMatMul));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToDense, std::make_shared<Primitive>("CSRSparseMatrixToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixTranspose, std::make_shared<Primitive>(kSparseMatrixTranspose));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixOrderingAMD, std::make_shared<Primitive>(kSparseMatrixOrderingAMD));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSum, std::make_shared<Primitive>("SparseSegmentSum"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumGrad, std::make_shared<Primitive>("SparseSegmentSumGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSumWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtN, std::make_shared<Primitive>("SparseSegmentSqrtN"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNGrad, std::make_shared<Primitive>("SparseSegmentSqrtNGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSqrtNWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToSparse, std::make_shared<Primitive>(kRaggedTensorToSparse));
GVAR_DEF(PrimitivePtr, kPrimSparseToDense, std::make_shared<Primitive>("SparseToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseCountSparseOutput, std::make_shared<Primitive>("SparseCountSparseOutput"));
GVAR_DEF(PrimitivePtr, kPrimSspaddmm, std::make_shared<Primitive>("Sspaddmm"));

// Sparse Grad ops
GVAR_DEF(PrimitivePtr, kPrimSparseAddGrad, std::make_shared<Primitive>(kSparseAddGrad));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRowsGrad, std::make_shared<Primitive>(kSparseFillEmptyRowsGrad));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseAdd, std::make_shared<Primitive>(kSparseTensorDenseAdd));
GVAR_DEF(PrimitivePtr, kPrimSparseSlice, std::make_shared<Primitive>(kSparseSlice));
GVAR_DEF(PrimitivePtr, kPrimSparseSliceGrad, std::make_shared<Primitive>(kSparseSliceGrad));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SPARSE_OPS_H_

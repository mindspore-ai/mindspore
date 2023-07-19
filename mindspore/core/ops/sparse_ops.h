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

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/sparse_op_name.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// Sparse ops
GVAR_DEF(PrimitivePtr, kPrimSparseReorder, std::make_shared<Primitive>(kSparseReorderOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseReshape, std::make_shared<Primitive>(kSparseReshapeOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMinimum, std::make_shared<Primitive>(kSparseSparseMinimumOpName));
GVAR_DEF(PrimitivePtr, kPrimDenseToSparseSetOperation, std::make_shared<Primitive>(kDenseToSparseSetOperationOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseCross, std::make_shared<Primitive>(kSparseCrossOpName));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToTensor, std::make_shared<Primitive>(kRaggedTensorToTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseMatmul, std::make_shared<Primitive>(kSparseTensorDenseMatmulOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRows, std::make_shared<Primitive>(kSparseFillEmptyRowsOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseToDenseV2, std::make_shared<Primitive>(kSparseToDenseV2OpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSoftmax, std::make_shared<Primitive>(kSparseSoftmaxOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseAddmm, std::make_shared<Primitive>(kSparseAddmmOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMul, std::make_shared<Primitive>(kSparseMatrixMulOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSparseMaximum, std::make_shared<Primitive>(kSparseSparseMaximumOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRReduceSum, std::make_shared<Primitive>(kCSRReduceSumOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRMV, std::make_shared<Primitive>(kCSRMVOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRMM, std::make_shared<Primitive>(kCSRMMOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRMul, std::make_shared<Primitive>(kCSRMulOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseAdd, std::make_shared<Primitive>(kSparseDenseCwiseAddOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseDiv, std::make_shared<Primitive>(kSparseDenseCwiseDivOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseDenseCwiseMul, std::make_shared<Primitive>(kSparseDenseCwiseMulOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRGather, std::make_shared<Primitive>(kCSRGatherOpName));
GVAR_DEF(PrimitivePtr, kPrimCSR2COO, std::make_shared<Primitive>(kCSR2COOOpName));
GVAR_DEF(PrimitivePtr, kPrimCOO2CSR, std::make_shared<Primitive>(kCOO2CSROpName));
GVAR_DEF(PrimitivePtr, kPrimCSRDiv, std::make_shared<Primitive>(kCSRDivOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSplit, std::make_shared<Primitive>(kSparseSplitOpName));
GVAR_DEF(PrimitivePtr, kPrimDenseToDenseSetOperation, std::make_shared<Primitive>(kDenseToDenseSetOperationOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixAdd, std::make_shared<Primitive>(kSparseMatrixAddOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseAdd, std::make_shared<Primitive>(kSparseAddOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMean, std::make_shared<Primitive>(kSparseSegmentMeanOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanGrad, std::make_shared<Primitive>("SparseSegmentMeanGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentMeanWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentMeanWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimDenseToCSRSparseMatrix, std::make_shared<Primitive>("DenseToCSRSparseMatrix"));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorToCSRSparseMatrix,
         std::make_shared<Primitive>(kSparseTensorToCSRSparseMatrixOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToSparseTensor,
         std::make_shared<Primitive>(kCSRSparseMatrixToSparseTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseConcat, std::make_shared<Primitive>(kSparseConcatOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixNNZ, std::make_shared<Primitive>(kSparseMatrixNNZOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSoftmax, std::make_shared<Primitive>(kSparseMatrixSoftmaxOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixMatMul, std::make_shared<Primitive>(kSparseMatrixMatMulOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixSparseMatMul, std::make_shared<Primitive>(kSparseMatrixSparseMatMulOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRSparseMatrixToDense, std::make_shared<Primitive>("CSRSparseMatrixToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixTranspose, std::make_shared<Primitive>(kSparseMatrixTransposeOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseMatrixOrderingAMD, std::make_shared<Primitive>(kSparseMatrixOrderingAMDOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSum, std::make_shared<Primitive>("SparseSegmentSum"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumGrad, std::make_shared<Primitive>("SparseSegmentSumGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSumWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSumWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtN, std::make_shared<Primitive>("SparseSegmentSqrtN"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNGrad, std::make_shared<Primitive>("SparseSegmentSqrtNGrad"));
GVAR_DEF(PrimitivePtr, kPrimSparseSegmentSqrtNWithNumSegments,
         std::make_shared<Primitive>("SparseSegmentSqrtNWithNumSegments"));
GVAR_DEF(PrimitivePtr, kPrimRaggedTensorToSparse, std::make_shared<Primitive>(kRaggedTensorToSparseOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseToDense, std::make_shared<Primitive>("SparseToDense"));
GVAR_DEF(PrimitivePtr, kPrimSparseCountSparseOutput, std::make_shared<Primitive>("SparseCountSparseOutput"));
GVAR_DEF(PrimitivePtr, kPrimSspaddmm, std::make_shared<Primitive>("Sspaddmm"));

// Sparse Grad ops
GVAR_DEF(PrimitivePtr, kPrimSparseAddGrad, std::make_shared<Primitive>(kSparseAddGradOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseFillEmptyRowsGrad, std::make_shared<Primitive>(kSparseFillEmptyRowsGradOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseTensorDenseAdd, std::make_shared<Primitive>(kSparseTensorDenseAddOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSlice, std::make_shared<Primitive>(kSparseSliceOpName));
GVAR_DEF(PrimitivePtr, kPrimSparseSliceGrad, std::make_shared<Primitive>(kSparseSliceGradOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SPARSE_OPS_H_

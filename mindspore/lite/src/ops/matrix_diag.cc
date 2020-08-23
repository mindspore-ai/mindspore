/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/matrix_diag.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int MatrixDiag::GetK() const { return this->primitive_->value.AsMatrixDiag()->k; }
int MatrixDiag::GetNumRows() const { return this->primitive_->value.AsMatrixDiag()->numRows; }
int MatrixDiag::GetNumCols() const { return this->primitive_->value.AsMatrixDiag()->numCols; }
float MatrixDiag::GetPaddingValue() const { return this->primitive_->value.AsMatrixDiag()->paddingValue; }

void MatrixDiag::SetK(int k) { this->primitive_->value.AsMatrixDiag()->k = k; }
void MatrixDiag::SetNumRows(int num_rows) { this->primitive_->value.AsMatrixDiag()->numRows = num_rows; }
void MatrixDiag::SetNumCols(int num_cols) { this->primitive_->value.AsMatrixDiag()->numCols = num_cols; }
void MatrixDiag::SetPaddingValue(float padding_value) {
  this->primitive_->value.AsMatrixDiag()->paddingValue = padding_value;
}

#else

int MatrixDiag::GetK() const { return this->primitive_->value_as_MatrixDiag()->k(); }
int MatrixDiag::GetNumRows() const { return this->primitive_->value_as_MatrixDiag()->numRows(); }
int MatrixDiag::GetNumCols() const { return this->primitive_->value_as_MatrixDiag()->numCols(); }
float MatrixDiag::GetPaddingValue() const { return this->primitive_->value_as_MatrixDiag()->paddingValue(); }

#endif
}  // namespace lite
}  // namespace mindspore

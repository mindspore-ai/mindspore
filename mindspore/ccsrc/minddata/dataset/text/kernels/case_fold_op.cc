/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/text/kernels/case_fold_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/utypes.h"

namespace mindspore {
namespace dataset {

Status CaseFoldOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "CaseFold: input is not string datatype.");
  icu::ErrorCode error;
  const icu::Normalizer2 *nfkc_case_fold = icu::Normalizer2::getNFKCCasefoldInstance(error);
  CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "CaseFold: getNFKCCasefoldInstance failed.");
  std::vector<std::string> strs(input->Size());
  int i = 0;
  for (auto iter = input->begin<std::string_view>(); iter != input->end<std::string_view>(); iter++) {
    icu::StringByteSink<std::string> sink(&strs[i++]);
    nfkc_case_fold->normalizeUTF8(0, icu::StringPiece((*iter).data(), (*iter).size()), sink, nullptr, error);
    CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "CaseFold: normalizeUTF8 failed.");
  }
  return Tensor::CreateFromVector(strs, input->shape(), output);
}
}  // namespace dataset
}  // namespace mindspore

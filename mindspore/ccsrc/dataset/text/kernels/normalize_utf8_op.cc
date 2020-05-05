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
#include "dataset/text/kernels/normalize_utf8_op.h"
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
const NormalizeForm NormalizeUTF8Op::kDefNormalizeForm = NormalizeForm::kNfkc;
Status NormalizeUTF8Op::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  icu::ErrorCode error;
  const icu::Normalizer2 *normalize = nullptr;
  switch (normalize_form_) {
    case NormalizeForm::kNone: {
      *output = input;
      return Status::OK();
    }
    case NormalizeForm::kNfc: {
      normalize = icu::Normalizer2::getNFCInstance(error);
      CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "getNFCInstance failed");
      break;
    }
    case NormalizeForm::kNfkc: {
      normalize = icu::Normalizer2::getNFKCInstance(error);
      CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "getNFKCInstance failed");
      break;
    }
    case NormalizeForm::kNfd: {
      normalize = icu::Normalizer2::getNFDInstance(error);
      CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "getNFDInstance failed");
      break;
    }
    case NormalizeForm::kNfkd: {
      normalize = icu::Normalizer2::getNFKDInstance(error);
      CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "getNFKDInstance failed");
      break;
    }
    default: {
      RETURN_STATUS_UNEXPECTED("unexpected normalize form");
      break;
    }
  }
  std::vector<std::string> strs(input->Size());
  int i = 0;
  for (auto iter = input->begin<std::string_view>(); iter != input->end<std::string_view>(); iter++) {
    icu::StringByteSink<std::string> sink(&strs[i++]);
    normalize->normalizeUTF8(0, icu::StringPiece((*iter).data(), (*iter).size()), sink, nullptr, error);
    CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "normalizeUTF8 failed.");
  }
  *output = std::make_shared<Tensor>(std::move(strs), input->shape());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore

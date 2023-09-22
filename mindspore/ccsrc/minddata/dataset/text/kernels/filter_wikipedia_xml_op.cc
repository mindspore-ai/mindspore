/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/text/kernels/filter_wikipedia_xml_op.h"

#include <memory>
#include <string_view>
#include <vector>

namespace mindspore {
namespace dataset {
std::map<icu::UnicodeString, icu::UnicodeString> patterns = {{R"(<.*>)", ""},
                                                             {R"(&amp;)", "&"},
                                                             {"&lt;", "<"},
                                                             {"&gt;", ">"},
                                                             {R"(<ef[^<]*<\/ef>)", ""},
                                                             {"<[^>]*>", ""},
                                                             {R"(\[http:[^] ]*)", "["},
                                                             {R"(\|thumb)", ""},
                                                             {R"(\|left)", ""},
                                                             {R"(\|right)", ""},
                                                             {R"(\|\d+px)", ""},
                                                             {R"(\[\[image:[^\[\]]*\|)", ""},
                                                             {R"(\[\[category:([^|\]]*)[^]]*\]\])", "[[$1]]"},
                                                             {R"(\[\[[a-z\-]*:[^\]]*\]\])", ""},
                                                             {R"(\[\[[^\|\]]*\|)", "[["},
                                                             {R"(\{\{[^\}]*\}\})", ""},
                                                             {R"(\{[^\}]*\})", ""},
                                                             {R"(\[)", ""},
                                                             {R"(\])", ""},
                                                             {"&[^;]*;", " "},
                                                             {"A", "a"},
                                                             {"B", "b"},
                                                             {"C", "c"},
                                                             {"D", "d"},
                                                             {"E", "e"},
                                                             {"F", "f"},
                                                             {"G", "g"},
                                                             {"H", "h"},
                                                             {"I", "i"},
                                                             {"J", "j"},
                                                             {"K", "k"},
                                                             {"L", "l"},
                                                             {"M", "m"},
                                                             {"N", "n"},
                                                             {"O", "o"},
                                                             {"P", "p"},
                                                             {"Q", "q"},
                                                             {"R", ""},
                                                             {"S", "s"},
                                                             {"T", "t"},
                                                             {"U", "u"},
                                                             {"V", "v"},
                                                             {"W", "w"},
                                                             {"X", "x"},
                                                             {"Y", "y"},
                                                             {"Z", "z"},
                                                             {"0", " zero "},
                                                             {"1", " one "},
                                                             {"2", " two "},
                                                             {"3", " three "},
                                                             {"4", " four "},
                                                             {"5", " five "},
                                                             {"6", " six "},
                                                             {"7", " seven "},
                                                             {"8", " eight "},
                                                             {"9", " nine "},
                                                             {R"([^a-z\n]+)", " "},
                                                             {R"(\n )", ""},
                                                             {R"(\s+)", " "},
                                                             {R"(\n\s*\n)", R"(\n)"}};

Status FilterWikipediaXMLOp::FilterWikipediaXML(const std::string_view &text, std::string *out) const {
  CHECK_FAIL_RETURN_UNEXPECTED((out != nullptr), "FilterWikipediaXML: icu init failed.");
  if (((text).find("#redirect") == std::string::npos) && ((text).find("#REDIRECT") == std::string::npos)) {
    (*out) = text;
    UErrorCode icu_error = U_ZERO_ERROR;
    for (auto pattern_iter = patterns.begin(); pattern_iter != patterns.end(); pattern_iter++) {
      icu::RegexMatcher matcher(pattern_iter->first, 0, icu_error);
      CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(icu_error),
                                   "RegexReplace: create icu RegexMatcher failed, you may input an error pattern.");
      icu::UnicodeString unicode_text = icu::UnicodeString::fromUTF8(*out);
      matcher.reset(unicode_text);
      icu::UnicodeString unicode_out = matcher.replaceAll(pattern_iter->second, icu_error);
      CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(icu_error), "FilterWikipediaXML: FilterWikipediaXML failed.");
      (*out) = "";
      unicode_out.trim().toUTF8String(*out);
    }
  } else {
    (*out) = "";
  }
  return Status::OK();
}

Status FilterWikipediaXMLOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "RegexReplace: input is not of type string.");
  std::vector<std::string> strs(input->Size());
  auto iter = input->begin<std::string_view>();
  RETURN_IF_NOT_OK(FilterWikipediaXML(*iter, &strs[0]));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(strs, input->shape(), output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore

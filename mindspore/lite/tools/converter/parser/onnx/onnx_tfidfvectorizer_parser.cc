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

#include "tools/converter/parser/onnx/onnx_tfidfvectorizer_parser.h"
#include <memory>
#include <string>
#include <vector>
#include "tools/converter/ops/ops_def.h"
#include "ir/value.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxTfIdfVectorizerParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::TfIdfVectorizer>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new TfIdfVectorizer prim failed.";
    return nullptr;
  }
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "max_gram_length") {
      int64_t max_gram_length = static_cast<int64_t>(onnx_node_attr.i());
      prim->set_attr("max_gram_length", MakeValue(max_gram_length));
    } else if (onnx_node_attr.name() == "max_skip_count") {
      int64_t max_skip_count = static_cast<int64_t>(onnx_node_attr.i());
      prim->set_attr("max_skip_count", MakeValue(max_skip_count));
    } else if (onnx_node_attr.name() == "min_gram_length") {
      int64_t min_gram_length = static_cast<int64_t>(onnx_node_attr.i());
      prim->set_attr("min_gram_length", MakeValue(min_gram_length));
    } else if (onnx_node_attr.name() == "mode") {
      std::string mode = onnx_node_attr.s();
      prim->set_attr("mode", MakeValue(mode));
    } else if (onnx_node_attr.name() == "ngram_counts") {
      std::vector<int64_t> ngram_counts;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        ngram_counts.push_back(static_cast<int64_t>(onnx_node_attr.ints(i)));
      }
      prim->set_attr("ngram_counts", MakeValue(ngram_counts));
    } else if (onnx_node_attr.name() == "ngram_indexes") {
      std::vector<int64_t> ngram_indexes;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        ngram_indexes.push_back(static_cast<int64_t>(onnx_node_attr.ints(i)));
      }
      prim->set_attr("ngram_indexes", MakeValue(ngram_indexes));
    } else if (onnx_node_attr.name() == "pool_int64s") {
      // pool_int64s is not required attr
      std::vector<int64_t> pool_int64s;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        pool_int64s.push_back(static_cast<int64_t>(onnx_node_attr.ints(i)));
      }
      prim->set_attr("pool_int64s", MakeValue(pool_int64s));
    } else if (onnx_node_attr.name() == "pool_strings") {
      // pool_strings is not required attr
      std::vector<std::string> pool_strings;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        pool_strings.push_back(onnx_node_attr.strings(i));
      }
      prim->set_attr("pool_strings", MakeValue(pool_strings));
    } else if (onnx_node_attr.name() == "weights") {
      // weights is not required attr
      std::vector<float> weights;
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        weights.push_back(onnx_node_attr.floats(i));
      }
      prim->set_attr("weights", MakeValue(weights));
    }
  }

  return prim;
}
OnnxNodeRegistrar g_onnxTfIdfVectorizerParser("TfIdfVectorizer", new OnnxTfIdfVectorizerParser());
}  // namespace lite
}  // namespace mindspore

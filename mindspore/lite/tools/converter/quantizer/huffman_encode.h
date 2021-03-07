/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_HUFFMANCODE_HUFFMAN_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_HUFFMANCODE_HUFFMAN_H

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <memory>
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "schema/inner/model_generated.h"
#include "securec/include/securec.h"
#include "src/common/log_adapter.h"
#include "src/param_value_lite.h"

namespace mindspore {
namespace lite {

using STATUS = int;

const int PSEUDO_EOF = 128;

struct HuffmanNode {
  int key;
  unsigned int freq;
  std::string code;
  HuffmanNode *left, *right, *parent;
};
using HuffmanNodePtr = HuffmanNode *;

struct cmp {
 public:
  bool operator()(const HuffmanNodePtr &c1, const HuffmanNodePtr &c2) const { return c1->freq > c2->freq; }
};
using HuffmanPriorityQueue = std::priority_queue<HuffmanNodePtr, std::vector<HuffmanNodePtr>, cmp>;

class HuffmanEncode {
 public:
  HuffmanEncode() = default;

  ~HuffmanEncode();

  STATUS DoHuffmanEncode(const ParamValueLitePtr &weight, const PrimitivePtr &primitive, void *quant_datas,
                         const size_t &bit_num);

 private:
  std::map<int, std::string> huffman_table_;
  std::string huffman_encoded_str_ = "";
  std::vector<HuffmanNodePtr> huffman_nodes_;

  STATUS GetHuffmanPriorityQueue(const int8_t *input_datas, size_t input_data_size, HuffmanPriorityQueue *pq);

  void GenerateHuffmanTable(HuffmanNodePtr node, bool is_left_node);

  STATUS BuildHuffmanTree(HuffmanPriorityQueue *pq);

  STATUS DoHuffmanCompress(const int8_t *input_datas, size_t data_size);
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_HUFFMANCODE_HUFFMAN_H

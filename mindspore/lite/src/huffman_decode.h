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

#ifndef MINDSPORE_LITE_MINDSPORE_LITE_SRC_HUFFMAN_DECODE_H_
#define MINDSPORE_LITE_MINDSPORE_LITE_SRC_HUFFMAN_DECODE_H_

#include <cstring>
#include <utility>
#include <string>
#include <vector>

#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace lite {

const int PSEUDO_EOF = 128;

struct HuffmanNode {
  int key;
  unsigned int freq;
  std::string code;
  HuffmanNode *left, *right, *parent;
};
using HuffmanNodePtr = HuffmanNode *;

class HuffmanDecode {
 public:
  HuffmanDecode() = default;

  ~HuffmanDecode();

  STATUS DoHuffmanDecode(const std::string &input_str, void *decoded_data);

 private:
  std::vector<HuffmanNodePtr> huffman_nodes_;
  STATUS RebuildHuffmanTree(std::string key, std::string code, const HuffmanNodePtr &root);

  STATUS DoHuffmanDecompress(HuffmanNodePtr root, std::string encoded_data, std::string *decoded_str);

  std::vector<std::string> Str2Vec(std::string s) {
    size_t i = 0;
    std::vector<std::string> vec;
    while (i < s.length()) {
      size_t j = i;
      while (j < s.length() && s[j] != ' ') {
        j++;
      }
      if (j != i) {
        vec.push_back(s.substr(i, j - i));
        i = j + 1;
      } else {
        i = j;
      }
    }
    return vec;
  }
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_MINDSPORE_LITE_SRC_HUFFMAN_DECODE_H_

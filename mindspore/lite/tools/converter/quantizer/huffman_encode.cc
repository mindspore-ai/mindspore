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

#include "tools/converter/quantizer/huffman_encode.h"
#include <utility>
#include "src/dequant.h"
#include "tools/converter/quantizer/quantize_util.h"

namespace mindspore {
namespace lite {
STATUS HuffmanEncode::DoHuffmanEncode(const ParamValueLitePtr &weight, const PrimitivePtr &primitive, void *quant_datas,
                                      const size_t &bit_num) {
  if (quant_datas == nullptr) {
    MS_LOG(ERROR) << "quant data is nullptr";
    return RET_ERROR;
  }
  auto *raw_datas = static_cast<int8_t *>(quant_datas);
  size_t elem_count = weight->tensor_shape_size();
  size_t packed_size = elem_count * bit_num;

  HuffmanPriorityQueue pq;
  auto status = GetHuffmanPriorityQueue(raw_datas, elem_count, &pq);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "GetHuffmanPriorityQueue failed";
    return status;
  }
  status = BuildHuffmanTree(&pq);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "BuildHuffmanTree failed";
    return status;
  }
  status = DoHuffmanCompress(raw_datas, elem_count);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoHuffmanCompress failed";
    return status;
  }
  size_t ch_size = huffman_encoded_str_.length();
  if (ch_size < packed_size) {
    auto encode_data = new (std::nothrow) char[ch_size];
    if (encode_data == nullptr) {
      MS_LOG(ERROR) << "new char[] failed.";
      return RET_MEMORY_FAILED;
    }
    if (memcpy_s(encode_data, ch_size, huffman_encoded_str_.c_str(), ch_size) != EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      delete[] encode_data;
      return RET_MEMORY_FAILED;
    }
    weight->SetTensorData(encode_data, ch_size);
    auto quant_param_holder = quant::GetCNodeQuantHolder(primitive);
    MS_ASSERT(quant_param_holder != nullptr);
    quant_param_holder->set_enable_huffman_code(true);
  }
  huffman_encoded_str_.clear();
  huffman_table_.clear();
  return RET_SUCCESS;
}

STATUS HuffmanEncode::GetHuffmanPriorityQueue(const int8_t *data, const size_t data_size, HuffmanPriorityQueue *pq) {
  MS_ASSERT(data != nullptr);

  std::map<int8_t, size_t> freq_map;

  for (size_t i = 0; i < data_size; i++) {
    freq_map[data[i]]++;
  }

  for (auto &kv : freq_map) {
    if (kv.second <= 0) {
      continue;
    }
    auto node = new (std::nothrow) HuffmanNode();
    if (node == nullptr) {
      MS_LOG(ERROR) << "new HuffmanNode failed.";
      return RET_MEMORY_FAILED;
    }
    this->huffman_nodes_.push_back(node);
    node->key = kv.first;
    node->freq = kv.second;
    node->code = "";
    node->left = nullptr;
    node->right = nullptr;
    node->parent = nullptr;

    pq->push(node);
  }

  // insert pseudo-EOF
  auto node = new (std::nothrow) HuffmanNode();
  if (node == nullptr) {
    MS_LOG(ERROR) << "new HuffmanNode failed.";
    return RET_MEMORY_FAILED;
  }
  this->huffman_nodes_.push_back(node);
  node->key = PSEUDO_EOF;
  node->freq = 1;
  node->code = "";
  node->left = nullptr;
  node->right = nullptr;
  node->parent = nullptr;

  pq->push(node);

  return RET_OK;
}

void HuffmanEncode::GenerateHuffmanTable(const HuffmanNodePtr node, bool is_left_node) {
  if (is_left_node) {
    node->code = node->parent->code + "0";
  } else {
    node->code = node->parent->code + "1";
  }

  if (node->left == nullptr && node->right == nullptr) {
    huffman_table_[node->key] = node->code;
  } else {
    if (node->left != nullptr) {
      GenerateHuffmanTable(node->left, true);
    }
    if (node->right != nullptr) {
      GenerateHuffmanTable(node->right, false);
    }
  }
}

STATUS HuffmanEncode::BuildHuffmanTree(HuffmanPriorityQueue *pq) {
  HuffmanNodePtr root = nullptr;

  while (!pq->empty()) {
    HuffmanNodePtr first = pq->top();
    pq->pop();

    if (pq->empty()) {
      root = first;
      break;
    }

    HuffmanNodePtr second = pq->top();
    pq->pop();

    auto new_node = new (std::nothrow) HuffmanNode();
    if (new_node == nullptr) {
      MS_LOG(ERROR) << "new HuffmanNode failed.";
      return RET_MEMORY_FAILED;
    }
    this->huffman_nodes_.push_back(new_node);
    new_node->freq = first->freq + second->freq;
    new_node->left = first;
    new_node->right = second;
    first->parent = new_node;
    second->parent = new_node;

    pq->push(new_node);
  }

  if (root == nullptr) {
    MS_LOG(ERROR) << "huffman tree root node is nullptr.";
    return RET_ERROR;
  }

  if (root->left != nullptr) {
    GenerateHuffmanTable(root->left, true);
  }
  if (root->right != nullptr) GenerateHuffmanTable(root->right, false);

  return RET_OK;
}

STATUS HuffmanEncode::DoHuffmanCompress(const int8_t *input_datas, const size_t data_size) {
  unsigned char out_c;
  string code_str;
  std::map<int, string>::iterator iter;
  std::vector<std::string> encode_str = {"", "", ""};

  huffman_encoded_str_.clear();
  for (iter = huffman_table_.begin(); iter != huffman_table_.end(); ++iter) {
    encode_str[0] += std::to_string(iter->first) + " ";
    encode_str[1] += iter->second + " ";
  }

  for (size_t i = 0; i < data_size; i++) {
    auto raw_num = input_datas[i];
    iter = huffman_table_.find(raw_num);
    if (iter != huffman_table_.end()) {
      code_str += iter->second;
    } else {
      MS_LOG(ERROR) << "Can't find the huffman code " << raw_num;
      return RET_ERROR;
    }
  }
  iter = huffman_table_.find(PSEUDO_EOF);
  if (iter != huffman_table_.end()) {
    code_str += iter->second;
  } else {
    MS_LOG(ERROR) << "Can't find the huffman code pseudo-EOF";
    return RET_ERROR;
  }
  out_c = 0;
  for (size_t i = 0; i < code_str.length(); i++) {
    auto tmp_c = code_str[i] == '0' ? 0 : 1;
    out_c += tmp_c << (7 - (i % 8));
    if (0 == (i + 1) % 8 || i == code_str.length() - 1) {
      encode_str[2] += out_c;
      out_c = 0;
    }
  }
  huffman_encoded_str_ = encode_str[0] + "#" + encode_str[1] + "#" + encode_str[2];
  return RET_OK;
}

HuffmanEncode::~HuffmanEncode() {
  for (auto &node : this->huffman_nodes_) {
    delete node;
  }
  this->huffman_nodes_.resize(0);
}

}  // namespace lite
}  // namespace mindspore

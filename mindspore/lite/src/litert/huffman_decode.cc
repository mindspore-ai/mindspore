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

#include "src/litert/huffman_decode.h"
#include <queue>

namespace mindspore {
namespace lite {
STATUS HuffmanDecode::DoHuffmanDecode(const std::string &input_str, void *decoded_data, size_t data_len) {
  if (decoded_data == nullptr) {
    MS_LOG(ERROR) << "decoded_data is nullptr.";
    return RET_ERROR;
  }

  int status;
  std::string huffman_decoded_str;
  auto key_pos = input_str.find_first_of('#');
  auto code_pos = input_str.find_first_of('#', key_pos + 1);
  if (key_pos == std::string::npos || code_pos == std::string::npos) {
    MS_LOG(ERROR) << "not found '#' in input_str";
    return RET_ERROR;
  }
  if (key_pos + 1 > input_str.size() || code_pos + 1 > input_str.size()) {
    MS_LOG(ERROR) << "pos extend input_str size.";
    return RET_ERROR;
  }
  auto key = input_str.substr(0, key_pos);
  auto code = input_str.substr(key_pos + 1, code_pos - key_pos - 1);
  auto encoded_data = input_str.substr(code_pos + 1);

  auto root = new (std::nothrow) HuffmanNode();
  if (root == nullptr) {
    MS_LOG(ERROR) << "new HuffmanNode failed.";
    return RET_MEMORY_FAILED;
  }
  root->left = nullptr;
  root->right = nullptr;
  root->parent = nullptr;

  status = RebuildHuffmanTree(key, code, root);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Rebuild huffman tree failed.";
    delete root;
    return status;
  }

  status = DoHuffmanDecompress(root, encoded_data, &huffman_decoded_str);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "DoHuffmanDecompress failed.";
    delete root;
    return status;
  }

  size_t len = huffman_decoded_str.length();
  if (data_len >= len) {
    memcpy(decoded_data, huffman_decoded_str.c_str(), len);
  } else {
    FreeHuffmanNodeTree(root);
    return RET_ERROR;
  }
  FreeHuffmanNodeTree(root);
  return RET_OK;
}

STATUS HuffmanDecode::RebuildHuffmanTree(std::string keys, std::string codes, const HuffmanNodePtr &root) {
  HuffmanNodePtr cur_node;
  HuffmanNodePtr tmp_node;
  HuffmanNodePtr new_node;

  auto huffman_keys = Str2Vec(std::move(keys));
  auto huffman_codes = Str2Vec(std::move(codes));

  for (size_t i = 0; i < huffman_codes.size(); ++i) {
    auto key = stoi(huffman_keys[i]);
    auto code = huffman_codes[i];
    auto code_len = code.length();
    cur_node = root;
    for (size_t j = 0; j < code_len; ++j) {
      if (code[j] == '0') {
        tmp_node = cur_node->left;
      } else if (code[j] == '1') {
        tmp_node = cur_node->right;
      } else {
        MS_LOG(ERROR) << "find huffman code is not 0 or 1";
        return RET_ERROR;
      }

      if (tmp_node == nullptr) {
        new_node = new (std::nothrow) HuffmanNode();
        if (new_node == nullptr) {
          MS_LOG(ERROR) << "new HuffmanNode failed.";
          return RET_MEMORY_FAILED;
        }
        new_node->left = nullptr;
        new_node->right = nullptr;
        new_node->parent = cur_node;

        if (j == code_len - 1) {
          new_node->key = key;
          new_node->code = code;
        }

        if (code[j] == '0') {
          cur_node->left = new_node;
        } else {
          cur_node->right = new_node;
        }

        tmp_node = new_node;
      } else if (j == code_len - 1) {
        MS_LOG(ERROR) << "the huffman code is incomplete.";
        return RET_ERROR;
      } else if (tmp_node->left == nullptr && tmp_node->right == nullptr) {
        MS_LOG(ERROR) << "the huffman code is incomplete";
        return RET_ERROR;
      }
      cur_node = tmp_node;
    }
  }
  return RET_OK;
}

STATUS HuffmanDecode::DoHuffmanDecompress(HuffmanNodePtr root, std::string encoded_data, std::string *decoded_str) {
  HuffmanNodePtr cur_node = root;
  bool pseudo_eof = false;
  size_t pos = 0;
  unsigned char flag;

  decoded_str->clear();
  while (pos < encoded_data.length()) {
    auto u_char = static_cast<unsigned char>(encoded_data[pos]);
    flag = 0x80;
    for (size_t i = 0; i < 8; ++i) {  // traverse the 8 bit num, to find the leaf node
      if (u_char & flag) {
        cur_node = cur_node->right;
      } else {
        cur_node = cur_node->left;
      }
      if (cur_node->left == nullptr && cur_node->right == nullptr) {
        auto key = cur_node->key;
        if (key == PSEUDO_EOF) {
          pseudo_eof = true;
          break;
        } else {
          *decoded_str += static_cast<char>(cur_node->key);
          cur_node = root;
        }
      }
      flag = flag >> 1;
    }
    pos++;
    if (pseudo_eof) {
      break;
    }
  }
  return RET_OK;
}

void HuffmanDecode::FreeHuffmanNodeTree(HuffmanNodePtr root) {
  if (root == nullptr) {
    return;
  }
  std::queue<HuffmanNodePtr> node_queue;
  node_queue.push(root);
  while (!node_queue.empty()) {
    auto cur_node = node_queue.front();
    node_queue.pop();
    if (cur_node->left != nullptr) {
      node_queue.push(cur_node->left);
    }
    if (cur_node->right != nullptr) {
      node_queue.push(cur_node->right);
    }
    delete (cur_node);
  }
}
}  // namespace lite
}  // namespace mindspore

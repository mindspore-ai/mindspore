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

#include "utils/crypto.h"
#include <regex>
#include <vector>
#include <fstream>
#include <algorithm>
#include "utils/log_adapter.h"

#ifdef ENABLE_OPENSSL
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

namespace mindspore {
void IntToByte(std::vector<Byte> *byteArray, int32_t n) {
  if (byteArray == nullptr) {
    MS_LOG(ERROR) << "byteArray is nullptr";
    return;
  }
  auto ptr = reinterpret_cast<const Byte *>(&n);
  (*byteArray).assign(ptr, ptr + sizeof(int32_t));
}

int32_t ByteToInt(const Byte *byteArray, size_t length) {
  if (length < sizeof(int32_t)) {
    MS_LOG(ERROR) << "Length of byteArray is " << length << ", less than sizeof(int32_t): 4.";
    return -1;
  }
  return *(reinterpret_cast<const int32_t *>(byteArray));
}

bool IsCipherFile(const std::string &file_path) {
  std::ifstream fid(file_path, std::ios::in | std::ios::binary);
  if (!fid) {
    MS_LOG(ERROR) << "Failed to open file " << file_path;
    return false;
  }
  std::vector<char> int_buf(sizeof(int32_t));
  fid.read(int_buf.data(), static_cast<int64_t>(sizeof(int32_t)));
  fid.close();
  auto flag = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
  return static_cast<unsigned int>(flag) == MAGIC_NUM;
}

bool IsCipherFile(const Byte *model_data) {
  MS_EXCEPTION_IF_NULL(model_data);
  std::vector<Byte> int_buf;
  int_buf.assign(model_data, model_data + sizeof(int32_t));
  auto flag = ByteToInt(int_buf.data(), int_buf.size());
  return static_cast<unsigned int>(flag) == MAGIC_NUM;
}
#ifndef ENABLE_OPENSSL
std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, const Byte *plain_data, size_t plain_len, const Byte *key,
                                size_t key_len, const std::string &enc_mode) {
  MS_LOG(ERROR) << "The feature is only supported on the Linux platform "
                   "when the OPENSSL compilation option is enabled.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                size_t key_len, const std::string &dec_mode) {
  MS_LOG(ERROR) << "The feature is only supported on the Linux platform "
                   "when the OPENSSL compilation option is enabled.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, size_t data_size, const Byte *key,
                                size_t key_len, const std::string &dec_mode) {
  MS_LOG(ERROR) << "The feature is only supported on the Linux platform "
                   "when the OPENSSL compilation option is enabled.";
  return nullptr;
}
#else
bool ParseEncryptData(const Byte *encrypt_data, size_t encrypt_len, std::vector<Byte> *iv,
                      std::vector<Byte> *cipher_data) {
  // encrypt_data is organized in order to iv_len, iv, cipher_len, cipher_data
  std::vector<Byte> int_buf(sizeof(int32_t));
  int_buf.assign(encrypt_data, encrypt_data + sizeof(int32_t));
  auto iv_len = ByteToInt(int_buf.data(), int_buf.size());

  int_buf.assign(encrypt_data + iv_len + sizeof(int32_t), encrypt_data + iv_len + sizeof(int32_t) + sizeof(int32_t));
  auto cipher_len = ByteToInt(int_buf.data(), int_buf.size());
  if (iv_len <= 0 || cipher_len <= 0 ||
      ((static_cast<size_t>(iv_len) + sizeof(int32_t) + static_cast<size_t>(cipher_len) + sizeof(int32_t)) !=
       encrypt_len)) {
    MS_LOG(ERROR) << "Failed to parse encrypt data.";
    return false;
  }

  (*iv).assign(encrypt_data + sizeof(int32_t), encrypt_data + sizeof(int32_t) + iv_len);
  (*cipher_data)
    .assign(encrypt_data + sizeof(int32_t) + iv_len + sizeof(int32_t),
            encrypt_data + sizeof(int32_t) + iv_len + sizeof(int32_t) + cipher_len);
  return true;
}

bool ParseMode(const std::string &mode, std::string *alg_mode, std::string *work_mode) {
  std::smatch results;
  std::regex re("([A-Z]{3})-([A-Z]{3})");
  if (!(std::regex_match(mode.c_str(), re) && std::regex_search(mode, results, re))) {
    MS_LOG(ERROR) << "Mode " << mode << " is invalid.";
    return false;
  }
  *alg_mode = results[1];
  *work_mode = results[2];
  return true;
}

EVP_CIPHER_CTX *GetEvpCipherCtx(const std::string &work_mode, const Byte *key, int32_t key_len, const Byte *iv,
                                bool is_encrypt) {
  constexpr int32_t key_length_16 = 16;
  constexpr int32_t key_length_24 = 24;
  constexpr int32_t key_length_32 = 32;
  const EVP_CIPHER *(*funcPtr)() = nullptr;
  if (work_mode == "GCM") {
    switch (key_len) {
      case key_length_16:
        funcPtr = EVP_aes_128_gcm;
        break;
      case key_length_24:
        funcPtr = EVP_aes_192_gcm;
        break;
      case key_length_32:
        funcPtr = EVP_aes_256_gcm;
        break;
      default:
        MS_LOG(ERROR) << "The key length must be 16, 24 or 32, but got key length is " << key_len << ".";
        return nullptr;
    }
  } else if (work_mode == "CBC") {
    switch (key_len) {
      case key_length_16:
        funcPtr = EVP_aes_128_cbc;
        break;
      case key_length_24:
        funcPtr = EVP_aes_192_cbc;
        break;
      case key_length_32:
        funcPtr = EVP_aes_256_cbc;
        break;
      default:
        MS_LOG(ERROR) << "The key length must be 16, 24 or 32, but got key length is " << key_len << ".";
        return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "Work mode " << work_mode << " is invalid.";
    return nullptr;
  }

  int32_t ret = 0;
  auto ctx = EVP_CIPHER_CTX_new();
  if (is_encrypt) {
    ret = EVP_EncryptInit_ex(ctx, funcPtr(), NULL, key, iv);
  } else {
    ret = EVP_DecryptInit_ex(ctx, funcPtr(), NULL, key, iv);
  }

  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptInit_ex failed";
    return nullptr;
  }
  if (work_mode == "CBC") {
    ret = EVP_CIPHER_CTX_set_padding(ctx, 1);
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_CIPHER_CTX_set_padding failed";
      return nullptr;
    }
  }
  return ctx;
}

bool BlockEncrypt(Byte *encrypt_data, size_t *encrypt_data_len, const std::vector<Byte> &plain_data, const Byte *key,
                  int32_t key_len, const std::string &enc_mode) {
  size_t encrypt_data_buf_len = *encrypt_data_len;
  int32_t cipher_len = 0;
  int32_t iv_len = AES_BLOCK_SIZE;
  std::vector<Byte> iv(iv_len);
  auto ret = RAND_bytes(iv.data(), iv_len);
  if (ret != 1) {
    MS_LOG(ERROR) << "RAND_bytes error, failed to init iv.";
    return false;
  }
  std::vector<Byte> iv_cpy(iv);

  std::string alg_mode;
  std::string work_mode;
  if (!ParseMode(enc_mode, &alg_mode, &work_mode)) {
    return false;
  }

  auto ctx = GetEvpCipherCtx(work_mode, key, key_len, iv.data(), true);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }

  std::vector<Byte> cipher_data_buf(plain_data.size() + AES_BLOCK_SIZE);
  auto ret_evp = EVP_EncryptUpdate(ctx, cipher_data_buf.data(), &cipher_len, plain_data.data(),
                                   static_cast<int32_t>(plain_data.size()));
  if (ret_evp != 1) {
    MS_LOG(ERROR) << "EVP_EncryptUpdate failed";
    return false;
  }
  if (work_mode == "CBC") {
    int32_t flen = 0;
    ret_evp = EVP_EncryptFinal_ex(ctx, cipher_data_buf.data() + cipher_len, &flen);
    if (ret_evp != 1) {
      MS_LOG(ERROR) << "EVP_EncryptFinal_ex failed";
      return false;
    }
    cipher_len += flen;
  }
  EVP_CIPHER_CTX_free(ctx);

  size_t offset = 0;
  std::vector<Byte> int_buf(sizeof(int32_t));
  *encrypt_data_len = sizeof(int32_t) + static_cast<size_t>(iv_len) + sizeof(int32_t) + static_cast<size_t>(cipher_len);
  IntToByte(&int_buf, static_cast<int32_t>(*encrypt_data_len));
  ret = memcpy_s(encrypt_data, encrypt_data_buf_len, int_buf.data(), int_buf.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  IntToByte(&int_buf, iv_len);
  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, int_buf.data(), int_buf.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, iv_cpy.data(), iv_cpy.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += iv_cpy.size();

  IntToByte(&int_buf, cipher_len);
  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, int_buf.data(), int_buf.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, cipher_data_buf.data(),
                 static_cast<size_t>(cipher_len));
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }

  *encrypt_data_len += sizeof(int32_t);
  return true;
}

bool BlockDecrypt(Byte *plain_data, int32_t *plain_len, const Byte *encrypt_data, size_t encrypt_len, const Byte *key,
                  int32_t key_len, const std::string &dec_mode) {
  std::string alg_mode;
  std::string work_mode;
  if (!ParseMode(dec_mode, &alg_mode, &work_mode)) {
    return false;
  }
  std::vector<Byte> iv;
  std::vector<Byte> cipher_data;
  if (!ParseEncryptData(encrypt_data, encrypt_len, &iv, &cipher_data)) {
    return false;
  }
  auto ctx = GetEvpCipherCtx(work_mode, key, key_len, iv.data(), false);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }
  auto ret =
    EVP_DecryptUpdate(ctx, plain_data, plain_len, cipher_data.data(), static_cast<int32_t>(cipher_data.size()));
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptUpdate failed";
    return false;
  }
  if (work_mode == "CBC") {
    int32_t mlen = 0;
    ret = EVP_DecryptFinal_ex(ctx, plain_data + *plain_len, &mlen);
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_DecryptFinal_ex failed";
      return false;
    }
    *plain_len += mlen;
  }
  EVP_CIPHER_CTX_free(ctx);
  return true;
}

std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, const Byte *plain_data, size_t plain_len, const Byte *key,
                                size_t key_len, const std::string &enc_mode) {
  MS_EXCEPTION_IF_NULL(plain_data);
  MS_EXCEPTION_IF_NULL(key);

  size_t block_enc_buf_len = MAX_BLOCK_SIZE + RESERVED_BYTE_PER_BLOCK;
  size_t encrypt_buf_len = plain_len + (plain_len + MAX_BLOCK_SIZE) / MAX_BLOCK_SIZE * RESERVED_BYTE_PER_BLOCK;
  std::vector<Byte> int_buf(sizeof(int32_t));
  std::vector<Byte> block_buf;
  std::vector<Byte> block_enc_buf(block_enc_buf_len);
  auto encrypt_data = std::make_unique<Byte[]>(encrypt_buf_len);

  size_t offset = 0;
  *encrypt_len = 0;
  while (offset < plain_len) {
    size_t block_enc_len = block_enc_buf.size();
    size_t cur_block_size = std::min(MAX_BLOCK_SIZE, plain_len - offset);
    block_buf.assign(plain_data + offset, plain_data + offset + cur_block_size);
    if (!BlockEncrypt(block_enc_buf.data(), &block_enc_len, block_buf, key, static_cast<int32_t>(key_len), enc_mode)) {
      MS_LOG(ERROR) << "Failed to encrypt data, please check if enc_key or enc_mode is valid.";
      return nullptr;
    }

    IntToByte(&int_buf, static_cast<int32_t>(MAGIC_NUM));
    size_t capacity = std::min(encrypt_buf_len - *encrypt_len, SECUREC_MEM_MAX_LEN);  // avoid dest size over 2gb
    auto ret = memcpy_s(encrypt_data.get() + *encrypt_len, capacity, int_buf.data(), sizeof(int32_t));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *encrypt_len += sizeof(int32_t);

    capacity = std::min(encrypt_buf_len - *encrypt_len, SECUREC_MEM_MAX_LEN);
    ret = memcpy_s(encrypt_data.get() + *encrypt_len, capacity, block_enc_buf.data(), block_enc_len);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *encrypt_len += block_enc_len;
    offset += cur_block_size;
  }
  return encrypt_data;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                size_t key_len, const std::string &dec_mode) {
  MS_EXCEPTION_IF_NULL(key);

  std::ifstream fid(encrypt_data_path, std::ios::in | std::ios::binary);
  if (!fid) {
    MS_LOG(ERROR) << "Open file '" << encrypt_data_path << "' failed, please check the correct of the file.";
    return nullptr;
  }
  fid.seekg(0, std::ios_base::end);
  size_t file_size = static_cast<size_t>(fid.tellg());
  fid.clear();
  fid.seekg(0);

  std::vector<char> block_buf(MAX_BLOCK_SIZE + RESERVED_BYTE_PER_BLOCK);
  std::vector<char> int_buf(sizeof(int32_t));
  std::vector<Byte> decrypt_block_buf(MAX_BLOCK_SIZE);
  auto decrypt_data = std::make_unique<Byte[]>(file_size);
  int32_t decrypt_block_len;

  *decrypt_len = 0;
  while (static_cast<size_t>(fid.tellg()) < file_size) {
    fid.read(int_buf.data(), static_cast<int32_t>(sizeof(int32_t)));
    auto cipher_flag = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (static_cast<unsigned int>(cipher_flag) != MAGIC_NUM) {
      MS_LOG(ERROR) << "File \"" << encrypt_data_path << "\" is not an encrypted file and cannot be decrypted";
      return nullptr;
    }
    fid.read(int_buf.data(), static_cast<int64_t>(sizeof(int32_t)));
    auto block_size = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (block_size < 0) {
      MS_LOG(ERROR) << "The block_size read from the cipher file must be not negative, but got " << block_size;
      return nullptr;
    }
    fid.read(block_buf.data(), static_cast<int64_t>(block_size));
    if (!(BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                       static_cast<size_t>(block_size), key, static_cast<int32_t>(key_len), dec_mode))) {
      MS_LOG(ERROR) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
      return nullptr;
    }
    size_t capacity = std::min(file_size - *decrypt_len, SECUREC_MEM_MAX_LEN);
    auto ret = memcpy_s(decrypt_data.get() + *decrypt_len, capacity, decrypt_block_buf.data(),
                        static_cast<int32_t>(decrypt_block_len));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *decrypt_len += static_cast<size_t>(decrypt_block_len);
  }
  fid.close();
  return decrypt_data;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, size_t data_size, const Byte *key,
                                size_t key_len, const std::string &dec_mode) {
  MS_EXCEPTION_IF_NULL(model_data);
  MS_EXCEPTION_IF_NULL(key);

  std::vector<char> block_buf;
  std::vector<char> int_buf(sizeof(int32_t));
  std::vector<Byte> decrypt_block_buf(MAX_BLOCK_SIZE);
  auto decrypt_data = std::make_unique<Byte[]>(data_size);
  int32_t decrypt_block_len;

  size_t offset = 0;
  *decrypt_len = 0;
  while (offset < data_size) {
    int_buf.assign(model_data + offset, model_data + offset + sizeof(int32_t));
    offset += int_buf.size();
    auto cipher_flag = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (static_cast<unsigned int>(cipher_flag) != MAGIC_NUM) {
      MS_LOG(ERROR) << "model_data is not encrypted and therefore cannot be decrypted.";
      return nullptr;
    }

    int_buf.assign(model_data + offset, model_data + offset + sizeof(int32_t));
    offset += int_buf.size();
    auto block_size = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (block_size < 0) {
      MS_LOG(ERROR) << "The block_size read from the cipher data must be not negative, but got " << block_size;
      return nullptr;
    }
    block_buf.assign(model_data + offset, model_data + offset + block_size);
    offset += block_buf.size();
    if (!(BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                       block_buf.size(), key, static_cast<int32_t>(key_len), dec_mode))) {
      MS_LOG(ERROR) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
      return nullptr;
    }
    size_t capacity = std::min(data_size - *decrypt_len, SECUREC_MEM_MAX_LEN);
    auto ret = memcpy_s(decrypt_data.get() + *decrypt_len, capacity, decrypt_block_buf.data(),
                        static_cast<size_t>(decrypt_block_len));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *decrypt_len += static_cast<size_t>(decrypt_block_len);
  }
  return decrypt_data;
}
#endif
}  // namespace mindspore

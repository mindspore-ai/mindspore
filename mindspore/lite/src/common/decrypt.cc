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

#include "src/common/decrypt.h"
#ifdef ENABLE_OPENSSL
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <regex>
#include <vector>
#include <fstream>
#include <algorithm>
#include "src/common/dynamic_library_loader.h"
#include "src/common/file_utils.h"
#endif
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

#ifndef SECUREC_MEM_MAX_LEN
#define SECUREC_MEM_MAX_LEN 0x7fffffffUL
#endif

namespace mindspore::lite {
#ifndef ENABLE_OPENSSL
std::unique_ptr<Byte[]> Decrypt(const std::string &lib_path, size_t *, const Byte *, const size_t, const Byte *,
                                const size_t, const std::string &) {
  MS_LOG(ERROR) << "The feature is only supported on the Linux platform "
                   "when the OPENSSL compilation option is enabled.";
  return nullptr;
}
#else
namespace {
constexpr size_t MAX_BLOCK_SIZE = 64 * 1024 * 1024;  // Maximum ciphertext segment, units is Byte
constexpr size_t Byte16 = 16;                        // Byte16
constexpr unsigned int MAGIC_NUM = 0x7F3A5ED8;       // Magic number
DynamicLibraryLoader loader;
}  // namespace
int32_t ByteToInt(const Byte *byteArray, size_t length) {
  if (byteArray == nullptr) {
    MS_LOG(ERROR) << "There is a null pointer in the input parameter.";
    return -1;
  }
  if (length < sizeof(int32_t)) {
    MS_LOG(ERROR) << "Length of byteArray is " << length << ", less than sizeof(int32_t): 4.";
    return -1;
  }
  return *(reinterpret_cast<const int32_t *>(byteArray));
}

bool ParseEncryptData(const Byte *encrypt_data, size_t encrypt_len, std::vector<Byte> *iv,
                      std::vector<Byte> *cipher_data) {
  if (encrypt_data == nullptr || iv == nullptr || cipher_data == nullptr) {
    MS_LOG(ERROR) << "There is a null pointer in the input parameter.";
    return false;
  }
  // encrypt_data is organized in order to iv_len, iv, cipher_len, cipher_data
  std::vector<Byte> int_buf(sizeof(int32_t));
  if (sizeof(int32_t) > encrypt_len) {
    MS_LOG(ERROR) << "assign len is invalid.";
    return false;
  }
  int_buf.assign(encrypt_data, encrypt_data + sizeof(int32_t));
  auto iv_len = ByteToInt(int_buf.data(), int_buf.size());
  if (iv_len <= 0 || static_cast<size_t>(iv_len) + sizeof(int32_t) + sizeof(int32_t) > encrypt_len) {
    MS_LOG(ERROR) << "assign len is invalid.";
    return false;
  }
  int_buf.assign(encrypt_data + iv_len + sizeof(int32_t), encrypt_data + iv_len + sizeof(int32_t) + sizeof(int32_t));
  auto cipher_len = ByteToInt(int_buf.data(), int_buf.size());
  if (((static_cast<size_t>(iv_len) + sizeof(int32_t) + static_cast<size_t>(cipher_len) + sizeof(int32_t)) !=
       encrypt_len)) {
    MS_LOG(ERROR) << "Failed to parse encrypt data.";
    return false;
  }

  (*iv).assign(encrypt_data + sizeof(int32_t), encrypt_data + sizeof(int32_t) + iv_len);
  if (cipher_len <= 0 ||
      sizeof(int32_t) + static_cast<size_t>(iv_len) + sizeof(int32_t) + static_cast<size_t>(cipher_len) > encrypt_len) {
    MS_LOG(ERROR) << "assign len is invalid.";
    return false;
  }
  (*cipher_data)
    .assign(encrypt_data + sizeof(int32_t) + iv_len + sizeof(int32_t),
            encrypt_data + sizeof(int32_t) + iv_len + sizeof(int32_t) + cipher_len);
  return true;
}

bool ParseMode(const std::string &mode, std::string *alg_mode, std::string *work_mode) {
  if (alg_mode == nullptr || work_mode == nullptr) {
    MS_LOG(ERROR) << "There is a null pointer in the input parameter.";
    return false;
  }
  std::smatch results;
  std::regex re("([A-Z]{3})-([A-Z]{3})");
  if (!(std::regex_match(mode.c_str(), re) && std::regex_search(mode, results, re))) {
    MS_LOG(ERROR) << "Mode " << mode << " is invalid.";
    return false;
  }
  const size_t index_1 = 1;
  const size_t index_2 = 2;
  *alg_mode = results[index_1];
  *work_mode = results[index_2];
  return true;
}

EVP_CIPHER_CTX *GetEvpCipherCtx(const std::string &work_mode, const Byte *key, int32_t key_len, const Byte *iv,
                                int iv_len) {
  constexpr int32_t key_length_16 = 16;
  constexpr int32_t key_length_24 = 24;
  constexpr int32_t key_length_32 = 32;
  const EVP_CIPHER *(*funcPtr)() = nullptr;
  if (work_mode == "GCM") {
    switch (key_len) {
      case key_length_16:
        funcPtr = (const EVP_CIPHER *(*)())loader.GetFunc("EVP_aes_128_gcm");
        break;
      case key_length_24:
        funcPtr = (const EVP_CIPHER *(*)())loader.GetFunc("EVP_aes_192_gcm");
        break;
      case key_length_32:
        funcPtr = (const EVP_CIPHER *(*)())loader.GetFunc("EVP_aes_256_gcm");
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
  EVP_CIPHER_CTX *(*EVP_CIPHER_CTX_new)() = (EVP_CIPHER_CTX * (*)()) loader.GetFunc("EVP_CIPHER_CTX_new");
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "EVP_CIPHER_CTX_new failed";
    return nullptr;
  }

  int (*EVP_DecryptInit_ex)(EVP_CIPHER_CTX *, const EVP_CIPHER *, ENGINE *, const unsigned char *,
                            const unsigned char *) =
    (int (*)(EVP_CIPHER_CTX *, const EVP_CIPHER *, ENGINE *, const unsigned char *,
             const unsigned char *))loader.GetFunc("EVP_DecryptInit_ex");
  ret = EVP_DecryptInit_ex(ctx, funcPtr(), NULL, NULL, NULL);
  void (*EVP_CIPHER_CTX_free)(EVP_CIPHER_CTX *) = (void (*)(EVP_CIPHER_CTX *))loader.GetFunc("EVP_CIPHER_CTX_free");
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptInit_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return nullptr;
  }
  int (*EVP_CIPHER_CTX_ctrl)(EVP_CIPHER_CTX *, int, int, void *) =
    (int (*)(EVP_CIPHER_CTX * ctx, int type, int arg, void *ptr)) loader.GetFunc("EVP_CIPHER_CTX_ctrl");
  if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, NULL) != 1) {
    MS_LOG(ERROR) << "EVP_DecryptInit_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return nullptr;
  }
  ret = EVP_DecryptInit_ex(ctx, funcPtr(), NULL, key, iv);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptInit_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return nullptr;
  }
  return ctx;
}

bool BlockDecrypt(Byte *plain_data, int32_t *plain_len, const Byte *encrypt_data, size_t encrypt_len, const Byte *key,
                  int32_t key_len, const std::string &dec_mode, unsigned char *tag) {
  if (plain_data == nullptr || plain_len == nullptr || encrypt_data == nullptr || key == nullptr) {
    MS_LOG(ERROR) << "There is a null pointer in the input parameter.";
    return false;
  }
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
  auto ctx = GetEvpCipherCtx(work_mode, key, key_len, iv.data(), static_cast<int32_t>(iv.size()));
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }
  int (*EVP_DecryptUpdate)(EVP_CIPHER_CTX *, unsigned char *, int *, const unsigned char *, int) =
    (int (*)(EVP_CIPHER_CTX *, unsigned char *, int *, const unsigned char *, int))loader.GetFunc("EVP_DecryptUpdate");
  void (*EVP_CIPHER_CTX_free)(EVP_CIPHER_CTX *) = (void (*)(EVP_CIPHER_CTX *))loader.GetFunc("EVP_CIPHER_CTX_free");
  auto ret =
    EVP_DecryptUpdate(ctx, plain_data, plain_len, cipher_data.data(), static_cast<int32_t>(cipher_data.size()));
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptUpdate failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }

  int (*EVP_CIPHER_CTX_ctrl)(EVP_CIPHER_CTX *, int, int, void *) =
    (int (*)(EVP_CIPHER_CTX *, int, int, void *))loader.GetFunc("EVP_CIPHER_CTX_ctrl");
  if (!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, Byte16, tag)) {
    MS_LOG(ERROR) << "EVP_CIPHER_CTX_ctrl failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }

  int32_t mlen = 0;
  int (*EVP_DecryptFinal_ex)(EVP_CIPHER_CTX *, unsigned char *, int *) =
    (int (*)(EVP_CIPHER_CTX *, unsigned char *, int *))loader.GetFunc("EVP_DecryptFinal_ex");
  ret = EVP_DecryptFinal_ex(ctx, plain_data + *plain_len, &mlen);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptFinal_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }
  *plain_len += mlen;

  EVP_CIPHER_CTX_free(ctx);
  iv.assign(iv.size(), 0);
  return true;
}

std::unique_ptr<Byte[]> Decrypt(const std::string &lib_path, size_t *decrypt_len, const Byte *model_data,
                                const size_t data_size, const Byte *key, const size_t key_len,
                                const std::string &dec_mode) {
  if (model_data == nullptr) {
    MS_LOG(ERROR) << "model_data is nullptr.";
    return nullptr;
  }
  if (key == nullptr) {
    MS_LOG(ERROR) << "key is nullptr.";
    return nullptr;
  }
  if (decrypt_len == nullptr) {
    MS_LOG(ERROR) << "decrypt_len is nullptr.";
    return nullptr;
  }
  auto ret = loader.Open(lib_path);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "loader open failed.";
    return nullptr;
  }
  std::vector<char> block_buf;
  std::vector<char> int_buf(sizeof(int32_t));

  auto decrypt_data = std::make_unique<Byte[]>(data_size);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "decrypt_data is nullptr.";
    return nullptr;
  }
  int32_t decrypt_block_len;

  size_t offset = 0;
  *decrypt_len = 0;
  if (dec_mode != "AES-GCM") {
    MS_LOG(ERROR) << "dec_mode only support AES-GCM.";
    return nullptr;
  }
  if (key_len != Byte16) {
    MS_LOG(ERROR) << "key_len only support 16.";
    return nullptr;
  }
  while (offset < data_size) {
    if (offset + sizeof(int32_t) > data_size) {
      MS_LOG(ERROR) << "assign len is invalid.";
      return nullptr;
    }
    int_buf.assign(model_data + offset, model_data + offset + sizeof(int32_t));
    offset += int_buf.size();
    auto cipher_flag = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (static_cast<unsigned int>(cipher_flag) != MAGIC_NUM) {
      MS_LOG(ERROR) << "model_data is not encrypted and therefore cannot be decrypted.";
      return nullptr;
    }
    unsigned char tag[Byte16];
    if (offset + Byte16 > data_size) {
      MS_LOG(ERROR) << "buffer is invalid.";
      return nullptr;
    }
    memcpy(tag, model_data + offset, Byte16);
    offset += Byte16;
    if (offset + sizeof(int32_t) > data_size) {
      MS_LOG(ERROR) << "assign len is invalid.";
      return nullptr;
    }
    int_buf.assign(model_data + offset, model_data + offset + sizeof(int32_t));
    offset += int_buf.size();
    auto block_size = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (block_size <= 0) {
      MS_LOG(ERROR) << "The block_size read from the cipher data must be not negative, but got " << block_size;
      return nullptr;
    }
    if (offset + static_cast<size_t>(block_size) > data_size) {
      MS_LOG(ERROR) << "assign len is invalid.";
      return nullptr;
    }
    block_buf.assign(model_data + offset, model_data + offset + block_size);
    offset += block_buf.size();
    Byte *decrypt_block_buf = static_cast<Byte *>(malloc(MAX_BLOCK_SIZE * sizeof(Byte)));
    if (decrypt_block_buf == nullptr) {
      MS_LOG(ERROR) << "decrypt_block_buf is nullptr.";
      return nullptr;
    }
    if (!(BlockDecrypt(decrypt_block_buf, &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                       block_buf.size(), key, static_cast<int32_t>(key_len), dec_mode, tag))) {
      MS_LOG(ERROR) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
      free(decrypt_block_buf);
      return nullptr;
    }
    memcpy(decrypt_data.get() + *decrypt_len, decrypt_block_buf, static_cast<size_t>(decrypt_block_len));
    free(decrypt_block_buf);
    *decrypt_len += static_cast<size_t>(decrypt_block_len);
  }
  ret = loader.Close();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "loader close failed.";
    return nullptr;
  }
  return decrypt_data;
}
#endif
}  // namespace mindspore::lite

/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"

#if !defined(_MSC_VER) && !defined(_WIN32)
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
  return static_cast<unsigned int>(flag) == GCM_MAGIC_NUM || static_cast<unsigned int>(flag) == CBC_MAGIC_NUM;
}

bool IsCipherFile(const Byte *model_data) {
  MS_EXCEPTION_IF_NULL(model_data);
  std::vector<Byte> int_buf;
  int_buf.assign(model_data, model_data + sizeof(int32_t));
  auto flag = ByteToInt(int_buf.data(), int_buf.size());
  return static_cast<unsigned int>(flag) == GCM_MAGIC_NUM || static_cast<unsigned int>(flag) == CBC_MAGIC_NUM;
}
#if defined(_MSC_VER) || defined(_WIN32)
std::unique_ptr<Byte[]> Encrypt(size_t *, const Byte *, size_t, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *, const std::string &, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}

std::unique_ptr<Byte[]> Decrypt(size_t *, const Byte *, size_t, const Byte *, size_t, const std::string &) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return nullptr;
}
#else
bool ParseEncryptData(const Byte *encrypt_data, size_t encrypt_len, std::vector<Byte> *iv,
                      std::vector<Byte> *cipher_data) {
  // encrypt_data is organized in order to iv_len, iv, cipher_len, cipher_data
  std::vector<Byte> int_buf(sizeof(int32_t));
  int_buf.assign(encrypt_data, encrypt_data + sizeof(int32_t));
  auto iv_len = ByteToInt(int_buf.data(), int_buf.size());
  if (iv_len != AES_BLOCK_SIZE) {
    MS_LOG(ERROR) << "iv_len must be " << AES_BLOCK_SIZE << ", but got: " << iv_len;
    return false;
  }

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
  std::regex re("([A-Z]{3}|[A-Z]{2}\\d)-([A-Z]{3})");
  if (!(std::regex_match(mode.c_str(), re) && std::regex_search(mode, results, re))) {
    MS_LOG(ERROR) << "Mode " << mode << " is invalid.";
    return false;
  }
  *alg_mode = results[1];
  *work_mode = results[2];
  return true;
}

int InitCipherCtxAES(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *(*funcPtr)(), const std::string &work_mode, const Byte *key,
                     const Byte *iv, int iv_len, bool is_encrypt) {
  int32_t ret = 0;

  if (work_mode == "GCM") {
    if (is_encrypt) {
      ret = EVP_EncryptInit_ex(ctx, funcPtr(), nullptr, nullptr, nullptr);
      if (ret != 1) {
        MS_LOG(ERROR) << "EVP_EncryptInit_ex failed";
        EVP_CIPHER_CTX_free(ctx);
        return 1;
      }
      if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, nullptr) != 1) {
        MS_LOG(ERROR) << "EVP_EncryptInit_ex failed";
        EVP_CIPHER_CTX_free(ctx);
        return 1;
      }
      ret = EVP_EncryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
      if (ret != 1) {
        MS_LOG(ERROR) << "EVP_EncryptInit_ex failed";
        EVP_CIPHER_CTX_free(ctx);
        return 1;
      }
    } else {
      ret = EVP_DecryptInit_ex(ctx, funcPtr(), nullptr, nullptr, nullptr);
      if (ret != 1) {
        MS_LOG(ERROR) << "EVP_DecryptInit_ex failed";
        EVP_CIPHER_CTX_free(ctx);
        return 1;
      }
      if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, iv_len, nullptr) != 1) {
        MS_LOG(ERROR) << "EVP_DecryptInit_ex failed";
        EVP_CIPHER_CTX_free(ctx);
        return 1;
      }
      ret = EVP_DecryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
    }
  } else if (work_mode == "CBC") {
    if (is_encrypt) {
      ret = EVP_EncryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
    } else {
      ret = EVP_DecryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
    }
  }

  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptInit_ex/EVP_DecryptInit_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return 1;
  }
  if (work_mode == "CBC") {
    ret = EVP_CIPHER_CTX_set_padding(ctx, 1);
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_CIPHER_CTX_set_padding failed";
      EVP_CIPHER_CTX_free(ctx);
      return 1;
    }
  }
  return 0;
}

int InitCipherCtxSM4(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *(*funcPtr)(), const std::string &work_mode, const Byte *key,
                     const Byte *iv, bool is_encrypt) {
  int32_t ret = 0;

  if (work_mode == "CBC") {
    if (is_encrypt) {
      ret = EVP_EncryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
    } else {
      ret = EVP_DecryptInit_ex(ctx, funcPtr(), nullptr, key, iv);
    }
  }

  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptInit_ex/EVP_DecryptInit_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return 1;
  }
  if (work_mode == "CBC") {
    ret = EVP_CIPHER_CTX_set_padding(ctx, 1);
    if (ret != 1) {
      MS_LOG(ERROR) << "EVP_CIPHER_CTX_set_padding failed";
      EVP_CIPHER_CTX_free(ctx);
      return 1;
    }
  }
  return 0;
}

int InitCipherCtx(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *(*funcPtr)(), const std::string &alg_mode,
                  const std::string &work_mode, const Byte *key, int32_t, const Byte *iv, int iv_len, bool is_encrypt) {
  if (alg_mode == "AES") {
    return InitCipherCtxAES(ctx, funcPtr, work_mode, key, iv, iv_len, is_encrypt);
  } else if (alg_mode == "SM4") {
    return InitCipherCtxSM4(ctx, funcPtr, work_mode, key, iv, is_encrypt);
  }

  return 1;
}

EVP_CIPHER_CTX *GetEvpCipherCtx(const std::string &alg_mode, const std::string &work_mode, const Byte *key,
                                int32_t key_len, const Byte *iv, int iv_len, bool is_encrypt) {
  constexpr int32_t key_length_16 = 16;
  constexpr int32_t key_length_24 = 24;
  constexpr int32_t key_length_32 = 32;
  const EVP_CIPHER *(*funcPtr)() = nullptr;
  std::string alg_work_mode = alg_mode + "-" + work_mode;
  if (alg_work_mode == "AES-GCM") {
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
  } else if (alg_work_mode == "AES-CBC") {
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
  } else if (alg_work_mode == "SM4-CBC") {
    switch (key_len) {
      case key_length_16:
        funcPtr = EVP_sm4_cbc;
        break;
      default:
        MS_LOG(ERROR) << "The key length must be 16, but got key length is " << key_len << ".";
        return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "Crypto Algorithm " << alg_mode << " and "
                  << "Work mode " << work_mode << " is invalid.";
    return nullptr;
  }

  auto ctx = EVP_CIPHER_CTX_new();
  if (InitCipherCtx(ctx, funcPtr, alg_mode, work_mode, key, key_len, iv, iv_len, is_encrypt) != 0) {
    MS_LOG(ERROR) << "InitCipherCtx failed.";
    return nullptr;
  }
  return ctx;
}

bool BlockEncrypt(Byte *encrypt_data, size_t *encrypt_data_len, const std::vector<Byte> &plain_data, const Byte *key,
                  int32_t key_len, const std::string &enc_mode, unsigned char *tag) {
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

  auto ctx = GetEvpCipherCtx(alg_mode, work_mode, key, key_len, iv.data(), static_cast<int32_t>(iv.size()), true);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }

  std::vector<Byte> cipher_data_buf(plain_data.size() + AES_BLOCK_SIZE);
  auto ret_evp = EVP_EncryptUpdate(ctx, cipher_data_buf.data(), &cipher_len, plain_data.data(),
                                   static_cast<int32_t>(plain_data.size()));
  if (ret_evp != 1) {
    MS_LOG(ERROR) << "EVP_EncryptUpdate failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }
  int32_t flen = 0;
  ret_evp = EVP_EncryptFinal_ex(ctx, cipher_data_buf.data() + cipher_len, &flen);
  if (ret_evp != 1) {
    MS_LOG(ERROR) << "EVP_EncryptFinal_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }
  cipher_len += flen;

  if (enc_mode == "AES-GCM") {
    if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, Byte16, tag) != 1) {
      MS_LOG(ERROR) << "EVP_CIPHER_CTX_ctrl failed";
      EVP_CIPHER_CTX_free(ctx);
      return false;
    }
  }

  EVP_CIPHER_CTX_free(ctx);

  size_t offset = 0;
  std::vector<Byte> int_buf(sizeof(int32_t));
  *encrypt_data_len = sizeof(int32_t) + static_cast<size_t>(iv_len) + sizeof(int32_t) + static_cast<size_t>(cipher_len);
  IntToByte(&int_buf, static_cast<int32_t>(*encrypt_data_len));
  ret = memcpy_s(encrypt_data, encrypt_data_buf_len, int_buf.data(), int_buf.size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  IntToByte(&int_buf, iv_len);
  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, int_buf.data(), int_buf.size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, iv_cpy.data(), iv_cpy.size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += iv_cpy.size();

  IntToByte(&int_buf, cipher_len);
  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, int_buf.data(), int_buf.size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }
  offset += int_buf.size();

  ret = memcpy_s(encrypt_data + offset, encrypt_data_buf_len - offset, cipher_data_buf.data(),
                 static_cast<size_t>(cipher_len));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
  }

  *encrypt_data_len += sizeof(int32_t);
  return true;
}

bool BlockDecrypt(Byte *plain_data, int32_t *plain_len, const Byte *encrypt_data, size_t encrypt_len, const Byte *key,
                  int32_t key_len, const std::string &dec_mode, unsigned char *tag) {
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
  auto ctx = GetEvpCipherCtx(alg_mode, work_mode, key, key_len, iv.data(), SizeToInt(iv.size()), false);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }
  auto ret =
    EVP_DecryptUpdate(ctx, plain_data, plain_len, cipher_data.data(), static_cast<int32_t>(cipher_data.size()));
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptUpdate failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }

  if (dec_mode == "AES-GCM") {
    if (!EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, Byte16, tag)) {
      MS_LOG(ERROR) << "EVP_CIPHER_CTX_ctrl failed";
      EVP_CIPHER_CTX_free(ctx);
      return false;
    }
  }

  int32_t mlen = 0;
  ret = EVP_DecryptFinal_ex(ctx, plain_data + *plain_len, &mlen);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptFinal_ex failed";
    EVP_CIPHER_CTX_free(ctx);
    return false;
  }
  *plain_len += mlen;

  EVP_CIPHER_CTX_free(ctx);
  return true;
}

std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, const Byte *plain_data, size_t plain_len, const Byte *key,
                                size_t key_len, const std::string &enc_mode) {
  MS_EXCEPTION_IF_NULL(plain_data);
  MS_EXCEPTION_IF_NULL(key);
  if (enc_mode != "AES-GCM" && enc_mode != "AES-CBC" && enc_mode != "SM4-CBC") {
    MS_LOG(ERROR) << "Mode only support AES-GCM|AES-CBC|SM4-CBC.";
    return nullptr;
  }
  size_t block_enc_buf_len = MAX_BLOCK_SIZE + RESERVED_BYTE_PER_BLOCK;
  size_t encrypt_buf_len = plain_len + ((plain_len + MAX_BLOCK_SIZE) / MAX_BLOCK_SIZE) * RESERVED_BYTE_PER_BLOCK;
  std::vector<Byte> int_buf(sizeof(int32_t));
  std::vector<Byte> block_buf;
  std::vector<Byte> block_enc_buf(block_enc_buf_len);
  auto encrypt_data = std::make_unique<Byte[]>(encrypt_buf_len);

  size_t offset = 0;
  *encrypt_len = 0;
  while (offset < plain_len) {
    size_t cur_block_size = std::min(MAX_BLOCK_SIZE, plain_len - offset);
    block_buf.assign(plain_data + offset, plain_data + offset + cur_block_size);
    unsigned char tag[Byte16];
    if (!BlockEncrypt(block_enc_buf.data(), &block_enc_buf_len, block_buf, key, static_cast<int32_t>(key_len), enc_mode,
                      tag)) {
      MS_LOG(ERROR)
        << "Failed to encrypt data, please check if enc_key or enc_mode is valid or the file has been tempered with.";
      return nullptr;
    }
    if (enc_mode == "AES-GCM") {
      IntToByte(&int_buf, static_cast<int32_t>(GCM_MAGIC_NUM));
    } else if (enc_mode == "AES-CBC") {
      IntToByte(&int_buf, static_cast<int32_t>(CBC_MAGIC_NUM));
    } else if (enc_mode == "SM4-CBC") {
      IntToByte(&int_buf, static_cast<int32_t>(SM4_CBC_MAGIC_NUM));
    }
    size_t capacity = std::min(encrypt_buf_len - *encrypt_len, SECUREC_MEM_MAX_LEN);  // avoid dest size over 2gb
    errno_t ret = memcpy_s(encrypt_data.get() + *encrypt_len, capacity, int_buf.data(), sizeof(int32_t));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *encrypt_len += sizeof(int32_t);

    if (enc_mode == "AES-GCM") {
      capacity = std::min(encrypt_buf_len - *encrypt_len, SECUREC_MEM_MAX_LEN);  // avoid dest size over 2gb
      ret = memcpy_s(encrypt_data.get() + *encrypt_len, capacity, tag, Byte16);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
      }
      *encrypt_len += Byte16;
    }

    capacity = std::min(encrypt_buf_len - *encrypt_len, SECUREC_MEM_MAX_LEN);
    ret = memcpy_s(encrypt_data.get() + *encrypt_len, capacity, block_enc_buf.data(), block_enc_buf_len);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno " << ret;
    }
    *encrypt_len += block_enc_buf_len;
    offset += cur_block_size;
  }
  return encrypt_data;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                size_t key_len, const std::string &dec_mode) {
  MS_EXCEPTION_IF_NULL(key);
  if (dec_mode != "AES-GCM" && dec_mode != "AES-CBC" && dec_mode != "SM4-CBC") {
    MS_LOG(ERROR) << "Mode only support AES-GCM|AES-CBC|SM4-CBC.";
    return nullptr;
  }
  std::ifstream fid(encrypt_data_path, std::ios::in | std::ios::binary);
  if (!fid) {
    MS_LOG(ERROR) << "Open file '" << encrypt_data_path << "' failed, please check the correct of the file.";
    return nullptr;
  }
  fid.seekg(0, std::ios_base::end);
  size_t file_size = static_cast<size_t>(fid.tellg());
  fid.clear();
  fid.seekg(0);

  std::vector<char> block_buf(DECRYPT_BLOCK_BUF_SIZE);
  std::vector<char> int_buf(sizeof(int32_t));
  std::vector<Byte> decrypt_block_buf(DECRYPT_BLOCK_BUF_SIZE);
  auto decrypt_data = std::make_unique<Byte[]>(file_size);
  int32_t decrypt_block_len;

  *decrypt_len = 0;
  while (static_cast<size_t>(fid.tellg()) < file_size) {
    fid.read(int_buf.data(), static_cast<int32_t>(sizeof(int32_t)));
    auto cipher_flag = static_cast<unsigned int>(ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size()));
    if (dec_mode == "AES-GCM" && cipher_flag != GCM_MAGIC_NUM) {
      MS_LOG(ERROR) << "File \"" << encrypt_data_path << "\" is not an encrypted AES-GCM file and cannot be decrypted";
      return nullptr;
    } else if (dec_mode == "AES-CBC" && cipher_flag != CBC_MAGIC_NUM) {
      MS_LOG(ERROR) << "File \"" << encrypt_data_path << "\" is not an encrypted AES-CBC file and cannot be decrypted";
      return nullptr;
    } else if (dec_mode == "SM4-CBC" && cipher_flag != SM4_CBC_MAGIC_NUM) {
      MS_LOG(ERROR) << "File \"" << encrypt_data_path << "\" is not an encrypted SM4-CBC file and cannot be decrypted";
      return nullptr;
    }

    unsigned char tag[Byte16];
    if (dec_mode == "AES-GCM") {
      (void)fid.read(reinterpret_cast<char *>(tag), SizeToLong(Byte16));
    }
    fid.read(int_buf.data(), static_cast<int64_t>(sizeof(int32_t)));
    auto block_size = ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size());
    if (block_size < 0) {
      MS_LOG(ERROR) << "The block_size read from the cipher file must be not negative, but got " << block_size;
      return nullptr;
    }
    fid.read(block_buf.data(), static_cast<int64_t>(block_size));
    if (!(BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                       IntToSize(block_size), key, static_cast<int32_t>(key_len), dec_mode, tag))) {
      MS_LOG(ERROR) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
      return nullptr;
    }
    size_t capacity = std::min(file_size - *decrypt_len, SECUREC_MEM_MAX_LEN);
    errno_t ret = memcpy_s(decrypt_data.get() + *decrypt_len, capacity, decrypt_block_buf.data(),
                           static_cast<int32_t>(decrypt_block_len));
    if (ret != EOK) {
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
  std::unordered_set<std::string> dic = {"AES-GCM", "AES-CBC", "SM4-CBC"};
  if (dic.find(dec_mode) == dic.cend()) {
    MS_LOG(ERROR) << "Mode only support AES-GCM|AES-CBC|SM4-CBC.";
    return nullptr;
  }
  std::vector<char> block_buf;
  std::vector<char> int_buf(sizeof(int32_t));
  std::vector<Byte> decrypt_block_buf(DECRYPT_BLOCK_BUF_SIZE);
  auto decrypt_data = std::make_unique<Byte[]>(data_size);
  int32_t decrypt_block_len;

  size_t offset = 0;
  *decrypt_len = 0;
  while (offset < data_size) {
    if (offset + sizeof(int32_t) > data_size) {
      MS_LOG(ERROR) << "assign len is invalid.";
      return nullptr;
    }
    int_buf.assign(model_data + offset, model_data + offset + sizeof(int32_t));
    offset += int_buf.size();
    auto cipher_flag = static_cast<unsigned int>(ByteToInt(reinterpret_cast<Byte *>(int_buf.data()), int_buf.size()));
    if (dec_mode == "AES-GCM" && cipher_flag != GCM_MAGIC_NUM) {
      MS_LOG(ERROR) << "model_data is not encrypted AES-GCM and therefore cannot be decrypted.";
      return nullptr;
    } else if (dec_mode == "AES-CBC" && cipher_flag != CBC_MAGIC_NUM) {
      MS_LOG(ERROR) << "model_data is not encrypted AES-CBC and therefore cannot be decrypted.";
      return nullptr;
    } else if (dec_mode == "SM4-CBC" && cipher_flag != SM4_CBC_MAGIC_NUM) {
      MS_LOG(ERROR) << "model_data is not encrypted SM4-CBC and therefore cannot be decrypted.";
      return nullptr;
    }
    unsigned char tag[Byte16];
    if (dec_mode == "AES-GCM") {
      if (offset + Byte16 > data_size) {
        MS_LOG(ERROR) << "buffer is invalid.";
        return nullptr;
      }
      auto ret = memcpy_s(tag, Byte16, model_data + offset, Byte16);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "memcpy_s failed " << ret;
      }
      offset += Byte16;
    }
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
    if (offset + IntToSize(block_size) > data_size) {
      MS_LOG(ERROR) << "assign len is invalid.";
      return nullptr;
    }
    block_buf.assign(model_data + offset, model_data + offset + block_size);
    offset += block_buf.size();
    if (!(BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                       block_buf.size(), key, static_cast<int32_t>(key_len), dec_mode, tag))) {
      MS_LOG(ERROR) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
      return nullptr;
    }
    size_t capacity = std::min(data_size - *decrypt_len, SECUREC_MEM_MAX_LEN);
    errno_t ret = memcpy_s(decrypt_data.get() + *decrypt_len, capacity, decrypt_block_buf.data(),
                           static_cast<size_t>(decrypt_block_len));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s failed " << ret;
    }

    *decrypt_len += static_cast<size_t>(decrypt_block_len);
  }
  return decrypt_data;
}
#endif
}  // namespace mindspore

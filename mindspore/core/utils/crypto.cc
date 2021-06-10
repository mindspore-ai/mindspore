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

#include <fstream>
#include <regex>
#include <vector>
#include <algorithm>
#include "utils/crypto.h"
#include "utils/log_adapter.h"

#if not defined(_WIN32)
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

namespace mindspore {
void intToByte(Byte *byte, const int32_t &n) {
  memset(byte, 0, sizeof(Byte) * 4);
  byte[0] = (Byte)(0xFF & n);
  byte[1] = (Byte)((0xFF00 & n) >> 8);
  byte[2] = (Byte)((0xFF0000 & n) >> 16);
  byte[3] = (Byte)((0xFF000000 & n) >> 24);
}

int32_t ByteToint(const Byte *byteArray) {
  int32_t res = byteArray[0] & 0xFF;
  res |= ((byteArray[1] << 8) & 0xFF00);
  res |= ((byteArray[2] << 16) & 0xFF0000);
  res += ((byteArray[3] << 24) & 0xFF000000);
  return res;
}

bool IsCipherFile(const std::string &file_path) {
  std::ifstream fid(file_path, std::ios::in | std::ios::binary);
  if (!fid) {
    MS_LOG(ERROR) << "Failed to open file " << file_path;
    return false;
  }
  std::vector<char> int_buf(4);
  fid.read(int_buf.data(), sizeof(int32_t));
  fid.close();
  auto flag = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
  return flag == MAGIC_NUM;
}

bool IsCipherFile(const Byte *model_data) {
  std::vector<char> int_buf(4);
  memcpy(int_buf.data(), model_data, 4);
  auto flag = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
  return flag == MAGIC_NUM;
}
#if defined(_WIN32)
std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, Byte *plain_data, const size_t plain_len, const Byte *key,
                                const size_t key_len, const std::string &enc_mode) {
  MS_EXCEPTION(NotSupportError) << "Unsupported feature in Windows platform.";
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                const size_t key_len, const std::string &dec_mode) {
  MS_EXCEPTION(NotSupportError) << "Unsupported feature in Windows platform.";
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, const size_t data_size, const Byte *key,
                                const size_t key_len, const std::string &dec_mode) {
  MS_EXCEPTION(NotSupportError) << "Unsupported feature in Windows platform.";
}
#else

bool ParseEncryptData(const Byte *encrypt_data, const int32_t encrypt_len, std::vector<Byte> *iv,
                      std::vector<Byte> *cipher_data) {
  // encrypt_data is organized in order to iv_len, iv, cipher_len, cipher_data
  Byte buf[4];
  memcpy(buf, encrypt_data, 4);

  auto iv_len = ByteToint(buf);
  memcpy(buf, encrypt_data + iv_len + 4, 4);

  auto cipher_len = ByteToint(buf);
  if (iv_len <= 0 || cipher_len <= 0 || iv_len + cipher_len + 8 != encrypt_len) {
    MS_LOG(ERROR) << "Failed to parse encrypt data.";
    return false;
  }
  (*iv).resize(iv_len);
  memcpy((*iv).data(), encrypt_data + 4, iv_len);

  (*cipher_data).resize(cipher_len);
  memcpy((*cipher_data).data(), encrypt_data + iv_len + 8, cipher_len);

  return true;
}

bool ParseMode(std::string mode, std::string *alg_mode, std::string *work_mode) {
  std::smatch results;
  std::regex re("([A-Z]{3})-([A-Z]{3})");
  if (!std::regex_match(mode.c_str(), re)) {
    MS_LOG(ERROR) << "Mode " << mode << " is invalid.";
    return false;
  }
  std::regex_search(mode, results, re);
  *alg_mode = results[1];
  *work_mode = results[2];
  return true;
}

EVP_CIPHER_CTX *GetEVP_CIPHER_CTX(const std::string &work_mode, const Byte *key, const int32_t key_len, const Byte *iv,
                                  int flag) {
  int ret = 0;
  EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
  if (work_mode != "GCM" && work_mode != "CBC") {
    MS_LOG(ERROR) << "Work mode " << work_mode << " is invalid.";
    return nullptr;
  }

  const EVP_CIPHER *(*funcPtr)() = nullptr;
  if (work_mode == "GCM") {
    switch (key_len) {
      case 16:
        funcPtr = EVP_aes_128_gcm;
        break;
      case 24:
        funcPtr = EVP_aes_192_gcm;
        break;
      case 32:
        funcPtr = EVP_aes_256_gcm;
        break;
      default:
        MS_EXCEPTION(ValueError) << "The key length must be 16, 24 or 32, but got key length is " << key_len << ".";
    }
  } else if (work_mode == "CBC") {
    switch (key_len) {
      case 16:
        funcPtr = EVP_aes_128_cbc;
        break;
      case 24:
        funcPtr = EVP_aes_192_cbc;
        break;
      case 32:
        funcPtr = EVP_aes_256_cbc;
        break;
      default:
        MS_EXCEPTION(ValueError) << "The key length must be 16, 24 or 32, but got key length is " << key_len << ".";
    }
  }

  if (flag == 0) {
    ret = EVP_EncryptInit_ex(ctx, funcPtr(), NULL, key, iv);
  } else if (flag == 1) {
    ret = EVP_DecryptInit_ex(ctx, funcPtr(), NULL, key, iv);
  }

  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptInit_ex failed";
    return nullptr;
  }
  if (work_mode == "CBC") EVP_CIPHER_CTX_set_padding(ctx, 1);
  return ctx;
}

bool _BlockEncrypt(Byte *encrypt_data, size_t *encrypt_data_len, Byte *plain_data, const size_t plain_len,
                   const Byte *key, const int32_t key_len, const std::string &enc_mode) {
  int32_t cipher_len = 0;

  int32_t iv_len = AES_BLOCK_SIZE;
  Byte *iv = new Byte[iv_len];
  RAND_bytes(iv, sizeof(Byte) * iv_len);

  Byte *iv_cpy = new Byte[16];
  memcpy(iv_cpy, iv, 16);

  int32_t flen = 0;
  std::string alg_mode;
  std::string work_mode;
  if (!ParseMode(enc_mode, &alg_mode, &work_mode)) {
    return false;
  }

  auto ctx = GetEVP_CIPHER_CTX(work_mode, key, key_len, iv, 0);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }

  std::vector<Byte> cipher_data(plain_len + 16);
  int ret = EVP_EncryptUpdate(ctx, cipher_data.data(), &cipher_len, plain_data, plain_len);
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_EncryptUpdate failed";
    return false;
  }
  if (work_mode == "CBC") {
    EVP_EncryptFinal_ex(ctx, cipher_data.data() + cipher_len, &flen);
    cipher_len += flen;
  }
  EVP_CIPHER_CTX_free(ctx);

  size_t cur = 0;
  *encrypt_data_len = sizeof(int32_t) * 2 + iv_len + cipher_len;

  std::vector<Byte> byte_buf(4);
  intToByte(byte_buf.data(), *encrypt_data_len);
  memcpy(encrypt_data + cur, byte_buf.data(), 4);
  cur += 4;

  intToByte(byte_buf.data(), iv_len);
  memcpy(encrypt_data + cur, byte_buf.data(), 4);
  cur += 4;

  memcpy(encrypt_data + cur, iv_cpy, iv_len);
  cur += iv_len;

  intToByte(byte_buf.data(), cipher_len);
  memcpy(encrypt_data + cur, byte_buf.data(), 4);
  cur += 4;

  memcpy(encrypt_data + cur, cipher_data.data(), cipher_len);
  *encrypt_data_len += 4;

  return true;
}

bool _BlockDecrypt(Byte *plain_data, int32_t *plain_len, Byte *encrypt_data, const size_t encrypt_len, const Byte *key,
                   const int32_t key_len, const std::string &dec_mode) {
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

  int ret = 0;
  int mlen = 0;
  auto ctx = GetEVP_CIPHER_CTX(work_mode, key, key_len, iv.data(), 1);
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Failed to get EVP_CIPHER_CTX.";
    return false;
  }
  ret = EVP_DecryptUpdate(ctx, plain_data, plain_len, cipher_data.data(), cipher_data.size());
  if (ret != 1) {
    MS_LOG(ERROR) << "EVP_DecryptUpdate failed";
    return false;
  }
  if (work_mode == "CBC") {
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

std::unique_ptr<Byte[]> Encrypt(size_t *encrypt_len, Byte *plain_data, const size_t plain_len, const Byte *key,
                                const size_t key_len, const std::string &enc_mode) {
  size_t cur_pos = 0;
  size_t block_enc_len = 0;
  size_t encrypt_buf_len = plain_len + (plain_len / MAX_BLOCK_SIZE + 1) * 100;
  size_t block_enc_buf_len = MAX_BLOCK_SIZE + 100;
  std::vector<Byte> int_buf(4);
  std::vector<Byte> block_buf(MAX_BLOCK_SIZE);
  std::vector<Byte> block_enc_buf(block_enc_buf_len);
  auto encrypt_data = std::make_unique<Byte[]>(encrypt_buf_len);

  *encrypt_len = 0;
  while (cur_pos < plain_len) {
    size_t cur_block_size = std::min(MAX_BLOCK_SIZE, plain_len - cur_pos);
    memcpy(block_buf.data(), plain_data + cur_pos, cur_block_size);
    if (!_BlockEncrypt(block_enc_buf.data(), &block_enc_len, block_buf.data(), cur_block_size, key, key_len,
                       enc_mode)) {
      MS_EXCEPTION(ValueError) << "Failed to encrypt data, please check if enc_key or enc_mode is valid.";
    }
    intToByte(int_buf.data(), MAGIC_NUM);
    memcpy(encrypt_data.get() + *encrypt_len, int_buf.data(), sizeof(int32_t));
    *encrypt_len += sizeof(int32_t);
    memcpy(encrypt_data.get() + *encrypt_len, block_enc_buf.data(), block_enc_len);
    *encrypt_len += block_enc_len;
    cur_pos += cur_block_size;
  }
  return encrypt_data;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const std::string &encrypt_data_path, const Byte *key,
                                const size_t key_len, const std::string &dec_mode) {
  std::ifstream fid(encrypt_data_path, std::ios::in | std::ios::binary);
  if (!fid) {
    MS_EXCEPTION(ValueError) << "Open file '" << encrypt_data_path << "' failed, please check the correct of the file.";
  }
  fid.seekg(0, std::ios_base::end);
  size_t file_size = fid.tellg();
  fid.clear();
  fid.seekg(0);

  std::vector<char> block_buf(MAX_BLOCK_SIZE * 2);
  std::vector<char> int_buf(4);
  std::vector<Byte> decrypt_block_buf(MAX_BLOCK_SIZE * 2);
  auto decrypt_data = std::make_unique<Byte[]>(file_size);
  int32_t decrypt_block_len;

  *decrypt_len = 0;
  while (static_cast<size_t>(fid.tellg()) < file_size) {
    fid.read(int_buf.data(), sizeof(int32_t));
    int cipher_flag = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
    if (cipher_flag != MAGIC_NUM) {
      MS_EXCEPTION(ValueError) << "File \"" << encrypt_data_path
                               << "\" is not an encrypted file and cannot be decrypted";
    }
    fid.read(int_buf.data(), sizeof(int32_t));
    int32_t block_size = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
    fid.read(block_buf.data(), sizeof(char) * block_size);
    if (!(_BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                        block_size, key, key_len, dec_mode))) {
      MS_EXCEPTION(ValueError) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
    }
    memcpy(decrypt_data.get() + *decrypt_len, decrypt_block_buf.data(), decrypt_block_len);
    *decrypt_len += decrypt_block_len;
  }
  fid.close();
  return decrypt_data;
}

std::unique_ptr<Byte[]> Decrypt(size_t *decrypt_len, const Byte *model_data, const size_t data_size, const Byte *key,
                                const size_t key_len, const std::string &dec_mode) {
  std::vector<char> block_buf(MAX_BLOCK_SIZE * 2);
  std::vector<char> int_buf(4);
  std::vector<Byte> decrypt_block_buf(MAX_BLOCK_SIZE * 2);
  auto decrypt_data = std::make_unique<Byte[]>(data_size);
  int32_t decrypt_block_len;

  size_t cur_pos = 0;
  *decrypt_len = 0;
  while (cur_pos < data_size) {
    memcpy(int_buf.data(), model_data + cur_pos, 4);
    cur_pos += 4;
    int cipher_flag = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
    if (cipher_flag != MAGIC_NUM) {
      MS_EXCEPTION(ValueError) << "model_data is not encrypted and therefore cannot be decrypted.";
    }
    memcpy(int_buf.data(), model_data + cur_pos, 4);
    cur_pos += 4;

    int32_t block_size = ByteToint(reinterpret_cast<Byte *>(int_buf.data()));
    memcpy(block_buf.data(), model_data + cur_pos, block_size);
    cur_pos += block_size;
    if (!(_BlockDecrypt(decrypt_block_buf.data(), &decrypt_block_len, reinterpret_cast<Byte *>(block_buf.data()),
                        block_size, key, key_len, dec_mode))) {
      MS_EXCEPTION(ValueError) << "Failed to decrypt data, please check if dec_key or dec_mode is valid";
    }
    memcpy(decrypt_data.get() + *decrypt_len, decrypt_block_buf.data(), decrypt_block_len);
    *decrypt_len += decrypt_block_len;
  }
  return decrypt_data;
}
#endif
}  // namespace mindspore

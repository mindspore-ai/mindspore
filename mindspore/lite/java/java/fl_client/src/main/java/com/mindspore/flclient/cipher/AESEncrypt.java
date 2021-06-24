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

package com.mindspore.flclient.cipher;

import com.mindspore.flclient.Common;

import javax.crypto.Cipher;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.UnsupportedEncodingException;
import java.util.logging.Logger;

public class AESEncrypt {
    private static final Logger LOGGER = Logger.getLogger(AESEncrypt.class.toString());
    /**
     * 128, 192 or 256
     */
    private static final int KEY_SIZE = 256;
    private static final int I_VEC_LEN = 16;

    /**
     * encrypt/decrypt algorithm name
     */
    private static final String ALGORITHM = "AES";

    /**
     * algorithm/Mode/padding mode
     */
    private static final String CIPHER_MODE_CTR = "AES/CTR/NoPadding";
    private static final String CIPHER_MODE_CBC = "AES/CBC/PKCS5PADDING";

    private String CIPHER_MODE;

    private static final int RANDOM_LEN = KEY_SIZE / 8;

    private String iVecS = "1111111111111111";
    private byte[] iVec = iVecS.getBytes("utf-8");

    public AESEncrypt(byte[] key, byte[] iVecIn, String mode) throws UnsupportedEncodingException {
        if (key == null) {
            LOGGER.severe(Common.addTag("Key is null"));
            return;
        }
        if (key.length != KEY_SIZE / 8) {
            LOGGER.severe(Common.addTag("the length of key is not correct"));
            return;
        }

        if (mode.contains("CBC")) {
            CIPHER_MODE = CIPHER_MODE_CBC;
        } else if (mode.contains("CTR")) {
            CIPHER_MODE = CIPHER_MODE_CTR;
        } else {
            return;
        }
        if (iVecIn == null || iVecIn.length != I_VEC_LEN) {
            return;
        }
        iVec = iVecIn;
    }

    public byte[] encrypt(byte[] key, byte[] data) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(key, ALGORITHM);
        Cipher cipher = Cipher.getInstance(CIPHER_MODE);
        IvParameterSpec iv = new IvParameterSpec(iVec);
        cipher.init(Cipher.ENCRYPT_MODE, skeySpec, iv);
        byte[] encrypted = cipher.doFinal(data);
        String encryptResultStr = BaseUtil.byte2HexString(encrypted);
        return encrypted;
    }

    public byte[] encryptCTR(byte[] key, byte[] data) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(key, ALGORITHM);
        Cipher cipher = Cipher.getInstance(CIPHER_MODE);
        IvParameterSpec iv = new IvParameterSpec(iVec);
        cipher.init(Cipher.ENCRYPT_MODE, skeySpec, iv);
        byte[] encrypted = cipher.doFinal(data);
        return encrypted;
    }

    public byte[] decrypt(byte[] key, byte[] encryptData) throws Exception {
        SecretKeySpec skeySpec = new SecretKeySpec(key, ALGORITHM);
        Cipher cipher = Cipher.getInstance(CIPHER_MODE);
        IvParameterSpec iv = new IvParameterSpec(iVec);
        cipher.init(Cipher.DECRYPT_MODE, skeySpec, iv);
        byte[] origin = cipher.doFinal(encryptData);
        return origin;
    }

}

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

import static com.mindspore.flclient.LocalFLParameter.I_VEC_LEN;
import static com.mindspore.flclient.LocalFLParameter.KEY_LEN;

import com.mindspore.flclient.Common;

import java.io.UnsupportedEncodingException;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Arrays;
import java.util.logging.Logger;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * Define encryption and decryption methods.
 *
 * @since 2021-06-30
 */
public class AESEncrypt {
    private static final Logger LOGGER = Logger.getLogger(AESEncrypt.class.toString());

    /**
     * encrypt/decrypt algorithm name
     */
    private static final String ALGORITHM = "AES";

    /**
     * algorithm/Mode/padding mode
     */
    private static final String CIPHER_MODE_CTR = "AES/CTR/NoPadding";
    private static final String CIPHER_MODE_CBC = "AES/CBC/PKCS5PADDING";

    private String cipherMod;

    /**
     * Defining a Constructor of the class AESEncrypt.
     *
     * @param key  the Key.
     * @param mode the encryption Mode.
     */
    public AESEncrypt(byte[] key, String mode) {
        if (key == null) {
            LOGGER.severe(Common.addTag("Key is null"));
            return;
        }
        if (key.length != KEY_LEN) {
            LOGGER.severe(Common.addTag("the length of key is not correct"));
            return;
        }

        if (mode.contains("CBC")) {
            cipherMod = CIPHER_MODE_CBC;
        } else if (mode.contains("CTR")) {
            cipherMod = CIPHER_MODE_CTR;
        } else {
            return;
        }
    }

    /**
     * Defining the CBC encryption Mode.
     *
     * @param key  the Key.
     * @param data the data to be encrypted.
     * @return the data to be encrypted.
     */
    public byte[] encrypt(byte[] key, byte[] data) {
        if (key == null) {
            LOGGER.severe(Common.addTag("Key is null"));
            return new byte[0];
        }
        if (data == null) {
            LOGGER.severe(Common.addTag("data is null"));
            return new byte[0];
        }
        try {
            byte[] iVec = new byte[I_VEC_LEN];
            SecureRandom secureRandom = Common.getSecureRandom();
            secureRandom.nextBytes(iVec);

            SecretKeySpec skeySpec = new SecretKeySpec(key, ALGORITHM);
            Cipher cipher = Cipher.getInstance(cipherMod);

            IvParameterSpec iv = new IvParameterSpec(iVec);
            cipher.init(Cipher.ENCRYPT_MODE, skeySpec, iv);
            byte[] encrypted = cipher.doFinal(data);
            byte[] encryptedAddIv = new byte[encrypted.length + iVec.length];
            System.arraycopy(iVec, 0, encryptedAddIv, 0, iVec.length);
            System.arraycopy(encrypted, 0, encryptedAddIv, iVec.length, encrypted.length);
            return encryptedAddIv;
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidKeyException |
                InvalidAlgorithmParameterException | IllegalBlockSizeException | BadPaddingException ex) {
            LOGGER.severe(Common.addTag("catch NoSuchAlgorithmException or " +
                    "NoSuchPaddingException or InvalidKeyException or InvalidAlgorithmParameterException or " +
                    "IllegalBlockSizeException or BadPaddingException: " + ex.getMessage()));
            return new byte[0];
        }
    }

    /**
     * Defining the CTR encryption Mode.
     *
     * @param key  the Key.
     * @param data the data to be encrypted.
     * @param iVec the IV value.
     * @return the data to be encrypted.
     */
    public byte[] encryptCTR(byte[] key, byte[] data, byte[] iVec) {
        if (key == null) {
            LOGGER.severe(Common.addTag("Key is null"));
            return new byte[0];
        }
        if (data == null) {
            LOGGER.severe(Common.addTag("data is null"));
            return new byte[0];
        }
        if (iVec == null || iVec.length != I_VEC_LEN) {
            LOGGER.severe(Common.addTag("iVec is null or the length of iVec is not valid, it should be " + "I_VEC_LEN"
            ));
            return new byte[0];
        }
        try {
            SecretKeySpec skeySpec = new SecretKeySpec(key, ALGORITHM);
            Cipher cipher = Cipher.getInstance(cipherMod);
            IvParameterSpec iv = new IvParameterSpec(iVec);
            cipher.init(Cipher.ENCRYPT_MODE, skeySpec, iv);
            return cipher.doFinal(data);
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidKeyException |
                InvalidAlgorithmParameterException | IllegalBlockSizeException | BadPaddingException ex) {
            LOGGER.severe(Common.addTag("[encryptCTR] catch NoSuchAlgorithmException or " +
                    "NoSuchPaddingException or InvalidKeyException or InvalidAlgorithmParameterException or " +
                    "IllegalBlockSizeException or BadPaddingException: " + ex.getMessage()));
            return new byte[0];
        }
    }

    /**
     * Defining the decrypt method.
     *
     * @param key              the Key.
     * @param encryptDataAddIv the data to be decrypted.
     * @return the data to be decrypted.
     */
    public byte[] decrypt(byte[] key, byte[] encryptDataAddIv) {
        if (key == null) {
            LOGGER.severe(Common.addTag("Key is null"));
            return new byte[0];
        }
        if (encryptDataAddIv == null) {
            LOGGER.severe(Common.addTag("encryptDataAddIv is null"));
            return new byte[0];
        }
        if (encryptDataAddIv.length <= I_VEC_LEN) {
            LOGGER.severe(Common.addTag("the length of encryptDataAddIv is not valid: " + encryptDataAddIv.length +
                    ", it should be > " + I_VEC_LEN));
            return new byte[0];
        }
        try {
            byte[] iVec = Arrays.copyOfRange(encryptDataAddIv, 0, I_VEC_LEN);
            byte[] encryptData = Arrays.copyOfRange(encryptDataAddIv, I_VEC_LEN, encryptDataAddIv.length);
            SecretKeySpec sKeySpec = new SecretKeySpec(key, ALGORITHM);
            Cipher cipher = Cipher.getInstance(cipherMod);
            IvParameterSpec iv = new IvParameterSpec(iVec);
            cipher.init(Cipher.DECRYPT_MODE, sKeySpec, iv);
            return cipher.doFinal(encryptData);
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidKeyException |
                InvalidAlgorithmParameterException | IllegalBlockSizeException | BadPaddingException ex) {
            LOGGER.severe(Common.addTag("catch NoSuchAlgorithmException or " +
                    "NoSuchPaddingException or InvalidKeyException or InvalidAlgorithmParameterException or " +
                    "IllegalBlockSizeException or BadPaddingException: " + ex.getMessage()));
            return new byte[0];
        }
    }
}

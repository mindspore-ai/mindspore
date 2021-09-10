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

import static com.mindspore.flclient.LocalFLParameter.KEY_LEN;

import com.mindspore.flclient.Common;

import org.bouncycastle.crypto.digests.SHA256Digest;
import org.bouncycastle.crypto.generators.PKCS5S2ParametersGenerator;
import org.bouncycastle.crypto.params.KeyParameter;
import org.bouncycastle.math.ec.rfc7748.X25519;

import java.security.SecureRandom;
import java.util.logging.Logger;

/**
 * Generate public-private key pairs and DH Keys.
 *
 * @since 2021-06-30
 */
public class KEYAgreement {
    private static final Logger LOGGER = Logger.getLogger(KEYAgreement.class.toString());
    private static final int PBKDF2_ITERATIONS = 10000;
    private static final int HASH_BIT_SIZE = 256;

    private SecureRandom random = Common.getSecureRandom();

    /**
     * Generate private Key.
     *
     * @return the private Key.
     */
    public byte[] generatePrivateKey() {
        byte[] privateKey = new byte[KEY_LEN];
        X25519.generatePrivateKey(random, privateKey);
        return privateKey;
    }

    /**
     * Use private Key to generate public Key.
     *
     * @param privateKey the private Key.
     * @return the public Key.
     */
    public byte[] generatePublicKey(byte[] privateKey) {
        if (privateKey == null || privateKey.length == 0) {
            LOGGER.severe(Common.addTag("privateKey is null"));
            return new byte[0];
        }
        byte[] publicKey = new byte[KEY_LEN];
        X25519.generatePublicKey(privateKey, 0, publicKey, 0);
        return publicKey;
    }

    /**
     * Use private Key and public Key to generate DH Key.
     *
     * @param privateKey the private Key.
     * @param publicKey  the public Key.
     * @return the DH Key.
     */
    public byte[] keyAgreement(byte[] privateKey, byte[] publicKey) {
        if (privateKey == null || privateKey.length == 0) {
            LOGGER.severe(Common.addTag("privateKey is null"));
            return new byte[0];
        }
        if (publicKey == null || publicKey.length == 0) {
            LOGGER.severe(Common.addTag("publicKey is null"));
            return new byte[0];
        }
        byte[] secret = new byte[KEY_LEN];
        X25519.calculateAgreement(privateKey, 0, publicKey, 0, secret, 0);
        return secret;
    }

    /**
     * Encrypt DH Key.
     *
     * @param password the DH Key.
     * @param salt     the salt value.
     * @return encrypted DH Key.
     */
    public byte[] getEncryptedPassword(byte[] password, byte[] salt) {
        if (password == null || password.length == 0) {
            LOGGER.severe(Common.addTag("password is null"));
            return new byte[0];
        }
        PKCS5S2ParametersGenerator gen = new PKCS5S2ParametersGenerator(new SHA256Digest());
        gen.init(password, salt, PBKDF2_ITERATIONS);
        return ((KeyParameter) gen.generateDerivedParameters(HASH_BIT_SIZE)).getKey();
    }
}

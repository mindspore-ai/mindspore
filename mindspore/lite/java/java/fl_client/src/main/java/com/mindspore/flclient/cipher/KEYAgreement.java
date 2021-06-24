
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

import org.bouncycastle.crypto.digests.SHA256Digest;
import org.bouncycastle.crypto.generators.PKCS5S2ParametersGenerator;
import org.bouncycastle.crypto.params.KeyParameter;
import org.bouncycastle.math.ec.rfc7748.X25519;

import java.security.SecureRandom;
import java.util.logging.Logger;

public class KEYAgreement {
    private static final Logger LOGGER = Logger.getLogger(KEYAgreement.class.toString());
    private static final int PBKDF2_ITERATIONS = 10000;
    private static final int SALT_SIZE = 32;
    private static final int HASH_BIT_SIZE = 256;
    private static final int KEY_LEN = X25519.SCALAR_SIZE;

    private SecureRandom random = new SecureRandom();

    public byte[] generatePrivateKey() {
        byte[] privateKey = new byte[KEY_LEN];
        X25519.generatePrivateKey(random, privateKey);
        return privateKey;
    }

    public byte[] generatePublicKey(byte[] privatekey) {
        byte[] publicKey = new byte[KEY_LEN];
        X25519.generatePublicKey(privatekey, 0, publicKey, 0);
        return publicKey;
    }

    public byte[] keyAgreement(byte[] privatekey, byte[] publicKey) {
        byte[] secret = new byte[KEY_LEN];
        X25519.calculateAgreement(privatekey, 0, publicKey, 0, secret, 0);
        return secret;
    }

    public byte[] getEncryptedPassword(byte[] password, byte[] salt) {

        byte[] saltB = new byte[SALT_SIZE];
        PKCS5S2ParametersGenerator gen = new PKCS5S2ParametersGenerator(new SHA256Digest());
        gen.init(password, saltB, PBKDF2_ITERATIONS);
        byte[] dk = ((KeyParameter) gen.generateDerivedParameters(HASH_BIT_SIZE)).getKey();
        return dk;
    }
}

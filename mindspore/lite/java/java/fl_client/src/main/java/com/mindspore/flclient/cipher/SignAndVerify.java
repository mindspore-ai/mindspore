/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.mindspore.flclient.cipher;

import com.mindspore.flclient.Common;

import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.PublicKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.UnrecoverableKeyException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.logging.Logger;

/**
 * class used for sign data and verify data
 *
 * @since 2021-8-27
 */
public class SignAndVerify {
    private static final Logger LOGGER = Logger.getLogger(SignAndVerify.class.toString());

    /**
     * sign data
     *
     * @param clientID ID of this client
     * @param data data need to be signed
     * @return signed data
     */
    public static byte[] signData(String clientID, byte[] data) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter clientID is null or empty, please check!"));
            return null;
        }
        if (data == null || data.length == 0) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter data is null or empty, please check!"));
            return null;
        }
        byte[] signData = null;
        try {
            KeyStore ks = KeyStore.getInstance(CipherConsts.KEYSTORE_TYPE);
            ks.load(null);
            Key privateKey = ks.getKey(clientID, null);
            if (privateKey == null) {
                LOGGER.info("private key is null");
                return null;
            }
            Signature signature = Signature.getInstance(CipherConsts.SIGN_ALGORITHM, CipherConsts.PROVIDER_NAME);
            signature.initSign((PrivateKey) privateKey);
            signature.update(data);
            signData = signature.sign();
        } catch (KeyStoreException | CertificateException | NoSuchAlgorithmException | IOException
                | UnrecoverableKeyException | NoSuchProviderException | InvalidKeyException
                | SignatureException e) {
            LOGGER.severe(Common.addTag("[SignAndVerify] catch Exception: " + e.getMessage()));
        }
        return signData;
    }

    /**
     * verify signature with certifications
     *
     * @param clientID ID of this client
     * @param x509Certificates certificates
     * @param data original data
     * @param signed signed data
     * @return verify result
     */
    public static boolean verifySignatureByCert(String clientID, X509Certificate[] x509Certificates, byte[] data,
                                                byte[] signed) {
        if (clientID == null || clientID.isEmpty()) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter clientID is null or empty, please check!"));
            return false;
        }
        if (x509Certificates == null || x509Certificates.length < 1) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter x509Certificates is null or the length is not " +
                    "valid: < 1, please check!"));
            return false;
        }
        if (data == null || data.length == 0) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter data is null or empty, please check!"));
            return false;
        }
        if (signed == null || signed.length == 0) {
            LOGGER.severe(Common.addTag("[SignAndVerify] the parameter signed is null or empty, please check!"));
            return false;
        }
        if (!CertVerify.verifyCertificateChain(clientID, x509Certificates)) {
            LOGGER.info(Common.addTag("Verify chain failed!"));
            return false;
        }
        LOGGER.info(Common.addTag("Verify chain success!"));

        boolean isValid;
        try {
            if (x509Certificates[0].getPublicKey() == null) {
                LOGGER.severe(Common.addTag("[SignAndVerify] get public key failed!"));
                return false;
            }
            PublicKey publicKey = x509Certificates[0].getPublicKey();  // get public key
            Signature signature = Signature.getInstance(CipherConsts.SIGN_ALGORITHM);
            signature.initVerify(publicKey);  // set public key
            signature.update(data);  // set data
            isValid = signature.verify(signed); // verify the consistence between signature and data
        } catch (NoSuchAlgorithmException | SignatureException | InvalidKeyException e) {
            LOGGER.severe(Common.addTag("[SignAndVerify] catch Exception: " + e.getMessage()));
            return false;
        }
        return isValid;
    }

    /**
     * get hash result of bytes
     *
     * @param bytes inputs
     * @return hash value of bytes
     */
    public static byte[] getSHA256(byte[] bytes) {
        MessageDigest messageDigest;
        byte[] hash = new byte[0];
        try {
            messageDigest = MessageDigest.getInstance("SHA-256");
            hash = messageDigest.digest(bytes);
        } catch (NoSuchAlgorithmException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch NoSuchAlgorithmException: " + e.getMessage()));
        }
        return hash;
    }
}

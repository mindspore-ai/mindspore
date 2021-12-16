/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 */

package com.mindspore.flclient.pki;

import com.mindspore.flclient.Common;
import com.mindspore.flclient.LocalFLParameter;

import org.bouncycastle.util.io.pem.PemObject;
import org.bouncycastle.util.io.pem.PemWriter;

import java.io.IOException;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.Key;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.NoSuchProviderException;
import java.security.PrivateKey;
import java.security.Signature;
import java.security.SignatureException;
import java.security.UnrecoverableEntryException;
import java.security.UnrecoverableKeyException;
import java.security.cert.Certificate;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.Locale;
import java.util.logging.Logger;

/**
 * Pki Util
 *
 * @since 2021-08-25
 */
public class PkiUtil {
    private static final Logger LOGGER = Logger.getLogger(PkiUtil.class.toString());

    /**
     * generate PkiBean
     *
     * @param clientID String
     * @param time     long
     * @return PkiBean
     */
    public static PkiBean genPkiBean(String clientID, long time) {
        String sourceData = LocalFLParameter.getInstance().getFlID() + " " + time;
        byte[] signDataBytes = PkiUtil.signData(clientID,
                sourceData.getBytes(StandardCharsets.UTF_8));

        Certificate[] certificates = PkiUtil.getCertificateChain(clientID);

        return new PkiBean(signDataBytes, certificates);
    }

    /**
     * get str of SHA256
     *
     * @param str String
     * @return hash
     */
    public static byte[] getSHA256Str(String str) {
        MessageDigest messageDigest;
        byte[] hash = new byte[0];
        try {
            messageDigest = MessageDigest.getInstance("SHA-256");
            hash = messageDigest.digest(str.getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch NoSuchAlgorithmException: " + e.getMessage()));
        }
        return hash;
    }

    /**
     * get Pem format
     *
     * @param certificate Certificate
     * @return String result
     * @throws IOException e
     */
    public static String getPemFormat(Certificate certificate) throws IOException {
        StringWriter writer = new StringWriter();
        PemWriter pemWriter = new PemWriter(writer);
        try {
            pemWriter.writeObject(new PemObject("CERTIFICATE", certificate.getEncoded()));
        } catch (IOException | CertificateEncodingException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch IOException or CertificateEncodingException in getPermFormat: "
                    + e.getMessage()));
        } finally {
            pemWriter.flush();
            pemWriter.close();
        }

        return writer.toString();
    }

    private static byte[] signData(String clientID, byte[] data) {
        byte[] signData = null;
        try {
            KeyStore ks = KeyStore.getInstance(PkiConsts.KEYSTORE_TYPE);
            ks.load(null);
            Key privateKey = ks.getKey(clientID, null);
            if (privateKey == null) {
                return new byte[0];
            }
            Signature signature = Signature.getInstance(PkiConsts.ALGORITHM, PkiConsts.PROVIDER_NAME);
            if (privateKey instanceof PrivateKey) {
                signature.initSign((PrivateKey) privateKey);
            }
            signature.update(data);
            signData = signature.sign();
        } catch (KeyStoreException | CertificateException | NoSuchAlgorithmException | IOException
                | UnrecoverableKeyException | NoSuchProviderException | InvalidKeyException
                | SignatureException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch Exception: " + e.getMessage()));
        }
        return signData;
    }

    /**
     * get certificate chain
     *
     * @param clientID String
     * @return Certificate[]
     */
    public static Certificate[] getCertificateChain(String clientID) {
        Certificate[] certificates = null;
        try {
            KeyStore keyStore = KeyStore.getInstance(PkiConsts.KEYSTORE_TYPE);
            keyStore.load(null);

            KeyStore.Entry entry = keyStore.getEntry(clientID, null);
            if (entry == null) {
                return new Certificate[0];
            }

            if (!(entry instanceof KeyStore.PrivateKeyEntry)) {
                return new Certificate[0];
            }

            certificates = ((KeyStore.PrivateKeyEntry) entry).getCertificateChain();
        } catch (IOException | CertificateException | NoSuchAlgorithmException
                | UnrecoverableEntryException | KeyStoreException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch Exception: " + e.getMessage()));
        }

        return certificates;
    }

    /**
     * to hex format
     *
     * @param data byte[]
     * @return String
     */
    public static String toHexFormat(byte[] data) {
        if (data == null || data.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (byte byteData : data) {
            sb.append(String.format(Locale.ROOT, "%02x", byteData));
        }

        return sb.toString();
    }

    /**
     * gen equip cert hash
     *
     * @param clientID String
     * @return String
     */
    public static String genEquipCertHash(String clientID) {
        String equipCert;
        byte[] equipCertBytesHash = null;
        try {
            Certificate[] certificates = getCertificateChain(clientID);
            if (certificates == null || certificates.length < 2) {
                return "";
            }
            equipCert = readPemFormat(certificates[1]);
            equipCertBytesHash = getSHA256Str(equipCert);
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch Exception: " + e.getMessage()));
        }

        return toHexFormat(equipCertBytesHash);
    }

    /**
     * generate hash from cer
     *
     * @param certificateGradeOne X509Certificate
     * @return String
     */
    public static String genHashFromCer(X509Certificate certificateGradeOne) {
        String equipCert = null;
        byte[] equipCertBytesHash = null;
        try {
            equipCert = readPemFormat(certificateGradeOne);
            equipCertBytesHash = getSHA256Str(equipCert);
        } catch (IOException e) {
            LOGGER.severe(Common.addTag("[PkiUtil] catch Exception: " + e.getMessage()));
        }
        if (equipCertBytesHash == null) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (byte byteData : equipCertBytesHash) {
            sb.append(String.format(Locale.ROOT, "%02x", byteData));
        }
        return sb.toString();
    }

    /**
     * read pem format
     *
     * @param certificate Certificate
     * @return String result
     * @throws IOException e
     */
    public static String readPemFormat(Certificate certificate) throws IOException {
        StringWriter writer = new StringWriter();
        PemWriter pemWriter = new PemWriter(writer);
        if (certificate == null) {
            LOGGER.severe(Common.addTag("[PkiUtil] the input parameter certificate is null, please check"));
            throw new IllegalArgumentException();
        }
        try {
            pemWriter.writeObject(new PemObject("CERTIFICATE", certificate.getEncoded()));
        } catch (IOException | CertificateEncodingException e) {
            LOGGER.severe(
                    Common.addTag("[PkiUtil] catch IOException or CertificateEncodingException in getPermFormat: "
                            + e.getMessage()));
        } finally {
            pemWriter.flush();
            pemWriter.close();
        }

        return writer.toString();
    }
}
import hashlib

from config import config


def hash_password(plain_password, password_salt):
    digest = hashlib.pbkdf2_hmac(config.PASSWORD_HASH_ALGORITHM, plain_password.encode("utf-8"),
                                 password_salt.encode("utf-8"), config.PASSWORD_HASH_ROUNDS)
    return digest.hex()

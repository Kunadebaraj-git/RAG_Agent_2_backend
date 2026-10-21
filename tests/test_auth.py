"""Tests for auth: password hashing, JWT tokens."""

import os
os.environ.update({
    "DATABASE_URL": "postgresql+psycopg://test:test@localhost:5432/testdb",
    "SECRET_KEY": "test-secret-key-for-testing-only",
    "GROQ_API_KEY": "test-groq-key",
    "HF_API_TOKEN": "test-hf-token",
    "FRONTEND_URL": "http://localhost:5173",
})

import pytest
from uuid import uuid4
from app.auth import hash_password, verify_password, create_access_token, decode_token


class TestPasswordHashing:
    def test_hash_and_verify(self):
        password = "test_password_123"
        hashed = hash_password(password)
        assert hashed != password
        assert verify_password(password, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_each_time(self):
        password = "same_password"
        h1 = hash_password(password)
        h2 = hash_password(password)
        assert h1 != h2  # bcrypt uses random salts
        assert verify_password(password, h1)
        assert verify_password(password, h2)


class TestJWT:
    def test_create_and_decode_token(self):
        user_id = uuid4()
        token = create_access_token(user_id)
        decoded_id = decode_token(token)
        assert decoded_id == str(user_id)

    def test_invalid_token_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_empty_token_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            decode_token("")

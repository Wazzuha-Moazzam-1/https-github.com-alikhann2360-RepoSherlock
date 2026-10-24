"""Authentication helpers for the sample app."""
import hashlib
import secrets

from database import connect_database, fetch_user

SESSIONS = {}


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def validate_credentials(db, username, password):
    """Return True when the username/password pair matches a stored user."""
    user = fetch_user(db, username)
    if user is None:
        return False
    return user["password_hash"] == hash_password(password)


def create_session(username):
    """Create and store a new session token for a user."""
    token = secrets.token_hex(16)
    SESSIONS[token] = username
    return token


def login(config, username, password):
    """Log a user in and return a session token."""
    db = connect_database(config)
    if not validate_credentials(db, username, password):
        raise PermissionError("Invalid username or password")
    return create_session(username)

"""Database helpers for the sample app (an in-memory stand-in for a real database)."""
import time
import hashlib


class DatabaseError(Exception):
    pass


USERS = {
    "alice": {"password_hash": hashlib.sha256(b"wonderland").hexdigest()},
}


def connect_database(config, attempt=1):
    """Open a database connection using the app configuration."""
    url = config["db_url"]
    try:
        return _open_connection(url)
    except DatabaseError:
        return retry_connect(config, attempt)


def _open_connection(url):
    if not url.startswith("sqlite:///"):
        raise DatabaseError("Unsupported database URL: " + url)
    return {"url": url, "connected": True}


def retry_connect(config, attempt):
    """Try to connect again a few times before giving up."""
    if attempt >= 3:
        raise DatabaseError("Giving up after 3 attempts")
    time.sleep(0.1 * attempt)
    return connect_database(config, attempt + 1)


def fetch_user(db, username):
    """Look up a user record, or None if it does not exist."""
    return USERS.get(username)

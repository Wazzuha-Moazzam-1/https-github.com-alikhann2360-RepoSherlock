"""Entry point of the sample app."""
from auth import login
from database import connect_database


def build_config():
    """Collect settings for the app."""
    return {"database_url": "sqlite:///app.db", "debug": True}


def main():
    config = build_config()
    connect_database(config)
    session = login(config, "alice", "wonderland")
    print("Logged in with session:", session)


if __name__ == "__main__":
    main()

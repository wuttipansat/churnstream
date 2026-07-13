from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection

from churnstream.core.config import get_settings

@lru_cache
def get_mongodb_client() -> MongoClient:

    settings = get_settings()

    return MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms
    )

def get_collection() -> Collection:
    
    settings = get_settings()
    client = get_mongodb_client()

    database = client[settings.mongodb_database]
    collection = database[settings.mongodb_collection]

    return collection

def ping_mongodb() -> bool:

    client = get_mongodb_client()
    result = client.admin.command("ping")

    return result.get("ok") == 1



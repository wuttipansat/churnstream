from pymongo.errors import PyMongoError

from churnstream.core.config import get_settings
from churnstream.database.mongodb import get_collection, ping_mongodb

def main() -> None:
    settings = get_settings()

    try:
        connected = ping_mongodb()

        if not connected:
            print("MongoDB connection failed.")
            return
        
        collection = get_collection()
        document_count = collection.count_documents({})

        print("MongoDB connection successful.")
        print(f"Database: {settings.mongodb_database}")
        print(f"Collection: {settings.mongodb_collection}")
        print(f"Document count: {document_count}")

        sample = collection.find_one({}, {"_id": 0})

        if sample:
            print("\n Available fields:")
            print(list(sample.keys()))

        else:
            print("Collection is empty.")

    except PyMongoError as error:
        print(f"MongoDB error: {error}")

if __name__ == "__main__":
    main()

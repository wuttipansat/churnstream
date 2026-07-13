from typing import Any

import pandas as pd
from pymongo.collection import Collection

from churnstream.database.mongodb import get_collection

def load_data(
        query: dict[str, Any] | None = None,
        collection: Collection | None = None,
) -> pd.DataFrame:
    
    mongodb_collection = collection or get_collection()
    mongodb_query = query or {}

    documents = list(
        mongodb_collection.find(
            mongodb_query,
            {"_id": 0},
        )
    )

    if not documents:
        raise ValueError("No records were found in MongoDB.")
    
    return pd.DataFrame(documents)
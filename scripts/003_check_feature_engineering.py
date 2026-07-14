from pymongo.errors import PyMongoError

from churnstream.data.loader import load_data
from churnstream.features.config import build_feature_config
from churnstream.features.engineering import engineer_features

def main() -> None:
    try: 
        dataframe = load_data()

        config = build_feature_config()
        featured_dataframe = engineer_features(dataframe)

        print("Raw numeric:")
        print(config.numeric)

        print("\nEngineered numeric:")
        print(config.engineered_numeric)

        print("\nModel features:")
        print(config.model_features)

        print("\nFeatured dataframe:")
        print(featured_dataframe)


    except ValueError as error:
        print(f"Data or feature error: {error}")

    except PyMongoError as error:
        print(f"MongoDB error: {error}")

if __name__ == "__main__":
    main()

    
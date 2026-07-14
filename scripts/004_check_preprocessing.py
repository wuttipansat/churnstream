from sklearn.model_selection import train_test_split

from churnstream.data.loader import load_data
from churnstream.features.config import build_feature_config
from churnstream.features.engineering import engineer_features
from churnstream.preprocess.preprocessing import build_preprocessor

def main() -> None:
    dataframe = load_data()
    config = build_feature_config()

    X = dataframe.loc[:, list(config.input_features)]
    y = dataframe[config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    preprocessor = build_preprocessor(config)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Raw training shape:")
    print(X_train.shape)

    print("\nRaw testing shape:")
    print(X_test.shape)

    print("\nProcessed training shape:")
    print(X_train_processed.shape)

    print("\nProcessed testing shape:")
    print(X_test_processed.shape)

    print("\nTarget distribution:")
    print(y_train.value_counts(normalize=True))

if __name__ == "__main__":
    main()

import json

from churnstream.data.loader import load_data
from churnstream.data.validator import validate_data

def main() -> None:
    try:
        dataframe = load_data()

        print("Customer data loaded successfully.")
        print(f"Shape: {dataframe.shape}")
        print(f"Columns: {list(dataframe.columns)}")

        result = validate_data(
            dataframe=dataframe,
            target_column="Exited",
        )

        print("\nValidation report:")
        print(
            json.dumps(
                result.to_dict(),
                indent=2,
                ensure_ascii=False,
            )
        )

        if not result.is_valid:
            raise SystemExit(1)
        
    except Exception as error:
        print(f"Data validation failed: {error}")
        raise SystemExit(1) from error
    

if __name__ == "__main__":
    main()
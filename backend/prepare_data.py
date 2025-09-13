import pandas as pd
import os

def prepare_data():
    """
    Reads all CSV files from the 'data' directory, adds a 'category' column,
    and combines them into a single CSV file.
    """
    data_folder = 'data'
    output_filename = 'all_products.csv'
    
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: The '{data_folder}' directory was not found.")
        print("Please make sure you have your CSV files inside a 'data' folder in the 'backend' directory.")
        return

    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv') and f != output_filename]
    
    if not csv_files:
        print(f"Error: No source CSV files found in the '{data_folder}' directory.")
        return

    all_dfs = []
    
    print("Reading and processing CSV files...")
    for file in csv_files:
        try:
            # Extract category name from the filename (e.g., 'electronics.csv' -> 'Electronics')
            category = os.path.splitext(file)[0].capitalize()
            filepath = os.path.join(data_folder, file)
            
            df = pd.read_csv(filepath)
            df['category'] = category
            
            all_dfs.append(df)
            print(f" - Successfully processed {file}")
        except Exception as e:
            print(f" - Could not process {file}. Error: {e}")

    if not all_dfs:
        print("No dataframes were created. Exiting.")
        return

    # Combine all dataframes into one
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize column names (e.g., remove leading/trailing spaces, make lowercase)
    combined_df.columns = combined_df.columns.str.strip().str.lower()

    # Drop rows where 'reviews' or 'product_name' is missing, as they are crucial for our model
    combined_df.dropna(subset=['reviews', 'product_name'], inplace=True)
    
    # Save the combined and cleaned data to a new CSV file
    output_filepath = os.path.join(data_folder, output_filename)
    combined_df.to_csv(output_filepath, index=False)
    
    print(f"\nData preparation complete!")
    print(f"Combined data from {len(all_dfs)} files into '{output_filepath}'.")
    print(f"Total products: {len(combined_df)}")

if __name__ == '__main__':
    prepare_data()

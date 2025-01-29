from datetime import datetime
import os
import zipfile

from dateutil.rrule import MONTHLY, rrule
import pandas as pd
import requests

def download_and_extract_divvy_data(start_date, end_date, base_url="https://divvy-tripdata.s3.amazonaws.com/"):
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data will be downloaded and saved to: {data_dir}")

    dates = [dt.strftime('%Y%m') for dt in rrule(MONTHLY, dtstart=datetime.strptime(start_date, '%Y%m'), until=datetime.strptime(end_date, '%Y%m'))]

    csv_files = []

    for date_str in dates:
        zip_filename = f"{date_str}-divvy-tripdata.zip"
        zip_url = f"{base_url}{zip_filename}"
        csv_filename_base = zip_filename.replace(".zip", "")
        csv_filename = f"{csv_filename_base}.csv"
        zip_filepath = os.path.join(data_dir, zip_filename)
        csv_filepath = os.path.join(data_dir, csv_filename)

        print(f"Downloading: {zip_url}")
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()

            with open(zip_filepath, 'wb') as zip_file:
                for chunk in response.iter_content(chunk_size=8192):
                    zip_file.write(chunk)
            print(f"Downloaded: {zip_filename}")

            print(f"Extracting CSV from: {zip_filename}")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                if csv_filename in zip_ref.namelist():
                    zip_ref.extract(csv_filename, data_dir)
                    print(f"Extracted: {csv_filename}")
                    csv_files.append(csv_filepath)
                else:
                    print(f"Error: CSV file '{csv_filename}' not found inside {zip_filename}. Skipping.")

            os.remove(zip_filepath)
            print(f"Removed ZIP file: {zip_filename}")

        except Exception as e:
            print(f"An unexpected error occurred while processing {zip_filename}: {e}")

    if csv_files:
        print("\nConcatenating CSV files...")
        all_data = pd.DataFrame()
        for file in csv_files:
            try:
                print(f"Reading: {file}")
                df = pd.read_csv(file)
                all_data = pd.concat([all_data, df], ignore_index=True)
                print(f"Read and concatenated: {file}")
            except Exception as e:
                print(f"An unexpected error occurred while concatenating {file}: {e}")

        if not all_data.empty:
            combined_csv_path = os.path.join(data_dir, "combined_tripdata.csv")
            all_data.to_csv(combined_csv_path, index=False)
            print(f"\nAll CSVs concatenated and saved to: {combined_csv_path}")
        else:
            print("\nNo data to concatenate or an error occurred during concatenation.")
    else:
        print("\nNo CSV files were successfully downloaded and extracted to concatenate.")


if __name__ == "__main__":
    start_date = "202004"
    end_date = "202412"
    download_and_extract_divvy_data(start_date, end_date)
    print("\nScript finished.")

import csv
import plotly.graph_objects as go
from pathlib import Path

def read_csvs_from_folder(folder_path: str = "./results/") -> dict[str, list[dict]]:
    """
    Read all CSV files from the specified folder.

    Args:
        folder_path: Path to folder containing CSV files (default: "./results/")

    Returns:
        Dictionary with filename (without .csv) as keys and list of row dictionaries as values.
    """
    data_dict = {}
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder_path}' not found")

    for csv_file in folder.glob("*.csv"):
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = []
                for row in reader:
                    # Convert numeric columns to appropriate types
                    converted_row = {
                        'model': row['model'],
                        'power_draw_total': float(row['power_draw_total']),
                    }
                    rows.append(converted_row)

                data_dict[csv_file.stem] = rows

        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue

    return data_dict


if __name__ == "__main__":
    try:
        CC_data = read_csvs_from_folder("./CodeCarbon/results/")
        PJ_data = read_csvs_from_folder("./PyJoules/results/")

        print(f"Found {len(CC_data)} CSV files in CodeCarbon:")
        for filename, rows in CC_data.items():
            print(f"  {filename}.csv: {len(rows)} rows")

        print(f"Found {len(PJ_data)} CSV files in PyJoules:")
        for filename, rows in PJ_data.items():
            print(f"  {filename}.csv: {len(rows)} rows")

        CC_dict = {}
        models = []

        if CC_data:
            # Extract the dict with model and conusmed energy
            for file, row in CC_data.items():
                data = row[0]
                CC_dict[data['model']] = data['power_draw_total']
            # Sort data
            CC_dict = dict(sorted(CC_dict.items()))
            models = list(CC_dict.keys())
            CC_values = list(CC_dict.values())
            print(CC_dict)

        PJ_dict = {}
        PJ_values = []

        if PJ_data:
            # Extract the dict with model and conusmed energy
            for file, row in PJ_data.items():
                data = row[0]
                PJ_dict[data['model']] = data['power_draw_total'] + 2 * 5 * 30 * 60 #add 5W per 2 sticks of ram per 1800s run for PyJoules
            # Sort data
            PJ_dict= dict(sorted(PJ_dict.items()))
            PJ_values = list(PJ_dict.values())
            print(PJ_dict)

        fig = go.Figure(data=[
            go.Bar(name='CodeCarbon', x=models, y=CC_values),
            go.Bar(name='PyJoules', x=models, y=PJ_values)
        ])

        fig.update_layout(barmode='group')
        fig.show()

    except FileNotFoundError as e:
        print(e)

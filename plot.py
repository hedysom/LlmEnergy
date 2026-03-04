import csv
import plotly.graph_objects as go
from pathlib import Path

# TDP constants
TDP_CPU = 105  # Watts (CPU)
TDP_GPU = 350 # Watts (GPU)
TDP_RAM = 5        # Watts (RAM) per each stick
TDP_TOTAL = TDP_CPU + TDP_GPU + TDP_RAM  # 460W

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
                        'running_time_total': float(row['running_time_total']),
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
        CC_times = {}
        CC_values = []
        models = []

        if CC_data:
            # Extract the dict with model and conusmed energy
            for file, row in CC_data.items():
                data = row[0]
                CC_dict[data['model']] = data['power_draw_total']
                CC_times[data['model']] = data['running_time_total']
            # Sort dataa
            CC_dict = dict(sorted(CC_dict.items()))
            CC_times = dict(sorted(CC_times.items()))
            models = list(CC_dict.keys())
            CC_values = list(CC_dict.values())
            print(CC_dict)

        PJ_dict = {}
        PJ_times = {}
        PJ_values = []

        if PJ_data:
            # Extract the dict with model and conusmed energy
            for file, row in PJ_data.items():
                data = row[0]
                duration = data['running_time_total']
                PJ_dict[data['model']] = data['power_draw_total'] + 2 * TDP_RAM * duration #add 5W times 2 sticks of ram times time
                PJ_times[data['model']] = duration
                print(f'power draw {data['power_draw_total']}, ram energy {2 * TDP_RAM * duration},  time {duration}')
            # Sort data
            PJ_dict= dict(sorted(PJ_dict.items()))
            PJ_times= dict(sorted(PJ_times.items()))
            PJ_values = list(PJ_dict.values())
            print(PJ_dict)

        TDP_values = []
        for model in models:
            duration = CC_times.get(model) or PJ_times.get(model, 0)
            TDP_values.append(TDP_TOTAL * duration)

        fig = go.Figure(data=[
            go.Bar(name='CodeCarbon', x=models, y=CC_values),
            go.Bar(name='PyJoules', x=models, y=PJ_values),
            go.Bar(name='Static TDP', x=models, y=TDP_values,
                text=[f'{v:,.0f} J' for v in TDP_values],
                textposition='outside',
            ),
        ])

        fig.update_layout(barmode='group')
        fig.write_image("./figures/energy_comparison.pdf")
        print("Plot saved to figures/energy_comparison.pdf")
        fig.show()

    except FileNotFoundError as e:
        print(e)

import os
import json
import math
from dataset import HouseDataset, NetflixDataset
import analyzer
import visualizer
import report

def main():
    print("\n=========================================")
    print("      Analytics Dashboard Batch Run      ")
    print("=========================================\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(base_dir, '..', 'frontend_website')
    data_js_path = os.path.join(frontend_dir, 'data.js')
    
    datasets_to_process = [
        {
            "id": "dataset1",
            "name": "Housing Data (Green Theme)",
            "type": "housing",
            "files": {
                "house_csv": os.path.join(base_dir, 'data', 'house_prices.csv'),
                "location_csv": os.path.join(base_dir, 'data', 'location_info.csv')
            }
        },
        {
            "id": "dataset2",
            "name": "Netflix Titles (Red/Blue Theme)",
            "type": "netflix",
            "files": {
                "netflix_csv": os.path.join(base_dir, 'data', 'netflix_titles.csv')
            }
        }
    ]

    all_frontend_data = {}

    for ds_info in datasets_to_process:
        ds_id = ds_info["id"]
        ds_type = ds_info["type"]
        print(f"\n================ Processing {ds_info['name']} ================")
        
        if ds_type == "housing":
            dataset_manager = HouseDataset(ds_info["files"]["house_csv"], ds_info["files"]["location_csv"])
        elif ds_type == "netflix":
            dataset_manager = NetflixDataset(ds_info["files"]["netflix_csv"])

        print("--- Step 1: Loading Data ---")
        dataset_manager.load_data()
        
        print("\n--- Step 2: Cleaning Data ---")
        dataset_manager.clean_dataset()
        df = dataset_manager.get_data()
        
        if df is None:
            print("Skipping due to data load error.")
            continue
            
        print("\n--- Step 3: Generating Visualizations ---")
        visualizer.generate_all_visualizations(df, output_dir=os.path.join(frontend_dir, 'images'), prefix=f"{ds_id}_", dataset_type=ds_type)
        
        top_10 = df.head(10).to_dict(orient='records')
        operations = dataset_manager.get_operations()
        
        for row in top_10:
            for k, v in row.items():
                if isinstance(v, float) and math.isnan(v):
                    row[k] = None
                    
        if ds_type == "housing":
            chart_titles = ["Average Price by Location", "Price Distribution", "Area vs Price", "Price by Location"]
        else:
            chart_titles = ["Proportion of Movies vs TV Shows", "Release Year Distribution", "Top Content Ratings", "Top Production Countries"]

        all_frontend_data[ds_id] = {
            "name": ds_info["name"],
            "type": ds_type,
            "data": top_10,
            "operations": operations,
            "chartTitles": chart_titles
        }

    print("\n--- Step 4: Exporting Data for Frontend ---")
    js_content = f"const datasets = {json.dumps(all_frontend_data, indent=2)};"
    
    os.makedirs(frontend_dir, exist_ok=True)
    try:
        with open(data_js_path, 'w') as f:
            f.write(js_content)
        print(f"Successfully generated frontend data file at: {data_js_path}")
    except Exception as e:
        print(f"Error writing frontend data: {e}")

    print("\n=========================================")
    print(" Batch run completed successfully! ")
    print("=========================================\n")

if __name__ == "__main__":
    main()

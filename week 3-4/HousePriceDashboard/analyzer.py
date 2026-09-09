import pandas as pd
import numpy as np

def numpy_analysis(df):
    """Performs statistical analysis using NumPy."""
    if df is None or df.empty:
        print("No data available for analysis.")
        return

    print("\n--- NumPy Statistical Analysis ---")
    
    prices_array = df['Price'].to_numpy()
    area_array = df['Area'].to_numpy()
    
    print(f"Mean Price: ${np.mean(prices_array):.2f}")
    print(f"Median Price: ${np.median(prices_array):.2f}")
    print(f"Standard Deviation of Price: ${np.std(prices_array):.2f}")
    print(f"Maximum Price: ${np.max(prices_array)}")
    print(f"Minimum Price: ${np.min(prices_array)}")
    
    price_per_sqft = prices_array / area_array
    print(f"Average Price per Sq Ft: ${np.mean(price_per_sqft):.2f}")
    
    first_50_prices = prices_array[:50]
    print(f"Mean Price (First 50 Houses): ${np.mean(first_50_prices):.2f}")


def pandas_analysis(df):
    """Performs grouped and sorted analysis using Pandas."""
    if df is None or df.empty:
        print("No data available for analysis.")
        return

    print("\n--- Pandas Grouping and Sorting Analysis ---")
    
    print("\nAverage Price by Location:")
    grouped_location = df.groupby('Location')['Price'].mean().round(2)
    print(grouped_location)
    
    print("\nHouse Count by Number of Bedrooms:")
    grouped_beds = df.groupby('Bedrooms')['House_ID'].count()
    print(grouped_beds)
    
    print("\nTop 5 Most Expensive Houses:")
    top_houses = df.sort_values(by='Price', ascending=False).head(5)
    print(top_houses[['House_ID', 'Location', 'Price', 'Area', 'Bedrooms']])


def search_houses(df, location_query="Suburbs", min_price=200000, max_price=500000, min_beds=2):
    """Searches for houses based on predefined criteria (Simulated for batch run)."""
    if df is None or df.empty:
        print("No data available to search.")
        return

    print(f"\n--- Search Houses (Criteria: {location_query}, ${min_price}-${max_price}, {min_beds}+ beds) ---")

    filtered_df = df[
        (df['Price'] >= min_price) & 
        (df['Price'] <= max_price) & 
        (df['Bedrooms'] >= min_beds)
    ]
    
    if location_query:
        filtered_df = filtered_df[filtered_df['Location'].str.contains(location_query, case=False, na=False)]

    if filtered_df.empty:
        print("No houses found matching your criteria.")
    else:
        print(f"Found {len(filtered_df)} matching houses. Here are the top 5:")
        print(filtered_df[['House_ID', 'Location', 'Price', 'Bedrooms', 'Area']].head(5))
        
        matching_ids = [hid for hid in filtered_df['House_ID'].tolist()]
        
        id_to_price = {row['House_ID']: row['Price'] for index, row in filtered_df.head(5).iterrows()}
        print("\n(Sample) House ID to Price Mapping:", id_to_price)

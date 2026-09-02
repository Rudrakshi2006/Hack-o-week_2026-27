import pandas as pd

def generate_report(df, filename="report.txt"):
    """Generates a summary text report of the dataset."""
    if df is None or df.empty:
        print("No data available to generate report.")
        return

    try:
        num_houses = len(df)
        avg_price = df['Price'].mean()
        high_price = df['Price'].max()
        low_price = df['Price'].min()
        avg_area = df['Area'].mean()
        avg_bedrooms = df['Bedrooms'].mean()

        with open(filename, 'w') as file:
            file.write("=========================================\n")
            file.write("      HOUSE PRICE ANALYTICS REPORT       \n")
            file.write("=========================================\n\n")
            
            file.write(f"Total Number of Houses : {num_houses}\n")
            file.write(f"Average House Price    : ${avg_price:,.2f}\n")
            file.write(f"Highest House Price    : ${high_price:,.2f}\n")
            file.write(f"Lowest House Price     : ${low_price:,.2f}\n")
            file.write(f"Average Area (sq ft)   : {avg_area:,.2f}\n")
            file.write(f"Average Bedrooms       : {avg_bedrooms:.1f}\n\n")
            
            file.write("=========================================\n")
            file.write("        End of Summary Report            \n")
            file.write("=========================================\n")
            
        print(f"Report successfully generated and saved to '{filename}'.")
        
    except Exception as e:
        print(f"An error occurred while generating the report: {e}")

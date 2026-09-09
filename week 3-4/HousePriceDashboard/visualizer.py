import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_all_visualizations(df, output_dir="../frontend_website/images", prefix="", dataset_type="housing"):
    """Generates and saves all visualizations without showing popups."""
    if df is None or df.empty:
        print("No data available to visualize.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend('Agg') # Prevent popups

    print(f"Generating {dataset_type} visualizations...")
    
    if dataset_type == "housing":
        plot_bar_chart(df, os.path.join(output_dir, f"{prefix}chart1.png"))
        plot_histogram(df, os.path.join(output_dir, f"{prefix}chart2.png"))
        plot_scatter(df, os.path.join(output_dir, f"{prefix}chart3.png"))
        plot_boxplot(df, os.path.join(output_dir, f"{prefix}chart4.png"))
    elif dataset_type == "netflix":
        plot_netflix_type_pie(df, os.path.join(output_dir, f"{prefix}chart1.png"))
        plot_netflix_release_hist(df, os.path.join(output_dir, f"{prefix}chart2.png"))
        plot_netflix_rating_bar(df, os.path.join(output_dir, f"{prefix}chart3.png"))
        plot_netflix_country_bar(df, os.path.join(output_dir, f"{prefix}chart4.png"))
        
    print(f"Visualizations saved to {output_dir}")

def plot_bar_chart(df, save_path):
    plt.figure(figsize=(8, 5))
    avg_price = df.groupby('Location')['Price'].mean().sort_values()
    plt.bar(avg_price.index, avg_price.values, color='#4caf50')
    plt.title('Average House Price by Location')
    plt.xlabel('Location')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_histogram(df, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Price'].dropna(), bins=30, color='#81c784', edgecolor='black')
    plt.title('Distribution of House Prices')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_scatter(df, save_path):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['Area'], df['Price'], alpha=0.5, color='#388e3c')
    plt.title('House Area vs Price')
    plt.xlabel('Area (Sq Ft)')
    plt.ylabel('Price ($)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_boxplot(df, save_path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Location', y='Price', data=df, hue='Location', palette='Greens', legend=False)
    plt.title('Price Distribution by Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_netflix_type_pie(df, save_path):
    plt.figure(figsize=(8, 5))
    counts = df['Type'].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=['#e50914', '#f5f5f1'])
    plt.title('Proportion of Movies vs TV Shows')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_netflix_release_hist(df, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(df['Release_Year'].dropna(), bins=20, color='#e50914', edgecolor='black')
    plt.title('Distribution of Release Years')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_netflix_rating_bar(df, save_path):
    plt.figure(figsize=(8, 5))
    counts = df['Rating'].value_counts().head(5)
    plt.bar(counts.index, counts.values, color='#b20710')
    plt.title('Top 5 Content Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_netflix_country_bar(df, save_path):
    plt.figure(figsize=(8, 5))
    filtered = df[df['Country'] != 'Unknown']
    counts = filtered['Country'].value_counts().head(5)
    plt.bar(counts.index, counts.values, color='#e50914')
    plt.title('Top 5 Production Countries')
    plt.xlabel('Country')
    plt.ylabel('Titles Produced')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

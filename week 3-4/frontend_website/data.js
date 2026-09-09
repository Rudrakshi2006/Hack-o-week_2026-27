const datasets = {
  "dataset1": {
    "name": "Housing Data (Green Theme)",
    "type": "housing",
    "data": [
      {
        "House_ID": 1,
        "Location": "Rural",
        "Price": 215273.0,
        "Area": 1641.0,
        "Bedrooms": 2.0,
        "Bathrooms": 1,
        "Garage": 1,
        "YearBuilt": 1966,
        "CrimeRate": 1.2,
        "SchoolRating": 6.5
      },
      {
        "House_ID": 2,
        "Location": "Industrial",
        "Price": 181050.0,
        "Area": 909.0,
        "Bedrooms": 3.0,
        "Bathrooms": 3,
        "Garage": 1,
        "YearBuilt": 2017,
        "CrimeRate": 8.4,
        "SchoolRating": 4.5
      },
      {
        "House_ID": 3,
        "Location": "Uptown",
        "Price": 384097.0,
        "Area": 2614.0,
        "Bedrooms": 2.0,
        "Bathrooms": 2,
        "Garage": 1,
        "YearBuilt": 1997,
        "CrimeRate": 3.5,
        "SchoolRating": 8.0
      },
      {
        "House_ID": 4,
        "Location": "Industrial",
        "Price": 164486.0,
        "Area": 1274.0,
        "Bedrooms": 2.0,
        "Bathrooms": 1,
        "Garage": 2,
        "YearBuilt": 2001,
        "CrimeRate": 8.4,
        "SchoolRating": 4.5
      },
      {
        "House_ID": 5,
        "Location": "Industrial",
        "Price": 196208.0,
        "Area": 1324.0,
        "Bedrooms": 2.0,
        "Bathrooms": 2,
        "Garage": 1,
        "YearBuilt": 1974,
        "CrimeRate": 8.4,
        "SchoolRating": 4.5
      },
      {
        "House_ID": 6,
        "Location": "Suburbs",
        "Price": 243492.0,
        "Area": 1836.0,
        "Bedrooms": 1.0,
        "Bathrooms": 1,
        "Garage": 1,
        "YearBuilt": 1970,
        "CrimeRate": 2.1,
        "SchoolRating": 9.2
      },
      {
        "House_ID": 7,
        "Location": "Uptown",
        "Price": 354849.0,
        "Area": 2262.0,
        "Bedrooms": 1.0,
        "Bathrooms": 1,
        "Garage": 0,
        "YearBuilt": 2014,
        "CrimeRate": 3.5,
        "SchoolRating": 8.0
      },
      {
        "House_ID": 8,
        "Location": "Uptown",
        "Price": 379757.0,
        "Area": 2813.0,
        "Bedrooms": 2.0,
        "Bathrooms": 2,
        "Garage": 0,
        "YearBuilt": 2012,
        "CrimeRate": 3.5,
        "SchoolRating": 8.0
      },
      {
        "House_ID": 9,
        "Location": "Uptown",
        "Price": 309088.0,
        "Area": 2520.0,
        "Bedrooms": 5.0,
        "Bathrooms": 5,
        "Garage": 0,
        "YearBuilt": 2000,
        "CrimeRate": 3.5,
        "SchoolRating": 8.0
      },
      {
        "House_ID": 10,
        "Location": "Industrial",
        "Price": 284619.0,
        "Area": 1824.0,
        "Bedrooms": 3.0,
        "Bathrooms": 2,
        "Garage": 1,
        "YearBuilt": 1985,
        "CrimeRate": 8.4,
        "SchoolRating": 4.5
      }
    ],
    "operations": [
      "Dataset loaded successfully with 503 records.",
      "Removed 3 duplicate records.",
      "Filled 3 missing values with column medians/modes.",
      "Merged dataset with location statistics (CrimeRate, SchoolRating)."
    ],
    "chartTitles": [
      "Average Price by Location",
      "Price Distribution",
      "Area vs Price",
      "Price by Location"
    ]
  },
  "dataset2": {
    "name": "Netflix Titles (Red/Blue Theme)",
    "type": "netflix",
    "data": [
      {
        "Show_ID": "s1",
        "Type": "Movie",
        "Title": "Title 1",
        "Director": "Director 15",
        "Country": "Japan",
        "Release_Year": 2014,
        "Rating": "R",
        "Duration": "98 min"
      },
      {
        "Show_ID": "s2",
        "Type": "TV Show",
        "Title": "Title 2",
        "Director": "Director 1",
        "Country": "United Kingdom",
        "Release_Year": 2011,
        "Rating": "PG-13",
        "Duration": "3 Seasons"
      },
      {
        "Show_ID": "s3",
        "Type": "TV Show",
        "Title": "Title 3",
        "Director": "Unknown",
        "Country": "United States",
        "Release_Year": 2015,
        "Rating": "R",
        "Duration": "4 Seasons"
      },
      {
        "Show_ID": "s4",
        "Type": "Movie",
        "Title": "Title 4",
        "Director": "Director 98",
        "Country": "Canada",
        "Release_Year": 2023,
        "Rating": "R",
        "Duration": "85 min"
      },
      {
        "Show_ID": "s5",
        "Type": "Movie",
        "Title": "Title 5",
        "Director": "Unknown",
        "Country": "Japan",
        "Release_Year": 2018,
        "Rating": "R",
        "Duration": "124 min"
      },
      {
        "Show_ID": "s6",
        "Type": "Movie",
        "Title": "Title 6",
        "Director": "Director 33",
        "Country": "United States",
        "Release_Year": 2007,
        "Rating": "TV-Y",
        "Duration": "105 min"
      },
      {
        "Show_ID": "s7",
        "Type": "Movie",
        "Title": "Title 7",
        "Director": "Director 45",
        "Country": "Spain",
        "Release_Year": 2020,
        "Rating": "R",
        "Duration": "168 min"
      },
      {
        "Show_ID": "s8",
        "Type": "TV Show",
        "Title": "Title 8",
        "Director": "Director 28",
        "Country": "Canada",
        "Release_Year": 2016,
        "Rating": "R",
        "Duration": "1 Seasons"
      },
      {
        "Show_ID": "s9",
        "Type": "Movie",
        "Title": "Title 9",
        "Director": "Director 72",
        "Country": "Spain",
        "Release_Year": 2022,
        "Rating": "TV-14",
        "Duration": "165 min"
      },
      {
        "Show_ID": "s10",
        "Type": "TV Show",
        "Title": "Title 10",
        "Director": "Unknown",
        "Country": "South Korea",
        "Release_Year": 2004,
        "Rating": "TV-MA",
        "Duration": "3 Seasons"
      }
    ],
    "operations": [
      "Dataset loaded successfully with 503 records.",
      "Removed 3 duplicate records.",
      "Filled 171 missing values in Director/Country with 'Unknown'."
    ],
    "chartTitles": [
      "Proportion of Movies vs TV Shows",
      "Release Year Distribution",
      "Top Content Ratings",
      "Top Production Countries"
    ]
  }
};
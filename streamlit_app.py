import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib import patheffects
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Ensure datasets directories exist
os.makedirs("datasets", exist_ok=True)
os.makedirs("datasets/GemDataEXTR", exist_ok=True)

# Data source URLs
GEMDATA_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037798/DR0092042/GemDataEXTR.zip"
MAP_DATA_URL = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip"

# Page configuration
st.set_page_config(
    page_title="European Unemployment Rate Visualization", page_icon="🗺️", layout="wide"
)

# Title
st.title("🗺️ European Unemployment Rate Visualization")
st.markdown(
    "Interactive visualization of unemployment rates across Europe over the years"
)
st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing


# Helper function to download and extract zip files
def download_and_extract_zip(url: str, extract_to: str, zip_filename: str = None):
    """Download a zip file from URL and extract it to the specified directory"""
    os.makedirs(extract_to, exist_ok=True)
    
    if zip_filename is None:
        zip_filename = os.path.join(extract_to, os.path.basename(url))
    else:
        zip_filename = os.path.join(extract_to, zip_filename)
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, zip_filename)
    except Exception as e:
        st.error(f"Failed to download {url}: {str(e)}")
        st.stop()
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        st.error(f"Failed to extract {zip_filename}: {str(e)}")
        st.stop()
    
    # Clean up the zip file after extraction
    try:
        os.remove(zip_filename)
    except Exception:
        pass  # Ignore errors when removing zip file


# Cache data loading functions for better performance
@st.cache_data
def load_unemployment_data():
    """Load and clean unemployment rate data"""
    # Ensure directory exists
    os.makedirs("datasets/GemDataEXTR", exist_ok=True)

    # Try XLSX first, then CSV as fallback
    xlsx_filename = "datasets/GemDataEXTR/Unemployment Rate, seas. adj..xlsx"
    
    # If files don't exist, download and extract the zip
    if not os.path.exists(xlsx_filename):
        gemdata_zip = "datasets/GemDataEXTR.zip"
        if not os.path.exists(gemdata_zip):
            with st.spinner("Downloading unemployment data from World Bank..."):
                download_and_extract_zip(GEMDATA_URL, "datasets", "GemDataEXTR.zip")
        else:
            # Extract if zip exists but files don't
            with st.spinner("Extracting unemployment data..."):
                with zipfile.ZipFile(gemdata_zip, 'r') as zip_ref:
                    zip_ref.extractall("datasets")
                # Clean up zip after extraction
                try:
                    os.remove(gemdata_zip)
                except Exception:
                    pass
        
        # Handle nested folder structure (zip might extract to datasets/GemDataEXTR/GemDataEXTR/)
        extracted_folder = os.path.join("datasets", "GemDataEXTR", "GemDataEXTR")
        target_folder = os.path.join("datasets", "GemDataEXTR")
        if os.path.exists(extracted_folder):
            # Move all files from nested folder to target folder
            for item in os.listdir(extracted_folder):
                src = os.path.join(extracted_folder, item)
                dst = os.path.join(target_folder, item)
                if os.path.exists(dst):
                    # Remove existing file/folder if it exists
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            # Remove the now empty nested folder
            try:
                os.rmdir(extracted_folder)
            except Exception:
                pass
    
    if os.path.exists(xlsx_filename):
        df = pd.read_excel(xlsx_filename, header=0)
    else:
        st.error(
            f"❌ **Data file not found after download!**\n\n"
            f"Please check the download URL: {GEMDATA_URL}"
        )
        st.stop()

    # Rename the first column to 'Year'
    df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)

    # Remove the empty first row and convert Year to integer
    df = df.drop(0)
    df["Year"] = df["Year"].astype(int)

    # Melt the dataframe to long format (Year, Country, Unemployment)
    df_melted = df.melt(id_vars=["Year"], var_name="Country", value_name="Unemployment")

    # Convert Unemployment rate to numeric (handling European decimal format)
    df_melted["Unemployment"] = (
        df_melted["Unemployment"]
        .astype(str)
        .str.replace(",", ".")
        .apply(pd.to_numeric, errors="coerce")
    )

    return df_melted


@st.cache_data
def load_map_data():
    """Load Natural Earth map data"""
    # Ensure directory exists
    os.makedirs("datasets", exist_ok=True)

    map_file = "datasets/ne_10m_admin_0_countries.zip"
    if not os.path.exists(map_file):
        # Download the map file if it doesn't exist
        with st.spinner("Downloading map data from Natural Earth..."):
            download_and_extract_zip(MAP_DATA_URL, "datasets", "ne_10m_admin_0_countries.zip")

    world = gpd.read_file(map_file)

    # Filter for Europe only (excluding Turkey)
    europe_map = world[
        (world["CONTINENT"] == "Europe") & (world["SOVEREIGNT"] != "Turkey")
    ].copy()

    return europe_map


# Load data
with st.spinner("Loading data..."):
    df_melted = load_unemployment_data()
    europe_map = load_map_data()

# Get available years
available_years = sorted(df_melted["Year"].unique())
min_year = available_years[0]
max_year = available_years[-1]

# Sidebar for year selection
st.sidebar.header("Settings")
selected_year = st.sidebar.slider(
    "Select Year",
    min_value=min_year,
    max_value=max_year,
    value=2024 if 2024 in available_years else max_year,
    step=1,
)

# Filter data for selected year
df_target = df_melted[df_melted["Year"] == selected_year].copy()

# Name mapping to match CSV names to Map SOVEREIGNT names
name_mapping = {
    "Russian Federation": "Russia",
    "Czech Republic": "Czechia",
    "North Macedonia": "Macedonia",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Slovak Republic": "Slovakia",
    "United Kingdom": "United Kingdom",
}

df_target["Country_Map"] = df_target["Country"].replace(name_mapping)


# Merge data
merged = europe_map.merge(
    df_target, left_on="SOVEREIGNT", right_on="Country_Map", how="left"
)

# Calculate European statistics
europe_data = merged["Unemployment"].dropna()

if len(europe_data) == 0:
    st.warning(f"No unemployment data available for year {selected_year}")
    st.stop()

europe_avg = europe_data.mean()
europe_min = europe_data.min()
europe_max = europe_data.max()

# Calculate symmetric range around European average for diverging colormap
max_deviation = max(abs(europe_max - europe_avg), abs(europe_min - europe_avg))
vmin = max(0, europe_avg - max_deviation)
vmax = europe_avg + max_deviation

# Display statistics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("European Average", f"{europe_avg:.2f}%")
with col2:
    st.metric("Minimum", f"{europe_min:.2f}%")
with col3:
    st.metric("Maximum", f"{europe_max:.2f}%")
with col4:
    st.metric("Countries with Data", len(europe_data))

# Add small margin before map
st.markdown("<br>", unsafe_allow_html=True)

# Define fixed geographic bounds for Europe (consistent across all years)
x_min, x_max = -25, 45
y_min, y_max = 32, 72

# Calculate geographic aspect ratio
geo_width = x_max - x_min  # 70 degrees longitude
geo_height = y_max - y_min  # 40 degrees latitude
geo_aspect = geo_width / geo_height  # 1.75

# Create the plot with smaller fixed size to fit on screen
# Reduced dimensions to fit better on screen without scrolling
fig_width = 12
fig_height = 7  # Reduced height to fit on screen

# Use constrained_layout with tighter padding for more compact display
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)

# Set fixed axis limits BEFORE plotting to ensure consistent map size
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Set aspect ratio to 'equal' to prevent warping
# Use default adjustable='datalim' to maintain geographic accuracy
ax.set_aspect("equal")

# Create normalization centered at European average
norm = TwoSlopeNorm(vmin=vmin, vcenter=europe_avg, vmax=vmax)

# Separate countries with data from those without data
merged_with_data = merged[merged["Unemployment"].notna()].copy()
merged_no_data = merged[merged["Unemployment"].isna()].copy()

# Plot countries with data first
# Use consistent legend positioning to prevent layout shifts
merged_with_data.plot(
    column="Unemployment",
    ax=ax,
    legend=True,
    legend_kwds={
        "label": f"Unemployment Rate (%) in {selected_year} (Europe Avg: {europe_avg:.2f}%)",
        "orientation": "horizontal",
        "shrink": 0.8,
        "pad": 0.02,  # Slightly more padding for consistency
        "aspect": 20,
        "location": "bottom",  # Fixed location
    },
    cmap="coolwarm",
    norm=norm,
    edgecolor="black",
    linewidth=0.5,
)

# Add unemployment rate labels inside each country
# Group by country to calculate centroid of union of all geometries (handles multi-part countries)
for country_name in merged_with_data["SOVEREIGNT"].unique():
    # Get all geometries for this country
    country_data = merged_with_data[merged_with_data["SOVEREIGNT"] == country_name]

    # Skip if no unemployment data
    if country_data["Unemployment"].isna().all():
        continue

    # Get unemployment value (should be same for all rows of same country)
    unemployment_value = country_data["Unemployment"].dropna().iloc[0]
    if pd.isna(unemployment_value):
        continue

    # Determine label position: use centroid for all countries except Russia
    if country_name == "Russia":
        # Russia: use capital city (Moscow) since centroid is outside Europe bounds
        label_x = max(x_min, min(x_max, 37.6))
        label_y = max(y_min, min(y_max, 55.8))
    elif country_name == "France":
        # France: use hard-coded geographical center of France (mainland)
        # Coordinates: approximately 2.2°E, 46.6°N (center of France)
        label_x = max(x_min, min(x_max, 2.2))
        label_y = max(y_min, min(y_max, 46.6))
    else:
        # All other countries: calculate centroid of union of all geometries
        # This ensures we get the true geographic center for multi-part countries
        try:
            # Union all geometries for this country
            country_union = country_data.geometry.unary_union
            # Calculate centroid of the union
            union_centroid = gpd.GeoSeries([country_union]).centroid.iloc[0]

            if (
                x_min <= union_centroid.x <= x_max
                and y_min <= union_centroid.y <= y_max
            ):
                # Centroid is within bounds, use it
                label_x, label_y = union_centroid.x, union_centroid.y
            else:
                # Centroid outside bounds and no capital city, skip
                continue
        except Exception:
            # Fallback to first geometry's centroid if union fails
            first_geometry = country_data.iloc[0].geometry
            centroid = first_geometry.centroid
            if x_min <= centroid.x <= x_max and y_min <= centroid.y <= y_max:
                label_x, label_y = centroid.x, centroid.y
            else:
                continue

    # Format the value to 1 decimal place
    label_text = f"{unemployment_value:.1f}"

    # Add text annotation with outline for readability
    text = ax.text(
        label_x,
        label_y,
        label_text,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        zorder=10,
    )

    # Add white outline to make text readable on any background color
    text.set_path_effects(
        [patheffects.withStroke(linewidth=3, foreground="white", alpha=0.8)]
    )

# Plot countries with no data separately with diagonal hatching
if len(merged_no_data) > 0:
    # Track number of collections before plotting no-data countries
    collections_before = len(ax.collections)

    # Plot no-data countries with light grey
    merged_no_data.plot(
        ax=ax, color="lightgrey", edgecolor="black", linewidth=0.5, zorder=0
    )

    # Apply diagonal hatching to the newly added collection(s)
    for i, collection in enumerate(ax.collections):
        if i >= collections_before:
            collection.set_hatch("///")
            collection.set_facecolor("white")
            collection.set_edgecolor("black")

    # Add legend entry for no data
    no_data_patch = Patch(
        facecolor="white", edgecolor="black", hatch="///", label="No Data"
    )
    handles, labels = ax.get_legend_handles_labels()
    if "No Data" not in labels:
        handles.append(no_data_patch)
        labels.append("No Data")
        ax.legend(handles=handles, labels=labels, loc="upper left")

# Ensure axis limits remain fixed (in case plotting changed them)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Ensure aspect ratio is maintained (critical for consistent sizing)
ax.set_aspect("equal")

ax.set_axis_off()

# Ensure figure size remains fixed (critical for consistency across years)
# This must be done after all plotting to override any automatic adjustments
fig.set_size_inches(fig_width, fig_height)

# Add container with centered map and margins
col_left, col_center, col_right = st.columns([1, 10, 1])
with col_center:
    # Display the plot with use_container_width to fill available space
    st.pyplot(fig, use_container_width=True, clear_figure=False)

# Add small margin after map
st.markdown("<br>", unsafe_allow_html=True)

# Footer with information (more compact)
st.markdown("---")
st.caption(
    "**Data Source:** Unemployment Rate, seas. adj. (World Bank)  |  "
    "**Map Data:** Natural Earth (ne_10m_admin_0_countries)  |  "
    "**Colormap:** Coolwarm (diverging, centered at European average)"
)

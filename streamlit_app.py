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

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
BASE_DIR = SCRIPT_DIR

# Ensure datasets directories exist
datasets_dir = BASE_DIR / "datasets"
os.makedirs(datasets_dir, exist_ok=True)

# Data source URLs
GEMDATA_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037798/DR0092042/GemDataEXTR.zip"
MAP_DATA_URL = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"

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
def download_and_extract_zip(url: str, extract_to: Path, zip_filename: str = None):
    """Download a zip file from URL and extract it to the specified directory"""
    extract_to = Path(extract_to)
    os.makedirs(extract_to, exist_ok=True)
    
    if zip_filename is None:
        zip_path = extract_to / os.path.basename(url)
    else:
        zip_path = extract_to / zip_filename
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, str(zip_path))
    except Exception as e:
        st.error(f"Failed to download {url}: {str(e)}")
        st.stop()
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        st.error(f"Failed to extract {zip_path}: {str(e)}")
        st.stop()
    
    # Clean up the zip file after extraction
    try:
        zip_path.unlink()
    except Exception:
        pass  # Ignore errors when removing zip file


# Cache data loading functions for better performance
@st.cache_data
def load_unemployment_data():
    """Load and clean unemployment rate data"""
    # Use absolute paths relative to script location
    # Check both in datasets/ and datasets/GemDataEXTR/
    xlsx_path = datasets_dir / "Unemployment Rate, seas. adj..xlsx"
    xlsx_path_subdir = datasets_dir / "GemDataEXTR" / "Unemployment Rate, seas. adj..xlsx"
    gemdata_zip_path = datasets_dir / "GemDataEXTR.zip"
    
    # Determine which path has the file
    if xlsx_path_subdir.exists():
        xlsx_path = xlsx_path_subdir
    elif not xlsx_path.exists():
        # If files don't exist, download and extract the zip
        if not gemdata_zip_path.exists():
            with st.spinner("Downloading unemployment data from World Bank..."):
                download_and_extract_zip(GEMDATA_URL, datasets_dir, "GemDataEXTR.zip")
        else:
            # Extract if zip exists but files don't
            with st.spinner("Extracting unemployment data..."):
                with zipfile.ZipFile(gemdata_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(datasets_dir)
                # Clean up zip after extraction
                try:
                    gemdata_zip_path.unlink()
                except Exception:
                    pass
        
        # Check if files are in GemDataEXTR subdirectory
        gemdata_subdir = datasets_dir / "GemDataEXTR"
        if gemdata_subdir.exists() and (gemdata_subdir / "Unemployment Rate, seas. adj..xlsx").exists():
            xlsx_path = gemdata_subdir / "Unemployment Rate, seas. adj..xlsx"
        elif xlsx_path_subdir.exists():
            xlsx_path = xlsx_path_subdir
        elif not xlsx_path.exists():
            # List available files for debugging
            available_files = list(datasets_dir.glob("*"))
            if gemdata_subdir.exists():
                available_files.extend(list(gemdata_subdir.glob("*")))
            file_list = "\n".join([f"- {f.name}" for f in available_files[:20]])
            st.error(
                f"❌ **Data file not found after download!**\n\n"
                f"Looking for: `Unemployment Rate, seas. adj..xlsx`\n\n"
                f"Available files:\n{file_list}\n\n"
                f"Please check the download URL: {GEMDATA_URL}"
            )
            st.stop()
    
    # Read the file
    df = pd.read_excel(xlsx_path, header=0)

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
    map_file = datasets_dir / "ne_10m_admin_0_countries.zip"
    if not map_file.exists():
        # Download the map file if it doesn't exist (without extracting)
        with st.spinner("Downloading map data from Natural Earth..."):
            try:
                urllib.request.urlretrieve(MAP_DATA_URL, str(map_file))
            except Exception as e:
                st.error(f"Failed to download map data: {str(e)}")
                st.stop()

    # Read directly from zip file using /vsizip/ path
    # Geopandas can read shapefiles from zip files directly
    zip_path_str = f"/vsizip/{map_file}"
    world = gpd.read_file(zip_path_str)

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
    elif country_name == "United Kingdom":
        # United Kingdom: use hard-coded geographical center of United Kingdom (mainland)
        # Coordinates: approximately -2.2°W, 51.6°N (center of United Kingdom)
        label_x = max(x_min, min(x_max, -2.2))
        label_y = max(y_min, min(y_max, 51.6))
    elif country_name == "Norway":
        # Norway: use hard-coded geographical center of Norway (mainland)
        # Coordinates: approximately 10.0°E, 62.0°N (center of Norway)
        label_x = max(x_min, min(x_max, 10.0))
        label_y = max(y_min, min(y_max, 62.0))
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
st.markdown(
    "**Data Sources:**  \n"
    f"[Unemployment Rate, seas. adj. (World Bank)]({GEMDATA_URL})  \n"
    f"[Natural Earth Map Data (ne_10m_admin_0_countries)]({MAP_DATA_URL})  \n"
)

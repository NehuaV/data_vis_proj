import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import patheffects
import os
import urllib.request
import zipfile
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(
    page_title="European Unemployment Rate Visualization", page_icon="🗺️", layout="wide"
)

# Constants
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "datasets"
GEMDATA_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037798/DR0092042/GemDataEXTR.zip"
MAP_DATA_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
)

# Fixed Bounds for Europe
X_MIN, X_MAX = -25, 45
Y_MIN, Y_MAX = 32, 72


# --- DATA HELPERS ---
def ensure_directories():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_file(url: str, dest_path: Path):
    if not dest_path.exists():
        try:
            urllib.request.urlretrieve(url, str(dest_path))
        except Exception as e:
            st.error(f"Failed to download {url}: {e}")
            st.stop()


def extract_zip(zip_path: Path, extract_to: Path):
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except Exception as e:
        st.error(f"Failed to extract {zip_path}: {e}")
        st.stop()


# --- LOAD DATA ---
@st.cache_data
def load_data():
    ensure_directories()

    # 1. Map Data
    map_zip = DATA_DIR / "ne_10m_admin_0_countries.zip"
    download_file(MAP_DATA_URL, map_zip)

    try:
        world = gpd.read_file(f"/vsizip/{map_zip}")
        europe_map = world[
            (world["CONTINENT"] == "Europe") & (world["SOVEREIGNT"] != "Turkey")
        ].copy()
    except Exception as e:
        st.error(f"Error loading map: {e}")
        st.stop()

    # 2. Unemployment Data
    gem_zip = DATA_DIR / "GemDataEXTR.zip"
    excel_file = DATA_DIR / "Unemployment Rate, seas. adj..xlsx"
    subdir_file = DATA_DIR / "GemDataEXTR" / "Unemployment Rate, seas. adj..xlsx"

    if not (excel_file.exists() or subdir_file.exists()):
        download_file(GEMDATA_URL, gem_zip)
        extract_zip(gem_zip, DATA_DIR)
        try:
            gem_zip.unlink()
        except:
            pass

    target_file = subdir_file if subdir_file.exists() else excel_file

    if not target_file.exists():
        st.error("Data file missing.")
        st.stop()

    df = pd.read_excel(target_file, header=0)
    df.rename(columns={"Unnamed: 0": "Year"}, inplace=True)
    df = df.drop(0)
    df["Year"] = df["Year"].astype(int)

    df_melted = df.melt(id_vars=["Year"], var_name="Country", value_name="Unemployment")

    df_melted["Unemployment"] = (
        df_melted["Unemployment"]
        .astype(str)
        .str.replace(",", ".")
        .apply(pd.to_numeric, errors="coerce")
    )

    name_mapping = {
        "Russian Federation": "Russia",
        "Czech Republic": "Czechia",
        "North Macedonia": "Macedonia",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Slovak Republic": "Slovakia",
        "United Kingdom": "United Kingdom",
    }
    df_melted["Country_Map"] = df_melted["Country"].replace(name_mapping)

    return europe_map, df_melted


# --- MAIN APP ---
st.title("🗺️ European Unemployment Rate Visualization")
st.caption("Interactive visualization of unemployment rates across Europe")

with st.spinner("Loading data..."):
    europe_map, df_melted = load_data()

# --- GLOBAL STATISTICS ---
euro_data = df_melted[df_melted["Country_Map"].isin(europe_map["SOVEREIGNT"])]
euro_min = euro_data["Unemployment"].min()
euro_max = euro_data["Unemployment"].max()
euro_mean = euro_data["Unemployment"].mean()

max_deviation = max(abs(euro_max - euro_mean), abs(euro_min - euro_mean))
norm = TwoSlopeNorm(
    vmin=max(0, euro_mean - max_deviation),
    vcenter=euro_mean,
    vmax=euro_mean + max_deviation,
)

# --- LAYOUT ---
col_left, col_right = st.columns([5, 5], gap="large")

with col_left:
    st.markdown("### Settings")

    available_years = sorted(df_melted["Year"].unique())
    selected_year = st.slider(
        "Select Year",
        min_value=available_years[0],
        max_value=available_years[-1],
        value=2024 if 2024 in available_years else available_years[-1],
    )

    df_year = df_melted[df_melted["Year"] == selected_year].copy()
    merged = europe_map.merge(
        df_year, left_on="SOVEREIGNT", right_on="Country_Map", how="left"
    )

    year_data = merged["Unemployment"].dropna()

    if len(year_data) > 0:
        st.markdown("---")
        st.markdown(f"### Statistics for {selected_year}")
        m1, m2 = st.columns(2)
        m1.metric("Avg Rate", f"{year_data.mean():.2f}%")
        m2.metric("Max Rate", f"{year_data.max():.2f}%")

        m3, m4 = st.columns(2)
        m3.metric("Min Rate", f"{year_data.min():.2f}%")
        m4.metric("Countries", f"{len(year_data)}")

        max_country = merged.loc[merged["Unemployment"].idxmax()]
        st.info(
            f"🚨 Highest: **{max_country['SOVEREIGNT']}** ({max_country['Unemployment']:.1f}%)"
        )
    else:
        st.warning(f"No data available for {selected_year}")

with col_right:
    # 10x10 Figure
    fig = plt.figure(figsize=(10, 10))

    # Axes placement
    ax = fig.add_axes([0.05, 0.2, 0.9, 0.75])

    data_exists = merged[merged["Unemployment"].notna()]
    no_data = merged[merged["Unemployment"].isna()]

    # 1. Plot Background
    if not no_data.empty:
        # Pass aspect=None so Geopandas doesn't fight us yet
        no_data.plot(
            ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", hatch="////", aspect=None
        )

    # 2. Plot Data
    if not data_exists.empty:
        data_exists.plot(
            column="Unemployment",
            ax=ax,
            cmap="coolwarm",
            norm=norm,
            edgecolor="black",
            linewidth=0.5,
            aspect=None,  # Stop Geopandas from resetting aspect to 'equal'
        )

        for _, row in data_exists.iterrows():
            if pd.isna(row["Unemployment"]):
                continue

            country = row["SOVEREIGNT"]
            coords = None
            if country == "Russia":
                coords = (37.6, 55.8)
            elif country == "France":
                coords = (2.2, 46.6)
            elif country == "Norway":
                coords = (9.0, 61.0)
            elif country == "United Kingdom":
                coords = (-2.0, 52.5)
            else:
                cent = row.geometry.centroid
                if X_MIN <= cent.x <= X_MAX and Y_MIN <= cent.y <= Y_MAX:
                    coords = (cent.x, cent.y)

            if coords:
                txt = ax.text(
                    coords[0],
                    coords[1],
                    f"{row['Unemployment']:.1f}",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    color="black",
                )
                txt.set_path_effects(
                    [
                        patheffects.withStroke(
                            linewidth=2.5, foreground="white", alpha=0.8
                        )
                    ]
                )

    # 3. SET ASPECT RATIO & LIMITS LAST
    # We set this AFTER plotting so Geopandas doesn't overwrite it
    ax.set_aspect(1.5)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_axis_off()

    ax_pos = ax.get_position()
    # Align colorbar below map
    cax = fig.add_axes([ax_pos.x0, ax_pos.y0 - 0.02, ax_pos.width, 0.02])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Unemployment Rate (%)")

    st.pyplot(fig, width="stretch")

st.markdown("---")
st.markdown(
    """
    Data Sources: 
    [World Bank](https://data.worldbank.org/indicator/SL.UEM.TOTL.ZS) & [Natural Earth](https://www.naturalearthdata.com/)
    """
)

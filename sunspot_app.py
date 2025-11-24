import streamlit as st
import pandas as pd

from sun import Sun
from datetime import datetime, timezone
from streamlit_image_zoom import image_zoom
from typing import List


def highlight_visibility(row) -> List[str]:
    """Highlights dataframe rows based on the visibility classification of sunspots."""
    if row["visibility"] == "very easy":
        return ["font-weight: bold; color: green"] * len(row)
    elif row["visibility"] == "easy":
        return ["font-weight: bold; color: lightblue"] * len(row)
    elif row["visibility"] == "possible":
        return ["font-weight: bold; color: orange"] * len(row)
    elif row["visibility"] == "difficult":
        return ["font-weight: bold; color: red"] * len(row)
    elif row["visibility"] == "very difficult":
        return ["font-weight: bold; color: darkred"] * len(row)
    return [""] * len(row)


def get_visibility_classification(angular_size_arcmin: float) -> str:
    if angular_size_arcmin > 2.7:
        return "very easy"
    elif angular_size_arcmin > 2.0:
        return "easy"
    elif angular_size_arcmin > 1.0:
        return "possible"
    elif angular_size_arcmin > 0.75:
        return "difficult"
    elif angular_size_arcmin > 0.5:
        return "very difficult"
    else:
        return "not visible"


def display_sunspots_table(df_sunspots: pd.DataFrame):
    """
    Displays a styled sunspot data table in the Streamlit app.

    Parameters:
        df_sunspots (pandas.DataFrame): Pandas DataFrame of sunspots.
    """
    st.markdown(
        """<div style='font-size: 16px; margin-bottom: 15px'>
                <span style='color: red; font-weight: bold;'>WARNING:</span> 
                Directly observing the Sun can cause <b>severe eye damage</b>! Use proper protection such as eclipse glasses compliant with the ISO 12312-2 international standard.
            </div>""",
        unsafe_allow_html=True,
    )

    if df_sunspots.empty:
        st.markdown(
            """<div style='text-align: center; font-size: 20px; margin-top: 80px'>
                    <b>No sunspots detected.</b>
                </div>""",
            unsafe_allow_html=True,
        )
        return

    df = df_sunspots.copy()

    df["visibility"] = df["angular_size_arcmin"].apply(get_visibility_classification)
    df["diameter_km"] = df["diameter_km"].apply(lambda x: f"{x:,.0f}")
    df["angular_size_arcmin"] = df["angular_size_arcmin"].apply(lambda x: f"{x:.2f}")

    column_order = [
        "id",
        "diameter_km",
        "angular_size_arcmin",
        "visibility",
    ]
    column_config = {
        "id": "Sunspot ID",
        "diameter_km": "Diameter (km)",
        "angular_size_arcmin": "Angular size (arcmin)",
        "visibility": "Naked-eye visibility",
    }

    st.dataframe(
        df.style.apply(highlight_visibility, axis=1),
        column_order=column_order,
        column_config=column_config,
        hide_index=True,
        key="st_table",
        use_container_width=True,
        on_select="rerun",
    )


def show_sun_image(sun: Sun):
    """Shows sun image with detected sunspots."""
    st.markdown(
        f"**Image date and time:** {sun.date_time.strftime('%b %d, %Y - %H:%M')} (UTC)."
    )
    st.caption(
        """<div style='text-align: center; font-size: 13px; margin-bottom: -10px;'>
               Scroll on the image to zoom in.
            </div>""",
        unsafe_allow_html=True,
    )

    sun_img = sun.annotated_img
    if "st_table" in st.session_state:
        highlighted_ids = [
            st.session_state.df_sunspots.iloc[i]["id"]
            for i in st.session_state.st_table["selection"]["rows"]
        ]
        if highlighted_ids:
            sun_img = sun.highlight_sunspots(highlighted_ids)
    st.markdown("""
    <style>
    .sun-img-container img {
        width: 100% !important;
        height: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='sun-img-container'>", unsafe_allow_html=True)
    image_zoom(
    sun_img,
    mode="both",
    size=None,
    keep_resolution=True,
    zoom_factor=10.0,
    increment=0.5,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(
        """<div style='text-align: right; font-size: 13px; margin-top: -20px;'>
               Courtesy of NASA/SDO and the AIA, EVE, and HMI science teams.
            </div>""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        layout="centered", page_icon="🌞", page_title="Visible Sunspot Detector"
    )
    st.title("Visible Sunspot Detector")

    st.sidebar.header("Image Source")
    image_choice = st.sidebar.radio(
        "**Image source**",
        ["Latest Image", "Custom Date and Time"],
        label_visibility="collapsed",
        on_change=(lambda: st.session_state.clear()),
    )

    if image_choice == "Latest Image":
        sun = Sun()
    else:
        date = st.sidebar.date_input(
            "Date",
            value=datetime(2012, 7, 12, tzinfo=timezone.utc),
            min_value=datetime(2010, 5, 1, tzinfo=timezone.utc),
            max_value=datetime.now().date(),
            format="DD/MM/YYYY",
            on_change=(lambda: st.session_state.clear()),
        )
        time = st.sidebar.time_input(
            "Time", value="12:00", on_change=(lambda: st.session_state.clear())
        )

        sun = Sun(datetime.combine(date, time, tzinfo=timezone.utc))

    st.session_state.df_sunspots = pd.DataFrame(sun.sunspots)

    show_sun_image(sun)

    st.subheader("Identified sunspots")
    display_sunspots_table(st.session_state.df_sunspots)


if __name__ == "__main__":
    main()

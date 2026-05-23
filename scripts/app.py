from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# Load data
base_path = Path("E:/Data Storage/MDA/mda-cycling-traffic-analysis")
processed_folder = base_path / "data" / "processed"
model_development_data = pd.read_csv(processed_folder / "counts_model_final.csv", low_memory=False)
prediction_data = pd.read_csv(
    processed_folder / "prediction_data_with_factors.csv",
    dtype={
        'outdoor_music_event_type': str,
        'indoor_music_event_type': str,
        'sport_event_type': str
    }
)
expected_counts = pd.read_csv(
    processed_folder / "prediction_data_with_factors.csv",
    dtype={
        'outdoor_music_event_type': str,
        'indoor_music_event_type': str,
        'sport_event_type': str
    }
)
# 1. UI DEFINITION
app_ui = ui.page_navbar(
    # EDA SITE
    ui.nav_panel("Exploratory Data Analysis",
        ui.card(
            ui.markdown("""
                This EDA provides a comparative analysis of two primary cycling traffic datasets:
                * **Model Development Data (`model_development_data`):** Ground truth observations used to train our predictive model.* 
                * **Prediction Data (`prediction_data`):** The 2025-2026 dataset representing the target period for our analysis and eventual model-generated projections.*
                
                *Use the sidebars within each section to toggle between these datasets and explore their various patterns.*
            """)
        ),
        
        ui.hr(),
        ui.h3("Section 1: Traffic Overview"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    ### Data Diagnostic
                    This section establishes the baseline behavior of the selected dataset.
        
                    Use these metrics to assess:
                    * **Baseline Overview:** Key metrics covering dataset scale, site coverage, traffic intensity, and directional split (Panel 1).
                    * **Traffic Distribution:** The typical range of cyclist activity, excluding extreme outliers (Panel 2).
                    * **Intensity Scaling:** The distribution of positive traffic counts using a logarithmic scale to identify typical vs. peak activity (Panel 3).
        
                    *These baseline patterns provide the reference point for detecting deviations in subsequent sections.*
                """),
                ui.markdown("---"),
                ui.input_select("s1_dataset", "Choose Dataset:", 
                                choices={
                                    "model_development_data": "Model Development (df1)", 
                                    "prediction_data": "Prediction Data (2025-2026)"
                                }),
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Overview", 
                    ui.output_ui("value_box_EDA"),
                    ui.h5("Directional Split"),
                    ui.output_plot("directional_split_plot")),
                ui.nav_panel("Panel 2: Histogram", ui.markdown("Histogram (99th percentile).")),
                ui.nav_panel("Panel 3: Log Counts", ui.markdown("Log-scale distribution."))
            )
        ),
        ui.hr(),
        ui.h3("Section 2: Temporal Rhythm"),
        ui.layout_sidebar(
            ui.sidebar(ui.input_select("s2_dataset", "Choose Dataset:", 
                                      choices={
                                          "model_development_data": "Model Development (df1)", 
                                          "prediction_data": "Prediction Data (2025-2026)"
                                      })),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Distributions", ui.markdown("Hour/Day/Month balance.")),
                ui.nav_panel("Panel 2: Heatmaps", ui.markdown("Dynamic heatmap."))
            )
        )
    ),
    
    # --- DEVIATIONS SITE ---
    ui.nav_panel("Deviations Detecting",
        ui.h3("Section 1: Deviations Overview"),
        ui.layout_sidebar(
            ui.sidebar(ui.markdown("Threshold: Abs > 25, Rel > 0.75")),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Summary", ui.markdown("Value boxes (Total Obs, Deviations).")),
                ui.nav_panel("Panel 2: Deviation Size", ui.markdown("Deviation size histogram."))
            )
        ),
        ui.hr(),
        ui.h3("Section 3: Weather Impact"),
        ui.layout_sidebar(
            ui.sidebar(ui.markdown("How weather causes model failure.")),
            ui.navset_card_tab(
                ui.nav_panel("Temperature", ui.markdown("Temperature bin deviation chart.")),
                ui.nav_panel("Precipitation", ui.markdown("Precipitation stacked chart."))
            )
        )
    ),
    title="Cycling Traffic Analysis Dashboard"
)

# 2. SERVER LOGIC
def server(input, output, session):
    @reactive.Calc
    def selected_eda_data():
        if input.s1_dataset() == "model_development_data":
            return model_development_data
        else:
            return prediction_data
    
    @render.ui
    def value_box_EDA():
        data = selected_eda_data()
        data['date'] = pd.to_datetime(data['date'])
        start_date = data['date'].min().strftime('%b %Y')
        end_date = data['date'].max().strftime('%b %Y')
        
        return ui.layout_column_wrap(
            ui.value_box("Records", f"{len(data):,}"),
            ui.value_box("Sites", f"{data['site_id'].nunique()}"),
            ui.value_box("Zero Share", f"{(data['count'] == 0).mean()*100:.1f}%"),
            ui.value_box("Date Range", f"{start_date} - {end_date}"),
            ui.value_box("Variables", f"{len(data.columns)}"),
            ui.value_box("Time Bins", f"{data['hour_bin'].nunique()}"),
            width=1/3,
        )
    
    @render.plot
    def directional_split_plot():
        data = selected_eda_data()
        split = data['direction'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(['Direction'], [split.get('In', 50)], color='#66b3ff', label='In')
        ax.barh(['Direction'], [split.get('Out', 50)], left=[split.get('In', 50)], color='#99ff99', label='Out')
        ax.set_xlim(0, 100)
        ax.axis('off')
        ax.legend(loc='upper center', ncol=2)
        return fig

# 3. CREATE APP
app = App(app_ui, server)
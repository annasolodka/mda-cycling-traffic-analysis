from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import seaborn as sns
import numpy as np
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go

# Load data
# Load data using the relative path (so it works on the cloud!)
base_path = Path(__file__).parent.parent
processed_folder = base_path / "data" / "processed"

model_development_data = pd.read_parquet(processed_folder / "model_development_data.parquet")
prediction_data = pd.read_parquet(processed_folder / "expected_counts.parquet")

# Load your expected datasets here or you can also use same parquet file. (Because the csv is too large for shinyapps.io)

# 1. UI DEFINITION
app_ui = ui.page_navbar(
    # EDA SITE
    ui.nav_panel("Exploratory Data Analysis",
        ui.card(
            ui.markdown("""
                This EDA provides a comparative analysis of two primary cycling traffic datasets:
                * **Model Development Data (`model_development_data`):** Ground truth observations used to train our predictive model.
                * **Prediction Data (`prediction_data`):** The 2025-2026 dataset representing the target period for our analysis and eventual model-generated projections.
            """)
        ),
        
        ui.hr(),
        ui.h3("Section 1: Traffic Overview"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section establishes the baseline behavior of the selected dataset.
        
                    Use these metrics to assess:
                    * **Baseline Overview:** Key metrics covering dataset scale, site coverage, traffic intensity, and directional split (Panel 1).
                    * **Traffic Distribution:** The typical range of cyclist activity (Panel 2).
                    * **Intensity Scaling:** The distribution of positive traffic counts using a logarithmic scale to identify typical vs. peak activity (Panel 3).
                """),
                ui.input_select("s1_dataset", "Choose Dataset:", 
                                choices={
                                    "model_development_data": "Model Development Data", 
                                    "prediction_data": "Prediction Data"
                                }),
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Baseline Metrics", 
                    ui.output_ui("value_box_EDA"),
                    ui.markdown("---"),
                    ui.h5("Directional Split"),
                    ui.output_plot("directional_split_plot", height="100px"),
                ),
                ui.nav_panel("Panel 2: Traffic Distribution", 
                    ui.output_plot("traffic_distribution_plot")
                ),
                ui.nav_panel("Panel 3: Intensity Scaling", 
                    ui.output_plot("log_count_distribution_plot")
                )
            )
        ),
        ui.hr(),
        ui.h3("Section 2: Temporal Rhythm"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores the time-based patterns and behavioral rhythms of cycling traffic.
                    
                    **What to explore:**
                    * **Temporal Balance:** The distribution of traffic observations across hours, days, and months to check for uniform data coverage (Panel 1).
                    * **Traffic Heatmaps:** Density variations of cycling volume across different times of day, days of the week, and travel directions (Panel 2).
                """),
                ui.input_select("s2_dataset", "Choose Dataset:", 
                                choices={
                                    "model_development_data": "Model Development Data", 
                                    "prediction_data": "Prediction Data"
                                }),
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Temporal Balance", 
                    ui.input_radio_buttons(
                        "s2_time_var", 
                        "Select Time Scale:", 
                        choices={"hour_bin": "Hour bin", "weekday": "Day", "month": "Month"}, 
                        inline=True
                    ),
                    output_widget("temporal_dist_plot")
                ),
                ui.nav_panel("Panel 2: Traffic Heatmaps", 
                    ui.input_select(
                        "s2_heatmap_var", 
                        "Select Heatmap View:", 
                        choices={"weekday": "Hour × Day", "direction": "Hour × Direction"},
                        width="300px"
                    ),
                    ui.output_plot("temporal_heatmap_plot")
                )
            )
        ),
        ui.hr(),
        ui.h3("Section 3: Economics & Environmental Factors (Fuel Price)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores the temporal distribution and historical trends of fuel prices (Petrol 95).
                    
                    * **Time Series:** Displays the chronological variation of fuel prices across the model development period.
                    * **Daily Distribution:** Shows the frequency of different price points at a daily aggregate level.
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Time Series", 
                    ui.markdown("*Note: Displaying historical trends from the Model Development dataset.*"),
                    ui.output_plot("fuel_time_series_plot")
                ),
                ui.nav_panel("Panel 2: Daily Distribution", 
                    ui.input_select(
                        "s3_dataset", "Choose Dataset:", 
                        choices={"model_development_data": "Model Development Data", "prediction_data": "Prediction Data"},
                        width="300px"
                    ),
                    ui.output_plot("fuel_distribution_plot")
                )
            )
        ),
        ui.hr(),
        ui.h3("Section 4: Weather Impact (2025-2026)"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section isolates the 2025-2026 prediction dataset to explore how meteorological conditions impact cycling volumes.
                    
                    * **Dual-Axis View:** Compares the average traffic intensity (line) against the raw volume of observations (bars) for each condition.
                    * **Data Volume:** The gray bars indicate how much confidence we have in the average. Low bars mean rare weather events.
                """)
            ),
            ui.card(
                ui.input_select(
                    "s4_weather_var", "Select Weather Variable:", 
                    choices={
                        "temperature_mean": "Temperature", 
                        "precipitation_category": "Precipitation",
                        "wind_speed_mean": "Wind Speed"
                    },
                    width="300px"
                ),
                ui.output_plot("weather_impact_plot")
            )
        ),
        ui.hr(),
        ui.h3("Section 5: Special Events Impact"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section measures how special events and holidays alter typical cycling volumes.
                    
                    * **Baseline:** A "Normal Day" where no events, holidays, or strikes are occurring.
                    * **Relative Impact:** The percentage increase or decrease in traffic compared to the baseline.
                    * **Low Confidence:** Events representing less than 1% of the dataset are ghosted/striped, as their averages are highly sensitive to outliers.
                """)
            ),
            ui.card(
                ui.h5("Impact Summary Table"),
                ui.output_data_frame("events_frequency_table"),
                ui.hr(),
                output_widget("events_impact_plot")
            )
        )

    ),
    
    # DEVIATIONS SITE
    ui.nav_panel("Deviation Analaysis",
        ui.h3("Section 1: Deviations Overview and Sizing"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section provides an overview of detected traffic deviations.

                    A deviation represents a mismatch between:
                    * **Observed cyclist traffic**
                    * **Model-expected cyclist traffic**

                    Deviations are identified only when:
                    * the historical reference group has enough observations
                    * the absolute difference is sufficiently large
                    * the relative difference exceeds the predefined threshold

                    **What to explore:**
                    * Overall frequency of deviations
                    * Balance between higher- and lower-than-expected traffic
                    * Distribution of deviation magnitudes
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Summary Metrics",                     ui.output_ui("dev_value_boxes")
                ),
                ui.nav_panel("Panel 2: Deviation Size", 

                    ui.output_plot("dev_size_distribution_plot")
                )
            )
        ),
        
        ui.hr(),
        
        ui.h3("Section 2: Temporal Patterns"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section explores when deviations occur most frequently. It identifies recurring temporal disruption patterns such as rush-hour anomalies or weekend irregularities.

                    The heatmap highlights concentrations of abnormal traffic activity
                    across:
                    * days of the week
                    * hours of the day
                """)
            ),
            ui.card(
                ui.output_plot("dev_temporal_heatmap_plot")
            )
        ),
        
        ui.hr(),
        
        ui.h3("Section 3: Spatial Patterns"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section investigates where deviations occur across Flanders.

                    **What to explore:**
                    * Geographic clustering of deviations
                    * Sites with the highest abnormal activity
                    * Municipalities contributing most strongly to deviation frequency

                    Larger and darker markers indicate locations with higher deviation intensity.
                """)
            ),
            ui.navset_card_tab(
                ui.nav_panel("Panel 1: Spatial Maps", 
                    ui.input_select(
                        "dev_map_metric",
                        "Select map metric:",
                        choices={
                            "deviation_share": "Total deviation share",
                            "higher_share": "Higher than expected share",
                            "lower_share": "Lower than expected share"
                        },
                        width="300px"
                    ),

                    output_widget("dev_spatial_map_plot")
                ),
                ui.nav_panel("Panel 2: Top 25 Sites", 
                    ui.output_plot("dev_top_sites_plot")
                ),
                ui.nav_panel("Panel 3: Site Characterization",
                    ui.navset_card_tab(
                        ui.nav_panel(
                            "Direction Profile",
                            output_widget("dev_direction_profile_map")
                        ),
                        ui.nav_panel(
                            "Sensitivity",
                            output_widget("dev_sensitivity_map")
                        )
                    )
                ),
            )
        ),
        
        ui.hr(),
        
        ui.h3("Section 4: Weather Impact"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section evaluates whether weather conditions are associated
                    with abnormal cycling behavior.

                    The analysis compares:
                    * the share of higher-than-expected traffic deviations
                    * the share of lower-than-expected traffic deviations
                    * the total number of detected deviations

                    This helps identify weather conditions linked to unusual traffic patterns.
                """)
            ),

            ui.card(

                ui.input_select(
                    "dev_weather_var",
                    "Select Weather Variable:",
                    choices={
                        "temperature_mean": "Temperature",
                        "precipitation_category": "Precipitation",
                        "wind_speed_mean": "Wind Speed"
                    },
                    width="300px"
                ),

                ui.output_plot("dev_weather_impact_plot")
            )
        ),
        
        ui.hr(),
        
        ui.h3("Section 5: Special Events Impact"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.markdown("""
                    This section measures how holidays, strikes, and public events
                    influence deviation frequency.

                    Relative impacts are calculated against baseline periods without
                    special events.

                    Positive values indicate an increased likelihood of abnormal traffic behavior.
                """)
            ),
            ui.card(
                output_widget("dev_special_events_plot")
            )
        )
    ),
    title="Cycling Traffic Analysis Dashboard"
)

# 2. SERVER LOGIC
def server(input, output, session):
    # EDA - Section 1
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
        
        # Default to 0 if a direction is completely missing in a specific dataset
        in_pct = split.get('IN', 0)
        out_pct = split.get('OUT', 0)
        
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.barh([''], [in_pct], color='#66b3ff', label=f'In ({in_pct:.1f}%)')
        ax.barh([''], [out_pct], left=[in_pct], color='#99ff99', label=f'Out ({out_pct:.1f}%)')
        ax.set_xlim(0, 100)
        ax.axis('off')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        fig.tight_layout()
        
        return fig
    
    @render.plot
    def traffic_distribution_plot():
        data = selected_eda_data()
        upper_limit = data['count_rescaled'].quantile(0.99)
        filtered_data = data.loc[data['count_rescaled'] <= upper_limit, 'count_rescaled']
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(filtered_data, bins=50, color='#72B7B2', edgecolor='white', linewidth=0.5)
        ax.set_title("Distribution of Rescaled Bicycle Counts (below 99th percentile)", pad=15)
        ax.set_xlabel("2-hour cyclist count")
        ax.set_ylabel("Frequency")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    @render.plot
    def log_count_distribution_plot():
        data = selected_eda_data()
        positive_counts = data[data['count_rescaled'] > 0]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            data=positive_counts,
            x='count_rescaled',
            bins=30,
            kde=True,
            color="#F58518",
            log_scale=True,
            ax=ax 
        )
        ax.set_title("Distribution of Positive Rescaled Bicycle Counts (log scale)", pad=15)
        ax.set_xlabel("Cyclist count (log scale)")
        ax.set_ylabel("Frequency")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 2
    @reactive.Calc
    def selected_temporal_data():
        if input.s2_dataset() == "model_development_data":
            return model_development_data
        else:
            return prediction_data
        
    @render_widget
    def temporal_dist_plot():
        data = selected_temporal_data()
        var_name = input.s2_time_var()
        
        # 1. Define order and create a custom display name
        if var_name == "hour_bin":
            order = list(sorted(data["hour_bin"].dropna().unique()))
            display_name = "Hour bin"
        elif var_name == "weekday":
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            display_name = "Day"
        else: 
            order = list(range(1, 13))
            display_name = "Month"
            
        # 2. Calculate shares
        plot_df = (
            data[var_name]
            .value_counts(dropna=False)
            .reindex(order)
            .rename_axis("level")
            .reset_index(name="n")
        )
        plot_df["share"] = plot_df["n"] / plot_df["n"].sum()
        plot_df["share_pct"] = (plot_df["share"] * 100).round(1).astype(str) + "%"

        # 3. For hours, convert numeric bins to readable time intervals
        if var_name == "hour_bin":
            plot_df["level"] = plot_df["level"].apply(
                lambda x: f"{int(x):02d}:00 - {int(x)+1:02d}:59" if pd.notnull(x) else x
            )

        # 4. Build the chart
        fig = px.bar(
            plot_df, 
            x="level", 
            y="share", 
            labels={"level": display_name, "share": "Share of Traffic"},
            title=f"Distribution of Traffic by {display_name}"
        )

        # 5. Hover Interactivity
        fig.update_traces(
            marker_color="#4C78A8",
            hovertemplate="<b>%{x}</b><br>Share: %{customdata[1]}<br>Observations: %{customdata[0]:,}<extra></extra>",
            customdata=plot_df[["n", "share_pct"]]
        )
        
        fig.update_layout(
            yaxis_tickformat='.0%', 
            template="plotly_white",
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        fig.update_yaxes(range=[0, plot_df["share"].max() * 1.15])

        return fig
    
    @render.plot
    def temporal_heatmap_plot():
        data = selected_temporal_data()
        view = input.s2_heatmap_var()
        
        if view == "weekday":
            heatmap_data = (
                data.groupby(["weekday", "hour_bin"], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            
            heatmap_data = heatmap_data.reindex(
                index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))
            
            fig, ax = plt.subplots(figsize=(10, 4.8))
            sns.heatmap(heatmap_data, cmap="Greens", cbar_kws={"label": "Count"}, ax=ax)
            ax.set_title("Hour × Day Count Heatmap", pad=15)
            ax.set_xlabel("Hour bin")
            ax.set_ylabel("Day")
            
        else:
            col_name = 'direction'
            
            heatmap_data = (
                data.groupby([col_name, "hour_bin"], observed=False)
                .size()
                .unstack(fill_value=0)
            )
            heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))
            
            fig, ax = plt.subplots(figsize=(10, 3.6))
            sns.heatmap(heatmap_data, cmap="Greens", cbar_kws={"label": "Count"}, ax=ax)
            ax.set_title("Hour × Direction Count Heatmap", pad=15)
            ax.set_xlabel("Hour bin")
            ax.set_ylabel("Direction")
            
        fig.tight_layout()
        return fig
    
    # EDA - Section 3
    @render.plot
    def fuel_time_series_plot():
        df = model_development_data.copy()
        
        # 1. Data prep: Get daily level and drop missing values
        fuel_daily = (df[["date", "fuel_price_petrol_95"]].drop_duplicates().dropna().sort_values("date").reset_index(drop=True))
        
        fuel_daily["date"] = pd.to_datetime(fuel_daily["date"])
        
        # 2. Create the time series plot
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=fuel_daily, x="date", y="fuel_price_petrol_95", color="#F58518", ax=ax)
        
        ax.set_title("Fuel Price over Time (Model Development)", pad=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Fuel price petrol 95")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig


    @render.plot
    def fuel_distribution_plot():
        if input.s3_dataset() == "model_development_data":
            df = model_development_data.copy()
        else:
            df = prediction_data.copy()
            
        # 1. Data prep: Get daily level and drop missing values
        fuel_daily_nonmissing = (df[["date", "fuel_price_petrol_95"]].drop_duplicates().dropna(subset=["fuel_price_petrol_95"]))
        
        # 2. Create the distribution plot
        fig, ax = plt.subplots(figsize=(9, 5))
        
        sns.histplot(
            fuel_daily_nonmissing["fuel_price_petrol_95"],
            bins=20,
            kde=True,
            color="#4C78A8",
            ax=ax
        )
        
        ax.set_title("Distribution of Fuel Price (daily level)", pad=15)
        ax.set_xlabel("Fuel price petrol 95")
        ax.set_ylabel("Frequency")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 4
    @render.plot
    def weather_impact_plot():
        df = prediction_data.copy()
        var = input.s4_weather_var()
        
        # 1. Prepare data
        if var == "temperature_mean":
            df['plot_var'] = df['temperature_mean'].round()
            x_label = "Temperature (°C)"
            title_prefix = "Temperature"
        elif var == "wind_speed_mean":
            col_name = 'wind_speed_mean' if 'wind_speed_mean' in df.columns else 'wind_spead_mean'
            df['plot_var'] = df[col_name].round()
            x_label = "Wind Speed"
            title_prefix = "Wind Speed"
        else: 
            df['plot_var'] = df['precipitation_category'].str.replace('_precipitation', '').str.title()
            x_label = "Weather Condition"
            title_prefix = "Precipitation"
            
        # 2. GroupBy calculation
        summary = df.groupby('plot_var').agg(
            num_obs=('count_rescaled', 'count'),
            avg_traffic=('count_rescaled', 'mean')
        ).reset_index()
        
        # 3. Create dual-axis plot
        fig, ax1 = plt.subplots(figsize=(11, 9))
        ax2 = ax1.twinx()
        
        # Categorical/Numeric handling: Precipitation is categorical, Temp and Wind are numeric. This ensures proper X-axis formatting and spacing.
        if var == "precipitation_category":
            order = ['Dry', 'Light', 'Moderate', 'Heavy', 'Snow']
            summary['plot_var'] = pd.Categorical(summary['plot_var'], categories=[p for p in order if p in summary['plot_var'].values], ordered=True)
            summary = summary.sort_values('plot_var')
            x_data = summary['plot_var'].astype(str)
            ax1.bar(x_data, summary['num_obs'], color='lightgray', alpha=0.6)
            ax2.plot(x_data, summary['avg_traffic'], color='#F58518', linewidth=3, marker='o', markersize=8)
            
        else:
            x_data = pd.to_numeric(summary['plot_var'])
            ax1.bar(x_data, summary['num_obs'], color='lightgray', alpha=0.6, width=0.8)
            ax2.plot(x_data, summary['avg_traffic'], color='#F58518', linewidth=3, marker='o', markersize=8)

        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel("Number of Observations (2-hour intervals)", color='dimgray', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='dimgray')
        
        ax2.set_ylabel("Average Cyclist Count", color='#F58518', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#F58518')
        ax2.set_ylim(bottom=0)
        
        plt.title(f"{title_prefix} Impact: Cyclist Traffic vs. Data Volume", pad=15, fontsize=14, fontweight='bold')
        
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax2.grid(False)
        
        fig.tight_layout()
        return fig
    
    # EDA - Section 5
    @reactive.calc
    def special_events_data():
        df = prediction_data.copy()
        
        event_cols = {
            'is_public_holiday': 'Public Holiday',
            'is_school_holiday': 'School Holiday',
            'is_sport_event': 'Sports Event',
            'is_outdoor_music': 'Outdoor Music',
            'is_indoor_music': 'Indoor Music',
            'is_strike': 'Transport Strike'
        }
        
        # Recheck
        valid_cols = {k: v for k, v in event_cols.items() if k in df.columns}
        
        # Calculate Baseline (Normal Day)
        baseline_mask = ~df[list(valid_cols.keys())].any(axis=1)
        normal_obs = baseline_mask.sum()
        baseline_avg = df[baseline_mask]['count_rescaled'].mean()
        total_obs = len(df)
        
        results = []
        
        # 1. Add Normal Day to our results
        results.append({
            'Condition': 'Normal Day',
            'Total 2hr Intervals': normal_obs,
            '% of Total Year': (normal_obs / total_obs) * 100,
            'Impact (%)': 0.0, 
            'Confidence': 'High'
        })
        
        # 2. Add all other events
        for col, name in valid_cols.items():
            obs = df[col].sum()
            
            if obs > 0:
                avg = df[df[col] == 1]['count_rescaled'].mean()
                lift = ((avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
            else:
                avg, lift = 0, 0
                
            pct_year = (obs / total_obs) * 100
            
            results.append({
                'Condition': name,
                'Total 2hr Intervals': obs,
                '% of Total Year': pct_year,
                'Impact (%)': lift,
                'Confidence': 'Low (<1%)' if pct_year < 1.0 else 'High'
            })
            
        return pd.DataFrame(results)

    @render.data_frame
    def events_frequency_table():
        df = special_events_data()
        table_df = df[['Condition', 'Total 2hr Intervals', '% of Total Year']].copy()
        table_df['% of Total Year'] = table_df['% of Total Year'].map("{:.2f}%".format)
        
        # Sort from most frequent to least frequent
        table_df = table_df.sort_values('Total 2hr Intervals', ascending=False)
        
        return render.DataGrid(table_df, selection_mode="none", width="100%")

    @render_widget
    def events_impact_plot():
        df = special_events_data()
        
        # Filter out "Normal Day" for the chart, since it's just the 0% baseline
        plot_df = df[df['Condition'] != 'Normal Day'].copy()
        plot_df = plot_df.sort_values('Impact (%)')
        
        # Conditional styling based on confidence and impact
        colors = ['crimson' if val < 0 else 'lightseagreen' for val in plot_df['Impact (%)']]
        opacities = [0.4 if conf == 'Low (<1%)' else 1.0 for conf in plot_df['Confidence']]
        patterns = ['/' if conf == 'Low (<1%)' else '' for conf in plot_df['Confidence']]
        
        fig = go.Figure(go.Bar(
            x=plot_df['Impact (%)'],
            y=plot_df['Condition'],
            orientation='h',
            marker=dict(
                color=colors,
                opacity=opacities,
                pattern=dict(shape=patterns),
                line=dict(color='gray', width=[1 if p == '/' else 0 for p in patterns])
            ),
            hovertext=[f"{val:+.2f}%" for val in plot_df['Impact (%)']],
            customdata=plot_df[['Total 2hr Intervals', '% of Total Year']],
            hovertemplate="<b>%{y}</b><br>Impact vs Normal: %{hovertext}<br>Observations: %{customdata[0]} (%{customdata[1]:.2f}% of data)<extra></extra>"
        ))
        
        # Add the vertical baseline
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title="Relative Impact of Special Events vs. Normal Day",
            xaxis_title="Percentage Change vs. Normal Day (%)",
            yaxis_title="",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            annotations=[dict(
                x=0.5, y=-0.2, xref='paper', yref='paper',
                text="* Striped/ghosted bars indicate low statistical confidence (< 1% of total data).",
                showarrow=False, font=dict(color="gray", size=11)
            )]
        )
        
        return fig

    ###################################################################################

    # Deviations - Section 1
    @reactive.Calc
    def deviation_data():

        reference_variables = [
            "site_id",
            "direction",
            "month",
            "weekday",
            "hour_bin",
        ]

        reference_counts = (
            model_development_data
            .groupby(reference_variables)
            .size()
            .reset_index(name="reference_n")
        )

        df = prediction_data.merge(
            reference_counts,
            on=reference_variables,
            how="left"
        )

        df["reference_n"] = df["reference_n"].fillna(0).astype(int)

        df["difference"] = (
            df["count_rescaled"] - df["expected_count"]
        )

        df["relative_difference"] = (
            df["difference"] / df["expected_count"]
        )

        df["is_deviation"] = (
            (df["reference_n"] >= 10) &
            (df["difference"].abs() > 25) &
            (df["relative_difference"].abs() > 0.75)
        ).astype(int)

        df["deviation_direction"] = "No deviation"

        df.loc[
            (df["is_deviation"] == 1) &
            (df["difference"] > 0),
            "deviation_direction"
        ] = "Higher than expected"

        df.loc[
            (df["is_deviation"] == 1) &
            (df["difference"] < 0),
            "deviation_direction"
        ] = "Lower than expected"

        return df
    def summarize_deviations(data, group_variable):
        baseline_share = data["is_deviation"].mean()

        summary = (
            data
            .groupby(group_variable)
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum")
            )
            .reset_index()
        )

        summary["deviation_share"] = summary["deviations"] / summary["observations"]
        summary["baseline_deviation_share"] = baseline_share

        return summary

    def summarize_directional_deviations(data, group_variable):
        baseline_higher_share = (
            data["deviation_direction"] == "Higher than expected"
        ).mean()

        baseline_lower_share = (
            data["deviation_direction"] == "Lower than expected"
        ).mean()

        summary = (
            data
            .groupby(group_variable)
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Higher than expected").sum()
                ),
                lower_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Lower than expected").sum()
                )
            )
            .reset_index()
        )

        summary["deviation_share"] = summary["deviations"] / summary["observations"]
        summary["higher_deviation_share"] = summary["higher_deviations"] / summary["observations"]
        summary["lower_deviation_share"] = summary["lower_deviations"] / summary["observations"]
        summary["baseline_higher_deviation_share"] = baseline_higher_share
        summary["baseline_lower_deviation_share"] = baseline_lower_share

        return summary
    @render.ui
    def dev_value_boxes():

        df = deviation_data()

        total_obs = len(df)

        total_dev = (df["is_deviation"] == 1).sum()

        higher = (
            df["deviation_direction"] == "Higher than expected"
        ).sum()

        lower = (
            df["deviation_direction"] == "Lower than expected"
        ).sum()

        return ui.layout_column_wrap(
            ui.value_box(
                "Total Traffic Records",
                f"{total_obs:,}"
            ),

            ui.value_box(
                "Detected Deviations",
                f"{total_dev:,}"
            ),

            ui.value_box(
                "Higher-than-Expected",
                f"{higher:,}"
            ),

            ui.value_box(
                "Lower-than-Expected",
                f"{lower:,}"
            ),

            width=1/4
        )
    @render.plot
    def dev_size_distribution_plot():

        df = deviation_data()

        dev_df = df[df["is_deviation"] == 1]

        lower = dev_df["difference"].quantile(0.01)
        upper = dev_df["difference"].quantile(0.99)

        plot_df = dev_df[
            (dev_df["difference"] >= lower) &
            (dev_df["difference"] <= upper)
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.histplot(
            plot_df["difference"],
            bins=40,
            color="#F58518",
            edgecolor="white",
            ax=ax
        )

        ax.set_title(
            "Distribution of Observed vs. Expected Differences",
            pad=15
        )

        ax.set_xlabel(
            "Observed minus Expected Cyclist Count"
        )

        ax.set_ylabel("Frequency")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig
    
    @render.plot
    def dev_temporal_heatmap_plot():

        df = deviation_data()

        dev_df = df[df["is_deviation"] == 1]

        heatmap_data = (
            dev_df
            .groupby(["weekday", "hour_bin"])
            .size()
            .unstack(fill_value=0)
        )

        heatmap_data = heatmap_data.reindex(
            index=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        heatmap_data = heatmap_data.reindex(
            columns=sorted(heatmap_data.columns)
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.heatmap(
            heatmap_data,
            cmap="Reds",
            cbar_kws={"label": "Deviation Count"},
            ax=ax
        )

        ax.set_title(
            "Deviation Frequency by Weekday and Hour",
            pad=15
        )

        ax.set_xlabel("Hour bin")
        ax.set_ylabel("Weekday")

        fig.tight_layout()

        return fig
    
    @render.plot
    def dev_weather_impact_plot():

        df = deviation_data()

        dev_df = df[df["is_deviation"] == 1].copy()

        var = input.dev_weather_var()

        if var == "temperature_mean":

            dev_df["plot_var"] = (
                dev_df["temperature_mean"].round()
            )

            x_label = "Temperature (°C)"
            title_prefix = "Temperature"

        elif var == "wind_speed_mean":

            dev_df["plot_var"] = (
                dev_df["wind_speed_mean"].round()
            )

            x_label = "Wind Speed"
            title_prefix = "Wind Speed"

        else:

            dev_df["plot_var"] = (
                dev_df["precipitation_category"]
                .str.replace("_precipitation", "")
                .str.title()
            )

            x_label = "Precipitation"
            title_prefix = "Precipitation"

        summary = (
            dev_df
            .groupby("plot_var")
            .agg(
                observations=("is_deviation", "count"),
                higher_rate=(
                    "deviation_direction",
                    lambda x: (
                        x == "Higher than expected"
                    ).mean()
                ),
                lower_rate=(
                    "deviation_direction",
                    lambda x: (
                        x == "Lower than expected"
                    ).mean()
                )
            )
            .reset_index()
        )

        summary["Higher (%)"] = (
            summary["higher_rate"] * 100
        )

        summary["Lower (%)"] = (
            summary["lower_rate"] * 100
        )

        summary["Total (%)"] = (
            summary["Higher (%)"] +
            summary["Lower (%)"]
        )

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.bar(
            summary["plot_var"],
            summary["Higher (%)"],
            color="crimson",
            alpha=0.85,
            label="Higher than expected"
        )

        ax1.bar(
            summary["plot_var"],
            summary["Lower (%)"],
            bottom=summary["Higher (%)"],
            color="darkorange",
            alpha=0.85,
            label="Lower than expected"
        )

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(
            "Share of Deviations (%)",
            fontweight="bold"
        )

        ax2 = ax1.twinx()

        ax2.plot(
            summary["plot_var"],
            summary["observations"],
            color="dimgray",
            linewidth=3,
            marker="o",
            linestyle="--",
            label="Observations"
        )

        ax2.set_ylabel(
            "Detected Deviations",
            color="dimgray",
            fontweight="bold"
        )

        ax1.set_title(
            f"Weather Conditions Associated with Traffic Deviations ({title_prefix})",
            pad=15,
            fontweight="bold"
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left"
        )

        fig.tight_layout()

        return fig
    @render.plot
    def dev_top_sites_plot():

        df = deviation_data()

        site_deviation_summary = (
            df
            .groupby(["site_id", "site_name", "municipality", "latitude", "longitude"])
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Higher than expected").sum()
                ),
                lower_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Lower than expected").sum()
                ),
            )
            .reset_index()
        )

        site_deviation_summary["deviation_share"] = (
            site_deviation_summary["deviations"] /
            site_deviation_summary["observations"]
        )

        top_sites = (
            site_deviation_summary
            .sort_values("deviation_share", ascending=False)
            .head(25)
            .copy()
        )

        top_sites["site_label"] = (
            top_sites["site_id"].astype(str)
            + " - "
            + top_sites["site_name"].astype(str)
            + " ("
            + top_sites["municipality"].astype(str)
            + ")"
        )

        top_sites["rank"] = range(1, len(top_sites) + 1)

        top_sites["bar_color"] = np.where(
            top_sites["rank"] <= 5,
            "#b2182b",
            "#bdbdbd"
        )

        top_sites = top_sites.sort_values("deviation_share", ascending=True)

        fig, ax = plt.subplots(figsize=(11, 9))

        fig.patch.set_facecolor("#f2f2f2")
        ax.set_facecolor("#f2f2f2")

        ax.barh(
            top_sites["site_label"],
            top_sites["deviation_share"],
            color=top_sites["bar_color"],
        )

        ax.set_xlabel("Deviation share")
        ax.set_ylabel("Site")
        ax.set_title("Top 25 sites by deviation share")

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(axis="x", color="white", linewidth=1.2)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig
    
    @render_widget
    def dev_special_events_plot():

        df = deviation_data()

        event_cols = {
            'is_public_holiday': 'Public Holiday',
            'is_school_holiday': 'School Holiday',
            'is_sport_event': 'Sports Event',
            'is_outdoor_music': 'Outdoor Music',
            'is_indoor_music': 'Indoor Music',
            'is_strike': 'Transport Strike'
        }

        valid_cols = {
            k: v for k, v in event_cols.items()
            if k in df.columns
        }

        results = []

        baseline_mask = (
            ~df[list(valid_cols.keys())]
            .any(axis=1)
        )

        baseline_rate = (
            df.loc[baseline_mask, "is_deviation"]
            .mean()
        )

        total_obs = len(df)

        for col, label in valid_cols.items():

            subset = df[df[col] == 1]

            if len(subset) == 0:
                continue

            dev_rate = subset["is_deviation"].mean()

            relative_impact = (
                (dev_rate - baseline_rate)
                / baseline_rate
            ) * 100

            pct_dataset = (
                len(subset) / total_obs
            ) * 100

            results.append({
                "Event": label,
                "Deviation Impact (%)": relative_impact,
                "Observations": len(subset),
                "Dataset Share": pct_dataset
            })

        plot_df = pd.DataFrame(results)

        plot_df = plot_df.sort_values(
            "Deviation Impact (%)"
        )

        colors = [
            "crimson" if x < 0
            else "lightseagreen"
            for x in plot_df["Deviation Impact (%)"]
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df["Deviation Impact (%)"],
            y=plot_df["Event"],
            orientation="h",
            marker_color=colors,
            customdata=plot_df[
                ["Observations", "Dataset Share"]
            ],
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Relative impact: %{x:.2f}%<br>" +
                "Observations: %{customdata[0]}<br>" +
                "Dataset share: %{customdata[1]:.2f}%<extra></extra>"
            )
        ))

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )

        fig.update_layout(
            title="Impact of Special Events on Traffic Deviations",
            template="plotly_white",
            height=500,
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=40
            ),
            xaxis_title="Relative Change in Deviation Rate (%)",
            yaxis_title=""
        )

        return fig
    
    @render_widget
    def dev_spatial_map_plot():

        df = deviation_data()

        site_map = (
            df.groupby(
                [
                    "site_id",
                    "site_name",
                    "municipality",
                    "latitude",
                    "longitude"
                ],
                dropna=False
            )
            .agg(
                total_observations=("is_deviation", "count"),
                total_deviations=("is_deviation", "sum"),
                higher_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Higher than expected").sum()
                ),
                lower_deviations=(
                    "deviation_direction",
                    lambda x: (x == "Lower than expected").sum()
                )
            )
            .reset_index()
        )

        site_map = site_map.dropna(
            subset=["latitude", "longitude"]
        )

        site_map["deviation_share"] = (
            site_map["total_deviations"] /
            site_map["total_observations"] * 100
        )

        site_map["higher_share"] = (
            site_map["higher_deviations"] /
            site_map["total_observations"] * 100
        )

        site_map["lower_share"] = (
            site_map["lower_deviations"] /
            site_map["total_observations"] * 100
        )

        metric = input.dev_map_metric()

        metric_labels = {
            "deviation_share": "Total deviation share (%)",
            "higher_share": "Higher than expected share (%)",
            "lower_share": "Lower than expected share (%)"
        }

        fig = px.scatter_mapbox(
            site_map,
            lat="latitude",
            lon="longitude",
            color=metric,
            size="total_deviations",
            hover_name="site_name",
            hover_data={
                "municipality": True,
                "total_observations": True,
                "total_deviations": True,
                "higher_deviations": True,
                "lower_deviations": True,
                "latitude": False,
                "longitude": False,
                metric: ":.2f"
            },
            color_continuous_scale="Reds",
            zoom=7,
            height=700,
            title=metric_labels[metric]
        )

        fig.update_layout(
            mapbox_style="open-street-map",
            template="plotly_white",
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_colorbar=dict(
                    title="Deviation Share (%)"),
        )

        return fig
    @reactive.Calc
    def site_characterisation_data():
        df = deviation_data().copy()

        site_characterisation = (
            df.groupby(["site_id", "site_name", "municipality", "latitude", "longitude"])
            .agg(
                observations=("is_deviation", "size"),
                deviations=("is_deviation", "sum"),
                higher_deviations=("deviation_direction", lambda x: (x == "Higher than expected").sum()),
                lower_deviations=("deviation_direction", lambda x: (x == "Lower than expected").sum()),
            )
            .reset_index()
        )

        site_characterisation["deviation_share"] = site_characterisation["deviations"] / site_characterisation["observations"]
        site_characterisation["higher_deviation_share"] = site_characterisation["higher_deviations"] / site_characterisation["observations"]
        site_characterisation["lower_deviation_share"] = site_characterisation["lower_deviations"] / site_characterisation["observations"]

        site_characterisation["direction_profile"] = "Low deviation frequency"
        high = site_characterisation["deviation_share"] >= 0.10

        site_characterisation.loc[
            high & (site_characterisation["higher_deviation_share"] > 1.5 * site_characterisation["lower_deviation_share"]),
            "direction_profile"
        ] = "Mostly higher counts than expected"

        site_characterisation.loc[
            high & (site_characterisation["lower_deviation_share"] > 1.5 * site_characterisation["higher_deviation_share"]),
            "direction_profile"
        ] = "Mostly lower counts than expected"

        site_characterisation.loc[
            high
            & (site_characterisation["higher_deviation_share"] <= 1.5 * site_characterisation["lower_deviation_share"])
            & (site_characterisation["lower_deviation_share"] <= 1.5 * site_characterisation["higher_deviation_share"]),
            "direction_profile"
        ] = "Mixed direction of deviations"

        df["is_cultural_event"] = ((df["is_outdoor_music"] == 1) | (df["is_indoor_music"] == 1)).astype(int)

        df["meaningful_precipitation"] = df["precipitation_category"].isin([
            "moderate_precipitation",
            "heavy_precipitation",
            "snow",
        ]).astype(int)

        cold_threshold = df["temperature_mean"].quantile(0.05)
        warm_threshold = df["temperature_mean"].quantile(0.95)

        df["cold_weather"] = (df["temperature_mean"] <= cold_threshold).astype(int)
        df["warm_weather"] = (df["temperature_mean"] >= warm_threshold).astype(int)

        df["normal_external_conditions"] = (
            (df["is_strike"] == 0)
            & (df["is_cultural_event"] == 0)
            & (df["is_sport_event"] == 0)
            & (df["meaningful_precipitation"] == 0)
            & (df["cold_weather"] == 0)
            & (df["warm_weather"] == 0)
        ).astype(int)

        normal_site_summary = (
            df[df["normal_external_conditions"] == 1]
            .groupby("site_id")
            .agg(
                normal_observations=("is_deviation", "size"),
                normal_deviations=("is_deviation", "sum"),
            )
            .reset_index()
        )

        normal_site_summary["normal_deviation_share"] = (
            normal_site_summary["normal_deviations"] / normal_site_summary["normal_observations"]
        )

        site_characterisation = site_characterisation.merge(
            normal_site_summary,
            on="site_id",
            how="left"
        )

        site_characterisation["normal_deviation_share"] = site_characterisation["normal_deviation_share"].fillna(0)

        factor_columns = [
            "meaningful_precipitation",
            "cold_weather",
            "warm_weather",
            "is_strike",
            "is_cultural_event",
            "is_sport_event",
        ]

        factor_labels = {
            "meaningful_precipitation": "Precipitation",
            "cold_weather": "Cold weather",
            "warm_weather": "Warm weather",
            "is_strike": "Transport strike",
            "is_cultural_event": "Cultural event",
            "is_sport_event": "Sport event",
        }

        factor_summary_list = []

        for factor_column in factor_columns:
            temp = (
                df[df[factor_column] == 1]
                .groupby("site_id")
                .agg(
                    factor_observations=("is_deviation", "size"),
                    factor_deviations=("is_deviation", "sum"),
                )
                .reset_index()
            )

            temp["factor"] = factor_column
            temp["factor_label"] = factor_labels[factor_column]
            temp["factor_deviation_share"] = temp["factor_deviations"] / temp["factor_observations"]

            factor_summary_list.append(temp)

        site_factor_sensitivity = pd.concat(factor_summary_list, ignore_index=True)

        site_factor_sensitivity = site_factor_sensitivity.merge(
            site_characterisation[
                ["site_id", "site_name", "municipality", "deviation_share", "normal_deviation_share"]
            ],
            on="site_id",
            how="left"
        )

        site_factor_sensitivity["lift_from_normal"] = (
            site_factor_sensitivity["factor_deviation_share"]
            - site_factor_sensitivity["normal_deviation_share"]
        )

        site_factor_sensitivity["is_sensitive"] = (
            (site_factor_sensitivity["deviation_share"] >= 0.10)
            & (site_factor_sensitivity["lift_from_normal"] >= 0.05)
        ).astype(int)

        sensitive_only = site_factor_sensitivity[site_factor_sensitivity["is_sensitive"] == 1].copy()

        number_of_sensitivities = (
            sensitive_only.groupby("site_id")
            .size()
            .reset_index(name="number_of_sensitivities")
        )

        sensitive_factor_list = (
            sensitive_only
            .sort_values(["site_id", "lift_from_normal"], ascending=[True, False])
            .groupby("site_id")["factor_label"]
            .apply(lambda x: ", ".join(x))
            .reset_index(name="sensitive_factors")
        )

        main_sensitivity_factor = (
            sensitive_only
            .sort_values(["site_id", "lift_from_normal"], ascending=[True, False])
            .groupby("site_id")
            .first()
            .reset_index()[["site_id", "factor_label", "lift_from_normal"]]
            .rename(columns={
                "factor_label": "main_sensitivity_factor",
                "lift_from_normal": "main_sensitivity_lift",
            })
        )

        site_sensitivity_summary = (
            number_of_sensitivities
            .merge(sensitive_factor_list, on="site_id", how="left")
            .merge(main_sensitivity_factor, on="site_id", how="left")
        )

        site_characterisation = site_characterisation.merge(
            site_sensitivity_summary,
            on="site_id",
            how="left"
        )

        site_characterisation["number_of_sensitivities"] = site_characterisation["number_of_sensitivities"].fillna(0).astype(int)
        site_characterisation["sensitive_factors"] = site_characterisation["sensitive_factors"].fillna("No sensitivities identified")
        site_characterisation["main_sensitivity_factor"] = site_characterisation["main_sensitivity_factor"].fillna("No main sensitivity factor")
        site_characterisation["main_sensitivity_lift"] = site_characterisation["main_sensitivity_lift"].fillna(0)

        site_characterisation["site_category"] = "Stable"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] == 0),
            "site_category"
        ] = "Unstable independent of factors"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] == 1),
            "site_category"
        ] = "Single-factor-sensitive"

        site_characterisation.loc[
            (site_characterisation["deviation_share"] >= 0.10)
            & (site_characterisation["number_of_sensitivities"] >= 2),
            "site_category"
        ] = "Multiple-factor-sensitive"

        site_characterisation["site_label"] = (
            site_characterisation["site_id"].astype(str)
            + " - "
            + site_characterisation["site_name"].astype(str)
            + " ("
            + site_characterisation["municipality"].astype(str)
            + ")"
        )

        site_characterisation["deviation_share_for_size"] = site_characterisation["deviation_share"] * 100

        return site_characterisation
    
    @render_widget
    def dev_direction_profile_map():
        site_map = site_characterisation_data().dropna(subset=["latitude", "longitude"])

        direction_colors = {
            "Low deviation frequency": "#9e9e9e",
            "Mostly higher counts than expected": "#b2182b",
            "Mostly lower counts than expected": "#fdae61",
            "Mixed direction of deviations": "#7b3294",
        }

        fig = px.scatter_mapbox(
            site_map,
            lat="latitude",
            lon="longitude",
            color="direction_profile",
            size="deviation_share_for_size",
            color_discrete_map=direction_colors,
            size_max=24,
            zoom=7,
            height=700,
            hover_name="site_label",
            hover_data={
                "site_id": True,
                "municipality": True,
                "deviation_share": ":.1%",
                "higher_deviation_share": ":.1%",
                "lower_deviation_share": ":.1%",
                "direction_profile": True,
                "deviation_share_for_size": False,
                "latitude": False,
                "longitude": False,
            },
            title="Direction profile of deviations by counting site",
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            paper_bgcolor="#f2f2f2",
            plot_bgcolor="#f2f2f2",
            font=dict(color="#333333"),
            title=dict(x=0.02),
            legend_title_text="Direction profile",
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
        )

        return fig
    
    @render_widget
    def dev_sensitivity_map():
        site_map = site_characterisation_data().dropna(subset=["latitude", "longitude"])

        site_map["sensitivity_map_group"] = site_map["site_category"]

        site_map.loc[
            site_map["site_category"] == "Single-factor-sensitive",
            "sensitivity_map_group"
        ] = site_map["main_sensitivity_factor"]

        site_map.loc[
            site_map["site_category"] == "Multiple-factor-sensitive",
            "sensitivity_map_group"
        ] = "Multiple factors"

        sensitivity_colors = {
            "Stable": "#9e9e9e",
            "Unstable independent of factors": "#b2182b",
            "Warm weather": "#ef8a62",
            "Cold weather": "#67a9cf",
            "Precipitation": "#9ab5d0",
            "Transport strike": "#fdae61",
            "Cultural event": "#2166ac",
            "Sport event": "#d36c95",
            "Multiple factors": "#6a3d9a",
        }

        sensitivity_order = [
            "Stable",
            "Unstable independent of factors",
            "Precipitation",
            "Cold weather",
            "Warm weather",
            "Transport strike",
            "Cultural event",
            "Sport event",
            "Multiple factors",
        ]

        fig = px.scatter_mapbox(
            site_map,
            lat="latitude",
            lon="longitude",
            color="sensitivity_map_group",
            size="deviation_share_for_size",
            color_discrete_map=sensitivity_colors,
            category_orders={"sensitivity_map_group": sensitivity_order},
            size_max=24,
            zoom=7,
            height=700,
            hover_name="site_label",
            hover_data={
                "site_id": True,
                "municipality": True,
                "deviation_share": ":.1%",
                "normal_deviation_share": ":.1%",
                "site_category": True,
                "main_sensitivity_factor": True,
                "sensitive_factors": True,
                "direction_profile": True,
                "deviation_share_for_size": False,
                "latitude": False,
                "longitude": False,
            },
            title="Site sensitivity to external factors",
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            paper_bgcolor="#f2f2f2",
            plot_bgcolor="#f2f2f2",
            font=dict(color="#333333"),
            title=dict(x=0.02),
            legend_title_text="Sensitivity type",
            margin={"r": 0, "t": 55, "l": 0, "b": 0},
        )

        return fig

# 3. CREATE APP
app = App(app_ui, server)
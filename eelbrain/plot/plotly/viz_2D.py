import base64
import io
import random

import dash
from dash import dcc, html, Input, Output, State
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from eelbrain import set_parc
from typing import Optional, Union, List

# Handle import for both standalone and package usage
try:
    from .._data_obj import NDVar
except ImportError:
    from eelbrain import NDVar


class EelbrainPlotly2DViz:
    """Interactive 2D brain visualization for brain data using Plotly and Dash."""
    # typing
    # show arrows larger than 10 import plotly wen user call function
    # add test 
    # let user to control data showing
    #  data_source_location data format

    def __init__(
            self, 
            y: Optional[NDVar] = None, 
            data_source_location: Optional[str] = None, 
            region: Optional[str] = None, 
            cmap: Union[str, List] = 'Hot', 
            show_max_only: bool = False
    ):
        """Initialize the visualization app and load data.

        Parameters
        ----------
        y
            Data to plot ([case,] time, source[, space]).
            If ``y`` has a case dimension, the mean is plotted.
            If ``y`` has a space dimension, the norm is plotted.
            If None, uses MNE sample data.
        data_source_location
            Path to the data file. If None and y is None, uses MNE sample data.
            Ignored if y is provided.
        region
            Brain region to load using aparc+aseg parcellation.
            If None, loads all regions. Only used when loading from file.
        cmap
            Plotly colorscale for heatmaps. Can be:
            - Built-in colorscale name (e.g., 'Hot', 'Viridis', 'YlOrRd')
            - Custom colorscale list (e.g., [[0, 'yellow'], [1, 'red']])
            Default is 'Hot'.
        show_max_only
            If True, butterfly plot shows only mean and max traces.
            If False, butterfly plot shows individual source traces, mean, and max.
            Default is False.
        """
        self.app = dash.Dash(__name__)

        # Initialize data attributes
        self.glass_brain_data = None  # (n_sources, 3, n_times)
        self.butterfly_data = None    # (n_sources, n_times)
        self.source_coords = None     # (n_sources, 3)
        self.time_values = None       # (n_times,)
        self.region_of_brain = region  # Region of brain to visualize
        self.cmap = cmap  # Colorscale for heatmaps
        self.show_max_only = show_max_only  # Control butterfly plot display mode

        # Load data
        if y is not None:
            self._load_ndvar_data(y)
        else:
            self._load_source_data(data_source_location, region)

        # Setup app
        self._setup_layout()
        self._setup_callbacks()

    def _load_source_data(self, data_source_location=None, region=None):
        """Load MNE sample data and prepare for 2D brain visualization.

        Parameters
        ----------
        data_source_location : str, optional
            Path to the data file. If None, uses MNE sample data.
        region : str, optional
            Brain region to load using aparc+aseg parcellation.
            If None, loads all regions.
        """
        from eelbrain import datasets

        # Load MNE sample data
        if data_source_location is None:
            data_ds = datasets.get_mne_sample(src='vol', ori='vector')
        else:
            data_ds = datasets.load(data_source_location)

        # Set parcellation if region is specified
        if region is not None:
            data_ds['src'] = set_parc(data_ds['src'], region)
            self.region_of_brain = region
        else:
            self.region_of_brain = 'Full Brain'

        # Average over trials/cases
        src_ndvar = data_ds['src'].mean('case')

        # Extract coordinates and data 
        self.glass_brain_data = src_ndvar.get_data(('source','space', 'time'))  # (n_sources, 3, n_times) todo modify the order of data to be (n_sources, n_times, 3)
        self.source_coords = src_ndvar.source.coordinates  # (n_sources, 3)
        self.time_values = src_ndvar.time.times

        # Store source space info
        self.source_space = src_ndvar.source
        if hasattr(self.source_space, 'parc'):
            self.parcellation = self.source_space.parc
        else:
            self.parcellation = None

        # Compute norm for butterfly plot
        self.butterfly_data = np.linalg.norm(self.glass_brain_data, axis=1)

    def _load_ndvar_data(self, y):
        """Load data from NDVar directly.

        Parameters
        ----------
        y : NDVar
            Data with dimensions ([case,] time, source[, space]).
        """
        if y.has_case:
            y = y.mean('case')
        
        # Extract source dimension info
        source = y.get_dim('source')
        self.source_coords = source.coordinates
        self.time_values = y.time.times
        
        # Store source space info
        self.source_space = source
        if hasattr(self.source_space, 'parc'):
            self.parcellation = self.source_space.parc
            self.region_of_brain = str(self.parcellation)
        else:
            self.parcellation = None
            self.region_of_brain = 'Full Brain'

        # Handle space dimension (vector data)
        if y.has_dim('space'):
            # Extract 3D vector data
            self.glass_brain_data = y.get_data(('source', 'space', 'time'))  # (n_sources, 3, n_times)
            # Compute norm for butterfly plot
            self.butterfly_data = np.linalg.norm(self.glass_brain_data, axis=1)
        else:
            # Scalar data - no space dimension
            self.glass_brain_data = y.get_data(('source', 'time'))  # (n_sources, n_times)
            self.butterfly_data = self.glass_brain_data.copy()
            # Expand to 3D for consistency (assuming scalar represents magnitude)
            self.glass_brain_data = self.glass_brain_data[:, np.newaxis, :]  # (n_sources, 1, n_times)

    def _setup_layout(self):
        """Setup the Dash app layout."""
        # Create initial figures
        initial_butterfly = self.create_butterfly_plot(0)
        initial_brain_plots = self.create_2d_brain_projections_plotly(0)

        self.app.layout = html.Div([
            html.H1("Eelbrain Plotly 2D Brain Visualization",
                    style={'textAlign': 'center'}),

            # Hidden stores for state management
            dcc.Store(id='selected-time-idx', data=0),
            dcc.Store(id='selected-source-idx', data=None),

            # Main content - arranged vertically
            html.Div([
                # Top: Butterfly plot
                html.Div([
                    html.H3("Butterfly Plot"),
                    dcc.Graph(id='butterfly-plot', figure=initial_butterfly)
                ], style={'width': '100%', 'margin-bottom': '20px'}),

                # Bottom: 2D Brain projections using Plotly
                html.Div([
                    html.H3("2D Brain Projections"),

                    # Three brain view plots
                    html.Div([
                        html.Div([
                            html.H4("Axial (Z)", style={'textAlign': 'center', 'margin': '5px'}),
                            dcc.Graph(id='brain-axial-plot', figure=initial_brain_plots['axial'],
                                      style={'height': '300px'})
                        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%'}),

                        html.Div([
                            html.H4("Sagittal (X)", style={'textAlign': 'center', 'margin': '5px'}),
                            dcc.Graph(id='brain-sagittal-plot', figure=initial_brain_plots['sagittal'],
                                      style={'height': '300px'})
                        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%'}),

                        html.Div([
                            html.H4("Coronal (Y)", style={'textAlign': 'center', 'margin': '5px'}),
                            dcc.Graph(id='brain-coronal-plot', figure=initial_brain_plots['coronal'],
                                      style={'height': '300px'})
                        ], style={'width': '32%', 'display': 'inline-block', 'margin': '0.5%'}),
                    ], style={'textAlign': 'center'}),

                    # Status indicator
                    html.Div(id='update-status',
                             children="Click on butterfly plot to update brain views",
                             style={'textAlign': 'center', 'padding': '10px', 'fontStyle': 'italic', 'color': '#666'})
                ], style={'width': '100%'}),
            ]),

            # Info panel
            html.Div(id='info-panel', style={'clear': 'both', 'padding': '20px'})
        ])

    def _setup_callbacks(self):
        """Setup all Dash callbacks."""

        @self.app.callback(
            Output('butterfly-plot', 'figure'),
            Input('selected-time-idx', 'data')
        )
        def update_butterfly(time_idx):
            if time_idx is None:
                time_idx = 0
            return self.create_butterfly_plot(time_idx)

        @self.app.callback(
            [Output('brain-axial-plot', 'figure'),
             Output('brain-sagittal-plot', 'figure'),
             Output('brain-coronal-plot', 'figure')],
            Input('selected-time-idx', 'data'),
            Input('selected-source-idx', 'data')
        )
        def update_brain_projections(time_idx, source_idx):
            if time_idx is None:
                time_idx = 0

            try:
                brain_plots = self.create_2d_brain_projections_plotly(time_idx, source_idx)
                return brain_plots['axial'], brain_plots['sagittal'], brain_plots['coronal']
            except Exception:
                # Return empty plots on error
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Error loading brain data",
                                         xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                return empty_fig, empty_fig, empty_fig

        @self.app.callback(
            Output('selected-time-idx', 'data'),
            Output('selected-source-idx', 'data'),
            Input('butterfly-plot', 'clickData'),
            State('selected-time-idx', 'data')
        )
        def handle_butterfly_click(click_data, current_time_idx):
            if not click_data or self.time_values is None:
                return dash.no_update, dash.no_update

            point = click_data['points'][0]

            # Get clicked time
            clicked_time = point['x']
            time_idx = np.argmin(np.abs(self.time_values - clicked_time))

            # Get clicked source
            source_idx = point.get('customdata', None)

            return time_idx, source_idx

        @self.app.callback(
            Output('update-status', 'children'),
            Output('update-status', 'style'),
            Input('selected-time-idx', 'data')
        )
        def update_status(time_idx):
            if time_idx is not None and self.time_values is not None and 0 <= time_idx < len(self.time_values):
                time_val = self.time_values[time_idx]
                status_text = f"Brain views updated for time: {time_val:.3f}s (index {time_idx})"
                status_style = {
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontStyle': 'italic',
                    'color': '#2E8B57',
                    'backgroundColor': '#F0FFF0'
                }
            else:
                status_text = "Click on butterfly plot to update brain views"
                status_style = {'textAlign': 'center', 'padding': '10px', 'fontStyle': 'italic', 'color': '#666'}

            return status_text, status_style

        @self.app.callback(
            Output('info-panel', 'children'),
            Input('selected-time-idx', 'data'),
            Input('selected-source-idx', 'data')
        )
        def update_info(time_idx, source_idx):
            info = []

            if self.time_values is not None and time_idx is not None and 0 <= time_idx < len(self.time_values):
                info.append(f"Time: {self.time_values[time_idx]:.3f} s (index {time_idx})")

            if source_idx is not None and self.source_coords is not None and 0 <= source_idx < len(self.source_coords):
                coord = self.source_coords[source_idx]
                info.append(f"Selected source: {source_idx}")
                info.append(f"Coordinates: ({coord[0]:.3f}, {coord[1]:.3f}, {coord[2]:.3f}) m")

            result = html.P(' | '.join(info)) if info else html.P("Click on the plots to interact")
            return result

    def create_butterfly_plot(self, selected_time_idx=0):
        """Create butterfly plot figure."""
        fig = go.Figure()

        if self.butterfly_data is None or self.time_values is None:
            fig.add_annotation(text="No data loaded", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

        n_sources, n_times = self.butterfly_data.shape

        # Auto-scale data for visibility
        data_to_plot = self.butterfly_data.copy()
        scale_factor = 1.0
        unit_suffix = ""

        max_abs_val = np.max(np.abs(data_to_plot))
        if max_abs_val < 1e-10:
            scale_factor = 1e12
            unit_suffix = " (pA)"
        elif max_abs_val < 1e-6:
            scale_factor = 1e9
            unit_suffix = " (nA)"
        elif max_abs_val < 1e-3:
            scale_factor = 1e6
            unit_suffix = " (µA)"

        data_to_plot = data_to_plot * scale_factor

        # Add individual source traces only if show_max_only is False
        if not self.show_max_only:
            # Plot subset of traces for performance
            max_traces = 20
            step = max(1, n_sources // max_traces)
            indices_to_plot = list(range(0, n_sources, step))

            # Add individual traces
            for idx, i in enumerate(indices_to_plot[:10]):
                trace_data = data_to_plot[i, :]

                fig.add_trace(go.Scatter(
                    x=self.time_values,
                    y=trace_data,
                    mode='lines',
                    name=f'Source {i}',
                    customdata=[i] * n_times,
                    showlegend=(idx < 3),
                    opacity=0.6,
                    line=dict(width=1)
                ))

        # Add mean trace (always shown)
        mean_activity = np.mean(data_to_plot, axis=0)
        fig.add_trace(go.Scatter(
            x=self.time_values,
            y=mean_activity,
            mode='lines',
            name='Mean Activity',
            line=dict(color='red', width=3),
            showlegend=True
        ))

        # Add max trace (always shown)
        max_activity = np.max(data_to_plot, axis=0)
        fig.add_trace(go.Scatter(
            x=self.time_values,
            y=max_activity,
            mode='lines',
            name='Max Activity',
            line=dict(color='darkblue', width=3),
            showlegend=True
        ))

        # Add vertical line for selected time
        if 0 <= selected_time_idx < len(self.time_values):
            selected_time = self.time_values[selected_time_idx]
            fig.add_vline(x=selected_time, line_width=2, line_dash="dash", line_color="blue")

        # Set layout
        y_min, y_max = data_to_plot.min(), data_to_plot.max()
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1

        # Update title based on display mode
        if self.show_max_only:
            title_text = f"Source Activity Time Series - Mean & Max Only ({n_sources} sources)"
        else:
            title_text = f"Source Activity Time Series (showing subset of {n_sources} sources)"

        fig.update_layout(
            title=title_text,
            xaxis_title="Time (s)",
            yaxis_title=f"Activity{unit_suffix}",
            yaxis=dict(range=[y_min - y_margin, y_max + y_margin]),
            hovermode='closest',
            height=500,
            showlegend=True
        )

        return fig

    def create_2d_brain_projections_plotly(self, time_idx=0, source_idx=None):
        """Create 2D brain projections using Plotly scatter plots."""
        if self.glass_brain_data is None or self.source_coords is None or self.time_values is None:
            placeholder_fig = go.Figure()
            placeholder_fig.add_annotation(text="No brain data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return {
                'axial': placeholder_fig,
                'sagittal': placeholder_fig,
                'coronal': placeholder_fig
            }

        try:
            # Get time slice of data
            if time_idx >= len(self.time_values):
                time_idx = 0

            time_value = self.time_values[time_idx]

            # Get activity at this time point
            if self.glass_brain_data.ndim == 3:  # (n_sources, 3, n_times)
                time_activity = self.glass_brain_data[:, :, time_idx]  # (n_sources, 3)
                activity_magnitude = np.linalg.norm(time_activity, axis=1)  # (n_sources,)
            else:  # (n_sources, n_times)
                activity_magnitude = self.glass_brain_data[:, time_idx]

            # Create brain projections
            brain_plots = {}
            views = ['axial', 'sagittal', 'coronal']

            for view_name in views:
                try:
                    brain_fig = self._create_plotly_brain_projection(view_name, self.source_coords, activity_magnitude, time_value, source_idx)
                    brain_plots[view_name] = brain_fig
                except Exception:
                    brain_plots[view_name] = go.Figure()
                    brain_plots[view_name].add_annotation(text=f"Error: {view_name}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

            return brain_plots

        except Exception:
            placeholder_fig = go.Figure()
            placeholder_fig.add_annotation(text="No brain data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return {
                'axial': placeholder_fig,
                'sagittal': placeholder_fig,
                'coronal': placeholder_fig
            }

    def _create_plotly_brain_projection(self, view_name, coords, activity, time_value, selected_source=None):
        """Create a Plotly plot for a specific brain view with vector arrows."""
        # Show all data without filtering
        active_coords = coords
        active_activity = activity
        active_indices = np.arange(len(coords))

        # Create Plotly figure
        fig = go.Figure()

        # Get time index for vector components
        time_idx = np.argmin(np.abs(self.time_values - time_value))

        # Get 3D vector components for active sources
        if self.glass_brain_data is not None and len(active_indices) > 0:
            active_vectors = self.glass_brain_data[active_indices, :, time_idx]  # (n_active, 3)
        else:
            active_vectors = None

        # Project to 2D based on view
        if view_name == 'axial':  # Z view (X vs Y)
            x_coords = active_coords[:, 0]
            y_coords = active_coords[:, 1]
            if active_vectors is not None:
                u_vectors = active_vectors[:, 0]  # X components
                v_vectors = active_vectors[:, 1]  # Y components
            title = f'Axial View (Z) - {self.region_of_brain}'
            xlabel, ylabel = 'X (m)', 'Y (m)'
        elif view_name == 'sagittal':  # X view (Y vs Z)
            x_coords = active_coords[:, 1]
            y_coords = active_coords[:, 2]
            if active_vectors is not None:
                u_vectors = active_vectors[:, 1]  # Y components
                v_vectors = active_vectors[:, 2]  # Z components
            title = f'Sagittal View (X) - {self.region_of_brain}'
            xlabel, ylabel = 'Y (m)', 'Z (m)'
        elif view_name == 'coronal':  # Y view (X vs Z)
            x_coords = active_coords[:, 0]
            y_coords = active_coords[:, 2]
            if active_vectors is not None:
                u_vectors = active_vectors[:, 0]  # X components
                v_vectors = active_vectors[:, 2]  # Z components
            title = f'Coronal View (Y) - {self.region_of_brain}'
            xlabel, ylabel = 'X (m)', 'Z (m)'

        if len(active_coords) > 0:
            # Create data-driven grid using unique coordinate values
            unique_x = np.unique(x_coords)
            unique_y = np.unique(y_coords)

            # Create grid boundaries around each unique coordinate point
            x_spacing = np.diff(unique_x).min() / 2 if len(unique_x) > 1 else 0.001
            y_spacing = np.diff(unique_y).min() / 2 if len(unique_y) > 1 else 0.001

            # Create grid boundaries: small intervals around each data point
            x_edges = []
            for i, x_val in enumerate(unique_x):
                if i == 0:
                    x_edges.append(x_val - x_spacing)
                x_edges.append(x_val + x_spacing)

            y_edges = []
            for i, y_val in enumerate(unique_y):
                if i == 0:
                    y_edges.append(y_val - y_spacing)
                y_edges.append(y_val + y_spacing)

            x_edges = np.array(x_edges)
            y_edges = np.array(y_edges)

            # Create 2D histogram - now each data point falls into its own grid cell
            H, x_edges_used, y_edges_used = np.histogram2d(x_coords, y_coords,
                                                           bins=[x_edges, y_edges],
                                                           weights=active_activity)

            # Use grid center points for display
            x_centers = (x_edges_used[:-1] + x_edges_used[1:]) / 2
            y_centers = (y_edges_used[:-1] + y_edges_used[1:]) / 2

            # Set zero values to NaN to make them transparent in heatmap
            H_display = H.copy()
            H_display[H_display == 0] = np.nan

            # Add heatmap trace
            fig.add_trace(go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=H_display.T,  # Transpose to match Plotly orientation
                colorscale=self.cmap,
                colorbar=dict(title="Activity Magnitude"),
                showscale=True,
                hovertemplate=f'{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<br>Activity: %{{z:.2e}}<extra></extra>'
            ))

            # Update layout to have white background
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white'
            )

            # Add vector arrows if we have vector data
            if active_vectors is not None:
                arrow_scale = 0.008  # Scale arrows for visibility
                max_arrows = 50  # Limit number of arrows for performance
                step = max(1, len(active_coords) // max_arrows)

                for i in range(0, len(active_coords), step):
                    x_start = x_coords[i]
                    y_start = y_coords[i]
                    x_end = x_start + u_vectors[i] * arrow_scale
                    y_end = y_start + v_vectors[i] * arrow_scale

                    # Add arrow annotation
                    fig.add_annotation(
                        x=x_end, y=y_end,
                        ax=x_start, ay=y_start,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=0.8,
                        arrowwidth=1,
                        arrowcolor='black',
                        text="",
                        opacity=0.6
                    )

            # Highlight selected source if provided
            if selected_source is not None and selected_source in active_indices:
                selected_pos = np.where(active_indices == selected_source)[0]
                if len(selected_pos) > 0:
                    pos = selected_pos[0]
                    fig.add_trace(go.Scatter(
                        x=[x_coords[pos]],
                        y=[y_coords[pos]],
                        mode='markers',
                        marker=dict(size=12, color='cyan', symbol='circle-open', line=dict(width=3)),
                        name='Selected Source',
                        showlegend=False,
                        hovertemplate=f'SELECTED SOURCE<br>{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>'
                    ))

                    # Highlight selected source arrow if vectors available
                    if active_vectors is not None:
                        x_start = x_coords[pos]
                        y_start = y_coords[pos]
                        x_end = x_start + u_vectors[pos] * arrow_scale
                        y_end = y_start + v_vectors[pos] * arrow_scale

                        # Add highlighted arrow for selected source
                        fig.add_annotation(
                            x=x_end, y=y_end,
                            ax=x_start, ay=y_start,
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1.0,
                            arrowwidth=2,
                            arrowcolor='cyan',
                            text=""
                        )
        else:
            # Add annotation if no active sources
            fig.add_annotation(text=f"No active sources for {view_name} view",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False
        )

        return fig

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for Dash display."""
        try:
            # Save figure to bytes buffer
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
            img_buffer.seek(0)

            # Convert to base64 string
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            img_buffer.close()

            return f"data:image/png;base64,{img_base64}"

        except Exception:
            return self._create_placeholder_image("Conversion Error")

    def _create_placeholder_image(self, text="No Data"):
        """Create a placeholder image when brain plotting fails."""

        # Create a simple matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)

        return img_base64

    def run(self, port=None, debug=True):
        """Run the Dash app."""
        if port is None:
            port = random.randint(8001, 9001)

        print(f"\nStarting 2D Brain Visualization Dash app on port {port}...")
        print(f"Open http://127.0.0.1:{port}/ in your browser\n")

        self.app.run(debug=debug, port=port)

    def export_images(self, output_dir="./images", time_idx=None, format="png"):
        """Export current plots as image files.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save image files. Default is "./images".
        time_idx : int, optional
            Time index to export. If None, uses 0.
        format : str, optional
            Image format ('png', 'jpg', 'svg', 'pdf'). Default is 'png'.

        Returns
        -------
        dict
            Dictionary with exported file paths and status.
        """
        import os
        from datetime import datetime

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use provided time_idx or default to 0
        if time_idx is None:
            time_idx = 0

        # Timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = {}

        try:
            # Export butterfly plot
            butterfly_fig = self.create_butterfly_plot(time_idx)
            butterfly_path = os.path.join(output_dir, f"butterfly_plot_{timestamp}.{format}")
            butterfly_fig.write_image(butterfly_path, width=1200, height=600)
            exported_files['butterfly_plot'] = butterfly_path

            # Export brain projections
            brain_plots = self.create_2d_brain_projections_plotly(time_idx)

            for view_name, fig in brain_plots.items():
                brain_path = os.path.join(output_dir, f"{view_name}_view_{timestamp}.{format}")
                fig.write_image(brain_path, width=800, height=600)
                exported_files[f'{view_name}_view'] = brain_path

            print(f"✓ Successfully exported {len(exported_files)} image files to {output_dir}")
            for file_type, path in exported_files.items():
                print(f"  - {file_type}: {path}")

            return {"status": "success", "files": exported_files}

        except Exception as e:
            error_msg = f"Image export failed: {str(e)}"
            print(f"✗ {error_msg}")
            return {"status": "error", "message": error_msg, "files": exported_files}


# Run the app when script is executed directly
if __name__ == '__main__':
    try:
        # Example cmap options:
        # cmap = 'Hot'           # Black → Red → Yellow → White
        # cmap = 'YlOrRd'        # Yellow → Orange → Red
        # cmap = 'Viridis'       # Purple → Blue → Green → Yellow

        # Custom cmap example
        cmap = [
            [0, 'rgba(255,255,0,0.5)'],    # Yellow with 50% transparency
            [0.5, 'rgba(255,165,0,0.8)'],  # Orange with 80% transparency
            [1, 'rgba(255,0,0,1.0)']       # Red with full opacity
        ]

        # Butterfly plot display options:
        # show_max_only=False: Shows individual source traces + mean + max (default)
        # show_max_only=True:  Shows only mean + max traces (cleaner view)

        # Method 1: Pass data directly using y parameter (similar to plot.GlassBrain.butterfly)
        # from eelbrain import datasets
        #
        # # Load data
        # data_ds = datasets.get_mne_sample(src='vol', ori='vector')
        # y = data_ds['src']  # NDVar with dimensions (case, time, source, space)
        #
        # # Create visualization with direct data
        # viz_2d = EelbrainPlotly2DViz(
        #     y=y,  # Pass NDVar directly
        #     cmap=cmap,
        #     show_max_only=False
        # )

        # Method 2: Use the original approach with data_source_location and region
        viz_2d = EelbrainPlotly2DViz(
            region='aparc+aseg',
            cmap=cmap,
            show_max_only=False
        )

        # Example: Export plot images
        # Uncomment the lines below to export images before running the app:
        # result = viz_2d.export_images(
        #     output_dir="./brain_images",
        #     time_idx=10,  # Export plots for time index 10
        #     format="png"  # Can be 'png', 'jpg', 'svg', 'pdf'
        # )
        # print("Export result:", result)

        viz_2d.run()

    except Exception as e:
        print(f"Error starting 2D visualization app: {e}")
        import traceback
        traceback.print_exc()

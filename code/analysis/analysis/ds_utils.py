"""
Title:
    ds_utils.py
Creation Date:
    2016-10-15
Author(s):
    Manuel Razo-Mejia, Griffin Chure
Purpose:
    This file contains two functions which allow for simple utilization
    of Bokeh/DataShader visualation software for interpretation of
    flow-cytometry data.
Notes:
    These functions have not been unit tested nor will be. As they are only
    used for simple generation of plotting objects, we do not believe that
    errors in these functions have impacted our conclusions and inferences
    from the data.
License:
    MIT. See `mwc_induction_utils.py` for detailed licensing information.
"""


def base_plot(df, x_col, y_col, log=False):
    # Define the range to plot chekcing if it is a log scale or not
    if log:
        x_range = (np.min(np.log10(df[x_col])),
                   np.max(np.log10(df[x_col])))
        y_range = (np.min(np.log10(df[y_col])),
                   np.max(np.log10(df[y_col])))
    else:
        x_range = (df[x_col].min(), df[x_col].max())
        y_range = (df[y_col].min(), df[y_col].max())

    # Initialize the Bokeh plot
    p = bokeh.plotting.figure(
        x_range=x_range,
        y_range=y_range,
        tools='save,pan,wheel_zoom,box_zoom,reset',
        plot_width=500,
        plot_height=500)

    # Add all the features to the plot
    p.xgrid.grid_line_color = '#a6a6a6'
    p.ygrid.grid_line_color = '#a6a6a6'
    p.ygrid.grid_line_dash = [6, 4]
    p.xgrid.grid_line_dash = [6, 4]
    p.xaxis.axis_label = x_col
    p.yaxis.axis_label = y_col
    p.xaxis.axis_label_text_font_size = '15pt'
    p.yaxis.axis_label_text_font_size = '15pt'
    p.background_fill_color = '#F4F3F6'
    return p


# #################
def ds_plot(df, x_col, y_col, log=False):
    if log:
        data = np.log10(df[[x_col, y_col]])
    else:
        data = df[[x_col, y_col]]
    p = base_plot(data, x_col, y_col)
    pipeline = ds.Pipeline(data, ds.Point(x_col, y_col))
    return p, pipeline

from dash import dcc
from dash.exceptions import PreventUpdate


def handle_template_or_list_download(
    ctx,
    template_trigger_id,
    list_trigger_id,
    template_payload,
    template_filename,
    list_df_builder,
    list_filename,
):
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == template_trigger_id:
        if callable(template_payload):
            try:
                df = template_payload()
                return dcc.send_data_frame(df.to_csv, template_filename, index=False)
            except Exception as e:
                # Fallback or error handling if needed, but for now let it propagate or return None
                 return None
        return dcc.send_string(template_payload, template_filename)
    if trigger == list_trigger_id:
        df = list_df_builder()
        return dcc.send_data_frame(df.to_csv, list_filename, index=False)

    raise PreventUpdate

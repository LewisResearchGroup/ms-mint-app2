import uuid
import dash_bootstrap_components as dbc


def create_toast(message, title="Notification", icon="info"):
    return dbc.Toast(
                message,
                id=str(uuid.uuid4()),
                header=title,
                icon=icon,
                dismissable=True,
                is_open=True,
                duration=5000,
                style = {"marginBottom": "0.5rem", "minWidth": "250px", 'fontSize': '14px'}
            )

import uuid

def create_toast(message, title="Notification", icon="primary"):
    return {
        "id": str(uuid.uuid4()),
        "header": title,
        "message": message,
        "icon": icon
    }
from .base import BaseConnector


def load_data(uri: str, **kwargs):
    """
    Convenience function: picks the right connector,
    loads the data, and returns a DataFrame.
    """
    conn_cls = BaseConnector.get_connector_for(uri)
    connector = conn_cls()
    return connector.load(uri, **kwargs)
